from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from backtest_engine import performance_stats, run_weight_execution_engine
from plot_backtest_pnl import save_pnl_plot
from run_news_head_backtest import (
    ALLOWED_EXCHANGES,
    SymbolState,
    build_ret_and_elig_matrices,
    build_symbol_states,
    parse_news_timestamps_utc,
    yyyymmdd_to_ts,
)

SYMBOL_STATE_CACHE_VERSION = 4
SIZE_BUCKET_CACHE_VERSION = 2


class NullProgressBar:
    def update(self, _: int = 1) -> None:
        pass

    def set_postfix(self, **_: object) -> None:
        pass

    def close(self) -> None:
        pass


def make_progress_bar(total: int | None, desc: str, unit: str):
    if tqdm is None:
        return NullProgressBar()
    return tqdm(total=total, desc=desc, unit=unit, leave=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load LLM news responses, aggregate them into daily stock signals, and run backtests via backtest_engine.py."
        )
    )
    parser.add_argument("--response-dir", required=True, help="Directory with responses_*.parquet files.")
    parser.add_argument("--ohlcv-dir", required=True, help="Directory produced by fetch_eodhd_ohlcv_by_ticker.py.")
    parser.add_argument("--output-dir", required=True, help="Output directory for aggregated signals and backtest files.")
    parser.add_argument("--baseline-ohlcv-path", default=None, help="Optional OHLCV parquet path for plotting a baseline.")
    parser.add_argument(
        "--market-cap-dir",
        default=None,
        help=(
            "Optional directory produced by fetch_yfinance_historical_market_cap.py. "
            "If provided, the script also builds daily small/non-small size buckets, "
            "cached value-weighted baselines, and size-filtered strategy variants."
        ),
    )
    parser.add_argument(
        "--size-breakpoint-quantile",
        type=float,
        default=0.2,
        help="NYSE market-cap breakpoint quantile used to define daily small vs non-small buckets. Default: 0.2",
    )
    parser.add_argument("--start-date", default="20230719", help="Backtest start date, inclusive, YYYYMMDD.")
    parser.add_argument("--end-date", default="20260306", help="Backtest end date, inclusive, YYYYMMDD.")
    parser.add_argument("--min-price", type=float, default=1.0, help="Signal filter uses close(t-1) >= min-price.")
    parser.add_argument(
        "--min-adv-usd",
        type=float,
        default=1_000_000.0,
        help="Signal filter uses ADV20(t-1) >= this dollar volume threshold.",
    )
    parser.add_argument("--cost-bps", type=float, default=3.0, help="Transaction cost in bps.")
    parser.add_argument("--min-news-pool", type=int, default=10, help="Skip date if eligible news pool smaller than this.")
    parser.add_argument(
        "--min-short-count",
        type=int,
        default=1,
        help=(
            "For short-enabled strategies, require at least this many short candidates before opening any positions "
            "on the date. Default: 1"
        ),
    )
    parser.add_argument("--max-response-rows", type=int, default=0, help="Debug cap on response rows processed. 0 means no cap.")
    parser.add_argument(
        "--require-parsed-ok",
        action="store_true",
        help="If set, only use rows where response_parsed_ok is true. Default uses all rows and trusts stored response_label.",
    )
    parser.add_argument(
        "--overnight-news",
        action="store_true",
        help="If set, only use news between the previous trading day's close and the current trading day 9:00 America/New_York cutoff.",
    )
    return parser.parse_args()


def discover_response_files(response_dir: Path) -> list[Path]:
    shard_dirs = sorted(p for p in response_dir.iterdir() if p.is_dir() and p.name.startswith("shard_"))
    roots = shard_dirs or [response_dir]
    out: list[Path] = []
    for root in roots:
        out.extend(sorted(root.glob("responses_*.parquet")))
    if not out:
        raise FileNotFoundError(f"No responses_*.parquet files found under {response_dir}")
    return out


def collect_response_symbols(response_files: list[Path]) -> set[str]:
    symbols: set[str] = set()
    progress = make_progress_bar(total=len(response_files), desc="Response symbols", unit="file")
    try:
        for resp_path in response_files:
            df = pd.read_parquet(resp_path, columns=["symbol"])
            if not df.empty:
                vals = df["symbol"].fillna("").astype(str).str.upper()
                for sym in vals.unique().tolist():
                    if sym:
                        symbols.add(sym)
            progress.update(1)
    finally:
        progress.close()
    return symbols


def default_cache_dir() -> Path:
    return Path("/home/jyu197/LLMCrossSec/.cache/run_llm_response_backtest")


def trim_symbol_state(st: SymbolState, start_date: int) -> SymbolState | None:
    mask = st.trade_dates_int >= int(start_date)
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return SymbolState(
        symbol=st.symbol,
        trade_dates_int=st.trade_dates_int[idx],
        cutoff_ns=st.cutoff_ns[idx],
        signal_window_open_ns=st.signal_window_open_ns[idx],
        overnight_window_open_ns=st.overnight_window_open_ns[idx],
        ret_s=st.ret_s.iloc[idx].copy(),
        signal_elig_s=st.signal_elig_s.iloc[idx].copy(),
    )


def build_symbol_state_cache_key(
    *,
    ohlcv_dir: Path,
    min_price: float,
    min_adv_usd: float,
    start_date: int,
    response_symbols: set[str],
) -> str:
    payload = {
        "version": SYMBOL_STATE_CACHE_VERSION,
        "ohlcv_dir": str(ohlcv_dir.resolve()),
        "min_price": float(min_price),
        "min_adv_usd": float(min_adv_usd),
        "start_date": int(start_date),
        "response_symbols": sorted(response_symbols),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def load_or_build_symbol_state_cache(
    *,
    ohlcv_dir: Path,
    min_price: float,
    min_adv_usd: float,
    start_date: int,
    response_symbols: set[str],
) -> dict[str, SymbolState]:
    cache_dir = default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = build_symbol_state_cache_key(
        ohlcv_dir=ohlcv_dir,
        min_price=min_price,
        min_adv_usd=min_adv_usd,
        start_date=start_date,
        response_symbols=response_symbols,
    )
    cache_path = cache_dir / f"symbol_states_{cache_key}.pkl"
    meta_path = cache_dir / f"symbol_states_{cache_key}.json"

    if cache_path.is_file():
        print(f"[cache] loading symbol states from {cache_path}")
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    print("[cache] miss, building symbol states")
    symbol_states = build_symbol_states(
        ohlcv_dir=ohlcv_dir,
        min_price=float(min_price),
        min_adv_usd=float(min_adv_usd),
        allowed_symbols=response_symbols,
    )
    trimmed_states: dict[str, SymbolState] = {}
    trim_progress = make_progress_bar(total=len(symbol_states), desc="Trim symbol states", unit="alias")
    try:
        for alias, st in symbol_states.items():
            trimmed = trim_symbol_state(st, start_date=start_date)
            if trimmed is not None:
                trimmed_states[alias] = trimmed
            trim_progress.update(1)
    finally:
        trim_progress.close()

    with cache_path.open("wb") as handle:
        pickle.dump(trimmed_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "version": SYMBOL_STATE_CACHE_VERSION,
        "cache_path": str(cache_path),
        "ohlcv_dir": str(ohlcv_dir.resolve()),
        "min_price": float(min_price),
        "min_adv_usd": float(min_adv_usd),
        "start_date": int(start_date),
        "response_symbol_count": int(len(response_symbols)),
        "symbol_state_alias_count": int(len(trimmed_states)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[cache] saved symbol states to {cache_path}")
    return trimmed_states


def discover_market_cap_files(market_cap_dir: Path) -> list[Path]:
    files = sorted((market_cap_dir / "parquet").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No market-cap parquet files found under {market_cap_dir}")
    return files


def market_cap_symbol_from_path(path: Path) -> str:
    return path.stem.upper()


def extract_symbol_base(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        return text.rsplit(".", 1)[0].strip().upper()
    return text


def market_cap_exchange_from_path(path: Path) -> str:
    return path.parent.name.strip().upper()


def load_prev_market_cap_series(path: Path, start_date: int, end_date: int) -> pd.Series:
    df = pd.read_parquet(path, columns=["date", "market_cap"])
    if df.empty:
        return pd.Series(dtype="float64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df = df.dropna(subset=["date", "market_cap"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return pd.Series(dtype="float64")
    df["prev_market_cap"] = df["market_cap"].shift(1)
    date_mask = (df["date"] >= yyyymmdd_to_ts(int(start_date))) & (df["date"] <= yyyymmdd_to_ts(int(end_date)))
    s = df.loc[date_mask, ["date", "prev_market_cap"]].set_index("date")["prev_market_cap"].astype("float64")
    s = s[np.isfinite(s) & (s > 0)]
    return s.sort_index()


def build_size_bucket_cache_key(
    *,
    market_cap_dir: Path,
    ohlcv_dir: Path,
    min_price: float,
    min_adv_usd: float,
    start_date: int,
    end_date: int,
    size_breakpoint_quantile: float,
    response_symbols: set[str],
) -> str:
    payload = {
        "version": SIZE_BUCKET_CACHE_VERSION,
        "market_cap_dir": str(market_cap_dir.resolve()),
        "ohlcv_dir": str(ohlcv_dir.resolve()),
        "min_price": float(min_price),
        "min_adv_usd": float(min_adv_usd),
        "start_date": int(start_date),
        "end_date": int(end_date),
        "size_breakpoint_quantile": float(size_breakpoint_quantile),
        "response_symbols": sorted(response_symbols),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def build_bucket_baseline_frame(
    *,
    numer_by_date: dict[pd.Timestamp, float],
    denom_by_date: dict[pd.Timestamp, float],
    count_by_date: dict[pd.Timestamp, int],
) -> pd.DataFrame:
    rows = []
    equity = 1.0
    all_dates = sorted(set(numer_by_date) | set(denom_by_date) | set(count_by_date))
    for dt in all_dates:
        denom = float(denom_by_date.get(dt, 0.0))
        net_ret = float(numer_by_date.get(dt, 0.0) / denom) if denom > 0 else 0.0
        equity *= 1.0 + net_ret
        rows.append(
            {
                "trade_date": pd.Timestamp(dt),
                "net_ret": net_ret,
                "equity": equity,
                "n_constituents": int(count_by_date.get(dt, 0)),
                "gross_market_cap_weight": denom,
            }
        )
    return pd.DataFrame(rows)


def load_or_build_size_bucket_data(
    *,
    market_cap_dir: Path,
    ohlcv_dir: Path,
    min_price: float,
    min_adv_usd: float,
    start_date: int,
    end_date: int,
    size_breakpoint_quantile: float,
    response_symbols: set[str],
) -> dict[str, object]:
    cache_dir = default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = build_size_bucket_cache_key(
        market_cap_dir=market_cap_dir,
        ohlcv_dir=ohlcv_dir,
        min_price=min_price,
        min_adv_usd=min_adv_usd,
        start_date=start_date,
        end_date=end_date,
        size_breakpoint_quantile=size_breakpoint_quantile,
        response_symbols=response_symbols,
    )
    membership_path = cache_dir / f"size_membership_{cache_key}.parquet"
    breakpoints_path = cache_dir / f"size_breakpoints_{cache_key}.parquet"
    baseline_small_path = cache_dir / f"size_baseline_small_{cache_key}.parquet"
    baseline_non_small_path = cache_dir / f"size_baseline_non_small_{cache_key}.parquet"
    meta_path = cache_dir / f"size_bucket_meta_{cache_key}.json"

    if (
        membership_path.is_file()
        and breakpoints_path.is_file()
        and baseline_small_path.is_file()
        and baseline_non_small_path.is_file()
        and meta_path.is_file()
    ):
        print(f"[cache] loading size buckets from {membership_path}")
        return {
            "membership_df": pd.read_parquet(membership_path),
            "breakpoints_df": pd.read_parquet(breakpoints_path),
            "baseline_paths": {
                "small": str(baseline_small_path),
                "non_small": str(baseline_non_small_path),
            },
            "meta": json.loads(meta_path.read_text(encoding="utf-8")),
        }

    market_cap_files = discover_market_cap_files(market_cap_dir)
    market_cap_symbols = {market_cap_symbol_from_path(path) for path in market_cap_files}
    print("[size] loading symbol states for market-cap universe")
    baseline_symbol_states = load_or_build_symbol_state_cache(
        ohlcv_dir=ohlcv_dir,
        min_price=float(min_price),
        min_adv_usd=float(min_adv_usd),
        start_date=int(start_date),
        response_symbols=market_cap_symbols,
    )

    nyse_caps_by_date: dict[pd.Timestamp, list[float]] = {}
    bp_progress = make_progress_bar(total=len(market_cap_files), desc="NYSE size breakpoints", unit="file")
    try:
        for path in market_cap_files:
            if market_cap_exchange_from_path(path) != "NYSE":
                bp_progress.update(1)
                continue
            prev_mc = load_prev_market_cap_series(path, start_date=start_date, end_date=end_date)
            for dt, value in prev_mc.items():
                nyse_caps_by_date.setdefault(pd.Timestamp(dt), []).append(float(value))
            bp_progress.update(1)
    finally:
        bp_progress.close()

    breakpoint_by_date: dict[pd.Timestamp, float] = {}
    breakpoint_rows: list[dict[str, object]] = []
    for dt in sorted(nyse_caps_by_date):
        values = np.asarray(nyse_caps_by_date[dt], dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        if len(values) == 0:
            continue
        breakpoint = float(np.quantile(values, float(size_breakpoint_quantile)))
        breakpoint_by_date[pd.Timestamp(dt)] = breakpoint
        breakpoint_rows.append(
            {
                "trade_date": pd.Timestamp(dt),
                "nyse_market_cap_breakpoint": breakpoint,
                "n_nyse_symbols": int(len(values)),
                "quantile": float(size_breakpoint_quantile),
            }
        )
    if not breakpoint_rows:
        raise RuntimeError("No NYSE market-cap breakpoints could be constructed from the supplied market_cap_dir.")
    breakpoints_df = pd.DataFrame(breakpoint_rows)

    numer_by_bucket = {"small": {}, "non_small": {}}
    denom_by_bucket = {"small": {}, "non_small": {}}
    count_by_bucket = {"small": {}, "non_small": {}}
    response_rows: list[dict[str, object]] = []
    response_symbol_base_set = {extract_symbol_base(symbol) for symbol in response_symbols if extract_symbol_base(symbol)}

    membership_progress = make_progress_bar(total=len(market_cap_files), desc="Assign size buckets", unit="file")
    try:
        for path in market_cap_files:
            symbol = market_cap_symbol_from_path(path)
            symbol_base = extract_symbol_base(symbol)
            st = baseline_symbol_states.get(symbol)
            membership_progress.update(1)
            if st is None:
                continue

            prev_mc = load_prev_market_cap_series(path, start_date=start_date, end_date=end_date)
            if prev_mc.empty:
                continue

            aligned_prev_mc = prev_mc.reindex(st.ret_s.index)
            ret_s = st.ret_s.reindex(st.ret_s.index).astype("float64")
            elig_s = st.signal_elig_s.reindex(st.ret_s.index).fillna(False).astype(bool)

            for dt, trade_date_int, ret_value, elig_value, prev_mc_value in zip(
                st.ret_s.index,
                st.trade_dates_int,
                ret_s.to_numpy(dtype=np.float64),
                elig_s.to_numpy(dtype=bool),
                aligned_prev_mc.to_numpy(dtype=np.float64),
            ):
                breakpoint = breakpoint_by_date.get(pd.Timestamp(dt))
                if breakpoint is None or not np.isfinite(prev_mc_value) or prev_mc_value <= 0:
                    continue
                bucket = "small" if float(prev_mc_value) <= float(breakpoint) else "non_small"

                if symbol_base in response_symbol_base_set:
                    response_rows.append(
                        {
                            "trade_date": int(trade_date_int),
                            "symbol_base": symbol_base,
                            "size_bucket": bucket,
                            "prev_market_cap": float(prev_mc_value),
                            "nyse_market_cap_breakpoint": float(breakpoint),
                        }
                    )

                if bool(elig_value) and np.isfinite(ret_value):
                    numer_map = numer_by_bucket[bucket]
                    denom_map = denom_by_bucket[bucket]
                    count_map = count_by_bucket[bucket]
                    numer_map[dt] = float(numer_map.get(dt, 0.0) + (float(prev_mc_value) * float(ret_value)))
                    denom_map[dt] = float(denom_map.get(dt, 0.0) + float(prev_mc_value))
                    count_map[dt] = int(count_map.get(dt, 0) + 1)
    finally:
        membership_progress.close()

    membership_df = pd.DataFrame(response_rows)
    if membership_df.empty:
        membership_df = pd.DataFrame(
            columns=["trade_date", "symbol_base", "size_bucket", "prev_market_cap", "nyse_market_cap_breakpoint"]
        )
    else:
        membership_df = membership_df.sort_values(["trade_date", "symbol_base"]).drop_duplicates(
            subset=["trade_date", "symbol_base"], keep="last"
        )

    baseline_small_df = build_bucket_baseline_frame(
        numer_by_date=numer_by_bucket["small"],
        denom_by_date=denom_by_bucket["small"],
        count_by_date=count_by_bucket["small"],
    )
    baseline_non_small_df = build_bucket_baseline_frame(
        numer_by_date=numer_by_bucket["non_small"],
        denom_by_date=denom_by_bucket["non_small"],
        count_by_date=count_by_bucket["non_small"],
    )
    if baseline_small_df.empty or baseline_non_small_df.empty:
        raise RuntimeError("Failed to construct one or both size baselines.")

    membership_df.to_parquet(membership_path, index=False)
    breakpoints_df.to_parquet(breakpoints_path, index=False)
    baseline_small_df.to_parquet(baseline_small_path, index=False)
    baseline_non_small_df.to_parquet(baseline_non_small_path, index=False)
    meta = {
        "version": SIZE_BUCKET_CACHE_VERSION,
        "market_cap_dir": str(market_cap_dir.resolve()),
        "ohlcv_dir": str(ohlcv_dir.resolve()),
        "min_price": float(min_price),
        "min_adv_usd": float(min_adv_usd),
        "start_date": int(start_date),
        "end_date": int(end_date),
        "size_breakpoint_quantile": float(size_breakpoint_quantile),
        "market_cap_symbol_count": int(len(market_cap_symbols)),
        "market_cap_symbol_state_count": int(len(baseline_symbol_states)),
        "response_symbol_count": int(len(response_symbols)),
        "response_membership_rows": int(len(membership_df)),
        "breakpoint_rows": int(len(breakpoints_df)),
        "baseline_small_rows": int(len(baseline_small_df)),
        "baseline_non_small_rows": int(len(baseline_non_small_df)),
        "membership_path": str(membership_path),
        "breakpoints_path": str(breakpoints_path),
        "baseline_paths": {
            "small": str(baseline_small_path),
            "non_small": str(baseline_non_small_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[cache] saved size buckets to {membership_path}")
    return {
        "membership_df": membership_df,
        "breakpoints_df": breakpoints_df,
        "baseline_paths": {
            "small": str(baseline_small_path),
            "non_small": str(baseline_non_small_path),
        },
        "meta": meta,
    }


def in_bt_range(date_int: int, start_date: int, end_date: int) -> bool:
    return int(start_date) <= int(date_int) <= int(end_date)


def normalize_label(raw: object) -> str:
    label = str(raw or "").strip().upper()
    if label in {"YES", "NO", "UNKNOWN"}:
        return label
    return "UNKNOWN"


def normalize_signal_weights(raw_weights: pd.Series, symbols: pd.Index) -> pd.Series:
    w = pd.Series(0.0, index=symbols, dtype="float64")
    raw = pd.to_numeric(raw_weights, errors="coerce").astype("float64")
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    raw = raw[raw != 0.0]
    gross = float(raw.abs().sum())
    if gross > 0:
        w.loc[raw.index] = raw / gross
    return w


def aggregate_response_signals(
    *,
    response_files: list[Path],
    symbol_states: dict[str, SymbolState],
    start_date: int,
    end_date: int,
    max_response_rows: int,
    require_parsed_ok: bool,
    overnight_news: bool,
) -> pd.DataFrame:
    agg_counts: dict[tuple[int, str], np.ndarray] = {}
    processed = 0

    file_progress = make_progress_bar(total=len(response_files), desc="Response files", unit="file")
    try:
        for resp_path in response_files:
            cols = ["symbol", "date", "link", "response_label", "response_parsed_ok", "is_title_only"]
            df = pd.read_parquet(resp_path, columns=cols)
            file_progress.update(1)
            if df.empty:
                continue

            ts = parse_news_timestamps_utc(df["date"], df["link"])
            ts_ns = ts.view("int64").to_numpy()
            syms = df["symbol"].fillna("").astype(str).str.upper().to_numpy()
            labels = df["response_label"].to_numpy()
            parsed = df["response_parsed_ok"].fillna(False).astype(bool).to_numpy()
            title_only = df["is_title_only"].fillna(False).astype(bool).to_numpy()

            for i in range(len(df)):
                if require_parsed_ok and not parsed[i]:
                    continue
                symbol = syms[i]
                if not symbol:
                    continue
                st = symbol_states.get(symbol)
                if st is None:
                    continue
                trade_date_int = st.assign_trade_date_int(int(ts_ns[i]), overnight_only=overnight_news)
                if trade_date_int is None or not in_bt_range(int(trade_date_int), start_date, end_date):
                    continue

                key = (int(trade_date_int), symbol)
                counts = agg_counts.get(key)
                if counts is None:
                    counts = np.zeros(5, dtype=np.int32)
                    agg_counts[key] = counts

                label = normalize_label(labels[i])
                if label == "YES":
                    counts[0] += 1
                elif label == "NO":
                    counts[1] += 1
                else:
                    counts[2] += 1
                counts[3] += 1
                if title_only[i]:
                    counts[4] += 1

                processed += 1
                if processed % 5000 == 0:
                    file_progress.set_postfix(processed_rows=processed, stock_days=len(agg_counts))
                if max_response_rows > 0 and processed >= max_response_rows:
                    break

            if max_response_rows > 0 and processed >= max_response_rows:
                break
    finally:
        file_progress.close()

    rows = []
    for (trade_date_int, symbol), counts in agg_counts.items():
        rows.append(
            {
                "trade_date": int(trade_date_int),
                "symbol": symbol,
                "yes_count": int(counts[0]),
                "no_count": int(counts[1]),
                "unknown_count": int(counts[2]),
                "news_count": int(counts[3]),
                "title_only_count": int(counts[4]),
            }
        )
    if not rows:
        raise RuntimeError("No aggregated response rows generated. Check date range or input paths.")

    pred_df = pd.DataFrame(rows)
    pred_df["paper_score"] = pred_df["yes_count"] - pred_df["no_count"]
    pred_df["strict_long_candidate"] = (pred_df["yes_count"] > 0) & (pred_df["no_count"] == 0)
    pred_df["strict_short_candidate"] = (pred_df["no_count"] > 0) & (pred_df["yes_count"] == 0)

    elig_flags = []
    target_ret = []
    row_progress = make_progress_bar(total=len(pred_df), desc="Attach returns/eligibility", unit="row")
    try:
        for row in pred_df.itertuples(index=False):
            st = symbol_states[row.symbol]
            dt = yyyymmdd_to_ts(row.trade_date)
            elig = bool(st.signal_elig_s.get(dt, False))
            ret = st.ret_s.get(dt, np.nan)
            elig_flags.append(elig)
            target_ret.append(float(ret) if np.isfinite(ret) else np.nan)
            row_progress.update(1)
    finally:
        row_progress.close()

    pred_df["signal_eligible"] = np.asarray(elig_flags, dtype=bool)
    pred_df["target_return"] = np.asarray(target_ret, dtype=np.float64)
    pred_df = pred_df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return pred_df


def attach_size_bucket_membership(pred_df: pd.DataFrame, membership_df: pd.DataFrame) -> pd.DataFrame:
    pred_with_base = pred_df.copy()
    pred_with_base["symbol_base"] = pred_with_base["symbol"].map(extract_symbol_base)
    if membership_df.empty:
        out = pred_with_base.copy()
        out["size_bucket"] = pd.Series(pd.NA, index=out.index, dtype="object")
        out["prev_market_cap"] = np.nan
        out["nyse_market_cap_breakpoint"] = np.nan
        out = out.drop(columns=["symbol_base"])
        return out
    merged = pred_with_base.merge(
        membership_df,
        on=["trade_date", "symbol_base"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.drop(columns=["symbol_base"])
    return merged


def filter_pred_df_by_size_bucket(pred_df: pd.DataFrame, size_bucket: str | None) -> pd.DataFrame:
    if not size_bucket:
        return pred_df.copy()
    return pred_df[pred_df["size_bucket"] == size_bucket].copy()


def build_targets_strict_unknown_filtered(
    pred_df: pd.DataFrame,
    symbols: pd.Index,
    min_news_pool: int,
    min_short_count: int,
) -> tuple[dict[pd.Timestamp, pd.Series], dict[pd.Timestamp, dict], pd.Series]:
    target_by_date: dict[pd.Timestamp, pd.Series] = {}
    meta_by_date: dict[pd.Timestamp, dict] = {}
    rebalance_dates: list[pd.Timestamp] = []

    for trade_date_int, grp in pred_df.groupby("trade_date", sort=True):
        dt = yyyymmdd_to_ts(int(trade_date_int))
        eligible_grp = grp[grp["signal_eligible"]].copy()
        n_pool = len(eligible_grp)
        long_grp = eligible_grp[eligible_grp["strict_long_candidate"]]
        short_grp = eligible_grp[eligible_grp["strict_short_candidate"]]

        raw_weights = pd.Series(dtype="float64")
        if n_pool >= int(min_news_pool) and len(long_grp) > 0 and len(short_grp) >= int(min_short_count):
            long_strength = long_grp.set_index("symbol")["yes_count"].astype("float64")
            short_strength = -short_grp.set_index("symbol")["no_count"].astype("float64")
            raw_weights = pd.concat([long_strength, short_strength])
        w = normalize_signal_weights(raw_weights, symbols=symbols)

        target_by_date[dt] = w
        meta_by_date[dt] = {
            "signal_period_end": dt,
            "n_eligible": int(n_pool),
            "n_pb_pool": int(n_pool),
            "n_long": int(len(long_grp)),
            "n_short": int(len(short_grp)),
            "min_short_count": int(min_short_count),
            "weighting": "signal_strength_l1_normalized",
            "strategy": "strict_yes_or_unknown_vs_no_or_unknown",
        }
        rebalance_dates.append(dt)

    rebalance_mask = pd.Series(True, index=pd.DatetimeIndex(sorted(rebalance_dates)))
    return target_by_date, meta_by_date, rebalance_mask


def build_targets_paper_like(
    pred_df: pd.DataFrame,
    symbols: pd.Index,
    min_news_pool: int,
    min_short_count: int,
) -> tuple[dict[pd.Timestamp, pd.Series], dict[pd.Timestamp, dict], pd.Series]:
    target_by_date: dict[pd.Timestamp, pd.Series] = {}
    meta_by_date: dict[pd.Timestamp, dict] = {}
    rebalance_dates: list[pd.Timestamp] = []

    for trade_date_int, grp in pred_df.groupby("trade_date", sort=True):
        dt = yyyymmdd_to_ts(int(trade_date_int))
        eligible_grp = grp[grp["signal_eligible"]].copy()
        n_pool = len(eligible_grp)
        long_grp = eligible_grp[eligible_grp["paper_score"] > 0]
        short_grp = eligible_grp[eligible_grp["paper_score"] < 0]

        raw_weights = pd.Series(dtype="float64")
        if n_pool >= int(min_news_pool) and len(long_grp) > 0 and len(short_grp) >= int(min_short_count):
            raw_weights = eligible_grp.set_index("symbol")["paper_score"].astype("float64")
        w = normalize_signal_weights(raw_weights, symbols=symbols)

        target_by_date[dt] = w
        meta_by_date[dt] = {
            "signal_period_end": dt,
            "n_eligible": int(n_pool),
            "n_pb_pool": int(n_pool),
            "n_long": int(len(long_grp)),
            "n_short": int(len(short_grp)),
            "min_short_count": int(min_short_count),
            "weighting": "signal_strength_l1_normalized",
            "strategy": "paper_like_positive_negative",
        }
        rebalance_dates.append(dt)

    rebalance_mask = pd.Series(True, index=pd.DatetimeIndex(sorted(rebalance_dates)))
    return target_by_date, meta_by_date, rebalance_mask


def build_targets_long_only_yes_no_no(
    pred_df: pd.DataFrame,
    symbols: pd.Index,
    min_news_pool: int,
) -> tuple[dict[pd.Timestamp, pd.Series], dict[pd.Timestamp, dict], pd.Series]:
    target_by_date: dict[pd.Timestamp, pd.Series] = {}
    meta_by_date: dict[pd.Timestamp, dict] = {}
    rebalance_dates: list[pd.Timestamp] = []

    for trade_date_int, grp in pred_df.groupby("trade_date", sort=True):
        dt = yyyymmdd_to_ts(int(trade_date_int))
        eligible_grp = grp[grp["signal_eligible"]].copy()
        n_pool = len(eligible_grp)
        long_grp = eligible_grp[eligible_grp["strict_long_candidate"]]

        w = pd.Series(0.0, index=symbols, dtype="float64")
        if n_pool >= int(min_news_pool) and len(long_grp) > 0:
            w.loc[long_grp["symbol"].to_list()] = 1.0 / float(len(long_grp))

        target_by_date[dt] = w
        meta_by_date[dt] = {
            "signal_period_end": dt,
            "n_eligible": int(n_pool),
            "n_pb_pool": int(n_pool),
            "n_long": int(len(long_grp)),
            "n_short": 0,
            "strategy": "long_only_yes_no_no",
        }
        rebalance_dates.append(dt)

    rebalance_mask = pd.Series(True, index=pd.DatetimeIndex(sorted(rebalance_dates)))
    return target_by_date, meta_by_date, rebalance_mask


def run_strategy_backtest(
    *,
    strategy_name: str,
    strategy_kind: str,
    pred_df: pd.DataFrame,
    symbols: pd.Index,
    trade_dates: pd.DatetimeIndex,
    symbol_states: dict[str, SymbolState],
    out_dir: Path,
    cost_bps: float,
    min_news_pool: int,
    min_short_count: int,
    baseline_ohlcv_path: str | None,
    baseline_daily_path: str | None,
    baseline_label: str,
    secondary_baseline_ohlcv_path: str | None,
    secondary_baseline_daily_path: str | None,
    secondary_baseline_label: str,
) -> dict:
    print(f"[backtest] running strategy: {strategy_name}")
    if pred_df.empty or len(symbols) == 0 or len(trade_dates) == 0:
        return {
            "skipped": True,
            "reason": "No eligible rows remain after applying strategy universe filters.",
            "artifacts": {},
        }

    ret_d, elig_by_date = build_ret_and_elig_matrices(
        symbol_states=symbol_states,
        symbols=symbols,
        trade_dates=trade_dates,
    )
    cal = pd.DataFrame(index=trade_dates)
    cal["period_end"] = trade_dates

    if strategy_kind == "strict_yes_or_unknown_vs_no_or_unknown":
        target_by_date, meta_by_date, rebalance_mask = build_targets_strict_unknown_filtered(
            pred_df=pred_df,
            symbols=symbols,
            min_news_pool=min_news_pool,
            min_short_count=min_short_count,
        )
    elif strategy_kind == "paper_like_positive_negative":
        target_by_date, meta_by_date, rebalance_mask = build_targets_paper_like(
            pred_df=pred_df,
            symbols=symbols,
            min_news_pool=min_news_pool,
            min_short_count=min_short_count,
        )
    elif strategy_kind == "long_only_yes_no_no":
        target_by_date, meta_by_date, rebalance_mask = build_targets_long_only_yes_no_no(
            pred_df=pred_df,
            symbols=symbols,
            min_news_pool=min_news_pool,
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy_kind}")

    rebalance_info, daily_bt, w_daily = run_weight_execution_engine(
        ret_d=ret_d,
        cal=cal,
        symbols=symbols,
        rebalance_mask=rebalance_mask.reindex(trade_dates).fillna(False),
        target_by_date=target_by_date,
        meta_by_date=meta_by_date,
        elig_by_date=elig_by_date,
        cost_bps=float(cost_bps),
        vol_target_ann=0.0,
        period_col="period_end",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    rebalance_info.to_parquet(out_dir / "rebalance_info.parquet", index=False)
    daily_bt.to_parquet(out_dir / "daily_backtest.parquet", index=False)
    w_daily.to_parquet(out_dir / "weights_daily.parquet")
    pnl_plot_path = save_pnl_plot(
        daily_bt,
        out_dir / "pnl.png",
        title=f"Backtest PnL: {strategy_name}",
        baseline_ohlcv_path=baseline_ohlcv_path,
        baseline_daily_path=baseline_daily_path,
        baseline_label=baseline_label,
        secondary_baseline_ohlcv_path=secondary_baseline_ohlcv_path,
        secondary_baseline_daily_path=secondary_baseline_daily_path,
        secondary_baseline_label=secondary_baseline_label,
    )
    stats = performance_stats(pd.Series(daily_bt.set_index("trade_date")["net_ret"]).sort_index(), periods_per_year=252)
    return {
        "performance": stats,
        "artifacts": {
            "rebalance_info": str(out_dir / "rebalance_info.parquet"),
            "daily_backtest": str(out_dir / "daily_backtest.parquet"),
            "weights_daily": str(out_dir / "weights_daily.parquet"),
            "pnl_plot": str(pnl_plot_path),
        },
    }


def get_strategy_description(strategy_kind: str, size_bucket: str | None) -> str:
    base = {
        "strict_yes_or_unknown_vs_no_or_unknown": (
            "Long stock-days that contain at least one YES and no NO; short stock-days that contain at least "
            "one NO and no YES. Position sizes are proportional to YES/NO counts and normalized by gross "
            "signal strength. UNKNOWN articles are allowed on both sides."
        ),
        "paper_like_positive_negative": (
            "Paper-style approximation: daily rebalanced portfolio that buys positive LLM stock-days and sells "
            "negative stock-days with weights proportional to paper_score and normalized by gross signal "
            "strength, ignoring unknown/neutral stock-days."
        ),
        "long_only_yes_no_no": (
            "Long-only strategy that buys stock-days with at least one YES and no NO. UNKNOWN articles are allowed."
        ),
    }[strategy_kind]
    if size_bucket == "small":
        return (
            base
            + " Universe is filtered each day to small-cap names only, using the prior-day market cap and the daily "
            + "NYSE 20th-percentile breakpoint."
        )
    if size_bucket == "non_small":
        return (
            base
            + " Universe is filtered each day to non-small names only, using the prior-day market cap and the daily "
            + "NYSE 20th-percentile breakpoint."
        )
    return base


def export_size_artifacts(size_data: dict[str, object], out_dir: Path) -> dict[str, str]:
    baselines_dir = out_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    membership_df = pd.DataFrame(size_data["membership_df"]).copy()
    breakpoints_df = pd.DataFrame(size_data["breakpoints_df"]).copy()
    small_baseline_df = pd.read_parquet(str(size_data["baseline_paths"]["small"]))
    non_small_baseline_df = pd.read_parquet(str(size_data["baseline_paths"]["non_small"]))

    membership_out = baselines_dir / "response_size_membership.parquet"
    breakpoints_out = baselines_dir / "nyse_size_breakpoints.parquet"
    small_out = baselines_dir / "small_cap_baseline_daily.parquet"
    non_small_out = baselines_dir / "non_small_cap_baseline_daily.parquet"
    meta_out = baselines_dir / "size_bucket_meta.json"

    membership_df.to_parquet(membership_out, index=False)
    breakpoints_df.to_parquet(breakpoints_out, index=False)
    small_baseline_df.to_parquet(small_out, index=False)
    non_small_baseline_df.to_parquet(non_small_out, index=False)
    meta_out.write_text(json.dumps(size_data["meta"], indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "response_size_membership": str(membership_out),
        "nyse_size_breakpoints": str(breakpoints_out),
        "small_cap_baseline_daily": str(small_out),
        "non_small_cap_baseline_daily": str(non_small_out),
        "size_bucket_meta": str(meta_out),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    response_dir = Path(args.response_dir).expanduser().resolve()
    ohlcv_dir = Path(args.ohlcv_dir).expanduser().resolve()
    market_cap_dir = Path(args.market_cap_dir).expanduser().resolve() if args.market_cap_dir else None

    response_files = discover_response_files(response_dir)
    print("[init] collecting response symbols")
    response_symbols = collect_response_symbols(response_files)
    print(f"[init] collected {len(response_symbols)} unique response symbols")
    print("[init] loading cached symbol states")
    symbol_states = load_or_build_symbol_state_cache(
        ohlcv_dir=ohlcv_dir,
        min_price=float(args.min_price),
        min_adv_usd=float(args.min_adv_usd),
        start_date=int(args.start_date),
        response_symbols=response_symbols,
    )
    print(f"[init] built {len(symbol_states)} symbol-state aliases")
    pred_df = aggregate_response_signals(
        response_files=response_files,
        symbol_states=symbol_states,
        start_date=int(args.start_date),
        end_date=int(args.end_date),
        max_response_rows=int(args.max_response_rows),
        require_parsed_ok=bool(args.require_parsed_ok),
        overnight_news=bool(args.overnight_news),
    )

    size_data: dict[str, object] | None = None
    size_artifacts: dict[str, str] = {}
    if market_cap_dir is not None:
        print("[size] loading or building size-bucket data")
        size_data = load_or_build_size_bucket_data(
            market_cap_dir=market_cap_dir,
            ohlcv_dir=ohlcv_dir,
            min_price=float(args.min_price),
            min_adv_usd=float(args.min_adv_usd),
            start_date=int(args.start_date),
            end_date=int(args.end_date),
            size_breakpoint_quantile=float(args.size_breakpoint_quantile),
            response_symbols=response_symbols,
        )
        pred_df = attach_size_bucket_membership(pred_df, pd.DataFrame(size_data["membership_df"]))
        size_artifacts = export_size_artifacts(size_data, out_dir)
    else:
        pred_df["size_bucket"] = pd.Series(pd.NA, index=pred_df.index, dtype="object")
        pred_df["prev_market_cap"] = np.nan
        pred_df["nyse_market_cap_breakpoint"] = np.nan

    pred_df.to_parquet(out_dir / "aggregated_response_signals.parquet", index=False)

    bt_df = pred_df[pred_df["signal_eligible"]].copy()
    bt_df = bt_df[np.isfinite(bt_df["target_return"])]
    if bt_df.empty:
        raise RuntimeError("No eligible prediction rows for backtest after filters.")

    symbols = pd.Index(sorted(bt_df["symbol"].unique().tolist()))
    trade_dates = pd.DatetimeIndex(sorted({yyyymmdd_to_ts(int(v)) for v in bt_df["trade_date"].unique()}))

    base_strategy_names = (
        "strict_yes_or_unknown_vs_no_or_unknown",
        "paper_like_positive_negative",
        "long_only_yes_no_no",
    )
    strategy_specs: list[dict[str, object]] = []
    for strategy_kind in base_strategy_names:
        strategy_specs.append(
            {
                "strategy_name": strategy_kind,
                "strategy_kind": strategy_kind,
                "size_bucket": None,
                "baseline_ohlcv_path": args.baseline_ohlcv_path,
                "baseline_daily_path": None,
                "baseline_label": "SPY",
                "secondary_baseline_ohlcv_path": None,
                "secondary_baseline_daily_path": None,
                "secondary_baseline_label": "",
            }
        )

    if size_data is not None:
        size_variants = (
            ("small", "Small Cap VW Baseline", size_artifacts["small_cap_baseline_daily"]),
            ("non_small", "Non-Small Cap VW Baseline", size_artifacts["non_small_cap_baseline_daily"]),
        )
        for size_bucket, baseline_label, baseline_daily_path in size_variants:
            for strategy_kind in base_strategy_names:
                strategy_specs.append(
                    {
                        "strategy_name": f"{strategy_kind}_{size_bucket}_cap",
                        "strategy_kind": strategy_kind,
                        "size_bucket": size_bucket,
                        "baseline_ohlcv_path": None,
                        "baseline_daily_path": baseline_daily_path,
                        "baseline_label": baseline_label,
                        "secondary_baseline_ohlcv_path": args.baseline_ohlcv_path,
                        "secondary_baseline_daily_path": None,
                        "secondary_baseline_label": "SPY",
                    }
                )

    strategy_outputs = {}
    strategy_descriptions = {}
    for spec in strategy_specs:
        strategy_name = str(spec["strategy_name"])
        strategy_kind = str(spec["strategy_kind"])
        size_bucket = spec["size_bucket"]
        pred_df_filtered = filter_pred_df_by_size_bucket(pred_df, size_bucket=str(size_bucket) if size_bucket else None)
        bt_df_filtered = pred_df_filtered[pred_df_filtered["signal_eligible"]].copy()
        bt_df_filtered = bt_df_filtered[np.isfinite(bt_df_filtered["target_return"])]
        filtered_symbols = pd.Index(sorted(bt_df_filtered["symbol"].unique().tolist()))
        filtered_trade_dates = pd.DatetimeIndex(sorted({yyyymmdd_to_ts(int(v)) for v in bt_df_filtered["trade_date"].unique()}))

        strategy_result = run_strategy_backtest(
            strategy_name=strategy_name,
            strategy_kind=strategy_kind,
            pred_df=pred_df_filtered,
            symbols=filtered_symbols,
            trade_dates=filtered_trade_dates,
            symbol_states=symbol_states,
            out_dir=out_dir / strategy_name,
            cost_bps=float(args.cost_bps),
            min_news_pool=int(args.min_news_pool),
            min_short_count=int(args.min_short_count),
            baseline_ohlcv_path=spec["baseline_ohlcv_path"],
            baseline_daily_path=spec["baseline_daily_path"],
            baseline_label=str(spec["baseline_label"]),
            secondary_baseline_ohlcv_path=spec["secondary_baseline_ohlcv_path"],
            secondary_baseline_daily_path=spec["secondary_baseline_daily_path"],
            secondary_baseline_label=str(spec["secondary_baseline_label"]),
        )
        strategy_result["universe_filter"] = "all" if size_bucket is None else str(size_bucket)
        strategy_result["eligible_rows"] = int(len(bt_df_filtered))
        strategy_result["symbols_in_strategy_bt"] = int(len(filtered_symbols))
        strategy_outputs[strategy_name] = strategy_result
        strategy_descriptions[strategy_name] = get_strategy_description(strategy_kind, size_bucket if isinstance(size_bucket, str) else None)

    label_totals = {
        "yes": int(pred_df["yes_count"].sum()),
        "no": int(pred_df["no_count"].sum()),
        "unknown": int(pred_df["unknown_count"].sum()),
        "total_articles": int(pred_df["news_count"].sum()),
        "title_only_articles": int(pred_df["title_only_count"].sum()),
    }
    summary = {
        "response_dir": str(response_dir),
        "date_range": {"start": args.start_date, "end": args.end_date},
        "universe": {"exchanges": list(ALLOWED_EXCHANGES), "symbols_in_bt": int(len(symbols))},
        "signal_filters": {
            "min_price_prev_close": float(args.min_price),
            "min_adv20_usd_prev_day": float(args.min_adv_usd),
            "min_news_pool": int(args.min_news_pool),
            "min_short_count": int(args.min_short_count),
            "require_parsed_ok": bool(args.require_parsed_ok),
            "overnight_news": bool(args.overnight_news),
            "baseline_ohlcv_path": args.baseline_ohlcv_path,
            "market_cap_dir": str(market_cap_dir) if market_cap_dir is not None else None,
            "size_breakpoint_quantile": float(args.size_breakpoint_quantile),
        },
        "aggregated_rows": int(len(pred_df)),
        "eligible_rows": int(len(bt_df)),
        "label_totals": label_totals,
        "strategies": {name: {"description": strategy_descriptions[name], **strategy_outputs[name]} for name in strategy_outputs},
        "artifacts": {
            "aggregated_response_signals": str(out_dir / "aggregated_response_signals.parquet"),
        },
    }
    if size_data is not None:
        summary["size_buckets"] = {
            "definition": (
                "Daily size bucket uses prior-day market cap and the NYSE breakpoint at the configured quantile. "
                "Tickers without market-cap history are excluded from small/non-small strategies and baselines."
            ),
            "response_rows_with_size_bucket": int(pred_df["size_bucket"].notna().sum()),
            "eligible_rows_with_size_bucket": int(bt_df["size_bucket"].notna().sum()),
            "eligible_small_rows": int((bt_df["size_bucket"] == "small").sum()),
            "eligible_non_small_rows": int((bt_df["size_bucket"] == "non_small").sum()),
            "cache_meta": size_data["meta"],
            "artifacts": size_artifacts,
        }
        summary["artifacts"].update(size_artifacts)
    (out_dir / "backtest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["strategies"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
