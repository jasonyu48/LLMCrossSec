from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from backtest_engine import performance_stats, run_weight_execution_engine
from plot_backtest_pnl import save_pnl_plot


NY_TZ = ZoneInfo("America/New_York")
ALLOWED_EXCHANGES = ("NYSE", "NASDAQ")
MIDNIGHT_SHIFT_DOMAINS = ("investorplace.com", "fool.com")


class NullProgressBar:
    def update(self, _: int = 1) -> None:
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
            "Load trained news-return head, generate daily long/short signals from news embeddings, "
            "and run backtest via backtest_engine.py."
        )
    )
    parser.add_argument("--news-emb-dir", required=True, help="Directory with metadata_*.parquet and embeddings_*.npy.")
    parser.add_argument("--ohlcv-dir", required=True, help="Directory produced by fetch_eodhd_ohlcv_by_ticker.py.")
    parser.add_argument("--model-kind", choices=["linear", "ae_linear"], required=True)
    parser.add_argument("--model-path", required=True, help="Path to linear_head.pt or ae_linear_head.pt.")
    parser.add_argument("--ae-encoder-path", default=None, help="Required when --model-kind ae_linear.")
    parser.add_argument("--output-dir", required=True, help="Output directory for predictions/signals/backtest files.")
    parser.add_argument("--start-date", default="20230719", help="Backtest start date, inclusive, YYYYMMDD.")
    parser.add_argument("--end-date", default="20260306", help="Backtest end date, inclusive, YYYYMMDD.")
    parser.add_argument("--long-short-quantile", type=float, default=0.2, help="Long and short quantile within news pool.")
    parser.add_argument("--min-price", type=float, default=1.0, help="Signal filter uses close(t-1) >= min-price.")
    parser.add_argument(
        "--min-adv-usd",
        type=float,
        default=1_000_000.0,
        help="Signal filter uses ADV20(t-1) >= this dollar volume threshold.",
    )
    parser.add_argument("--cost-bps", type=float, default=3.0, help="Transaction cost in bps.")
    parser.add_argument("--min-news-pool", type=int, default=10, help="Skip date if eligible news pool smaller than this.")
    parser.add_argument("--max-news-rows", type=int, default=0, help="Debug cap on news rows processed. 0 means no cap.")
    parser.add_argument(
        "--overnight-news",
        "--overnight_news",
        action="store_true",
        help="If set, only use news between the previous trading day's close and the current trading day 9:00 America/New_York cutoff.",
    )
    return parser.parse_args()


def discover_chunk_pairs(news_emb_dir: Path) -> list[tuple[Path, Path]]:
    shard_dirs = sorted(p for p in news_emb_dir.iterdir() if p.is_dir() and p.name.startswith("shard_"))
    roots = shard_dirs or [news_emb_dir]
    pairs: list[tuple[Path, Path]] = []
    for root in roots:
        for meta_path in sorted(root.glob("metadata_*.parquet")):
            emb_name = meta_path.name.replace("metadata_", "embeddings_").replace(".parquet", ".npy")
            emb_path = root / emb_name
            if not emb_path.is_file():
                raise FileNotFoundError(f"Missing embedding file for {meta_path}: {emb_path}")
            pairs.append((meta_path, emb_path))
    if not pairs:
        raise FileNotFoundError(f"No metadata/embedding chunk pairs found under {news_emb_dir}")
    return pairs


def yyyymmdd_to_ts(v: int) -> pd.Timestamp:
    return pd.Timestamp(str(int(v)))


@dataclass
class SymbolState:
    symbol: str
    trade_dates_int: np.ndarray
    cutoff_ns: np.ndarray
    signal_window_open_ns: np.ndarray
    overnight_window_open_ns: np.ndarray
    ret_s: pd.Series
    signal_elig_s: pd.Series

    def assign_trade_date_int(self, ts_ns: int, overnight_only: bool = False) -> int | None:
        idx = int(np.searchsorted(self.cutoff_ns, ts_ns, side="left"))
        if idx <= 0 or idx >= len(self.trade_dates_int):
            return None
        window_open_ns = self.overnight_window_open_ns if overnight_only else self.signal_window_open_ns
        if ts_ns <= int(window_open_ns[idx]):
            return None
        return int(self.trade_dates_int[idx])


def load_symbol_state(path: Path, symbol: str, min_price: float, min_adv_usd: float) -> SymbolState | None:
    df = pd.read_parquet(path, columns=["date", "open", "close", "adjusted_close", "volume"])
    if df.empty:
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["date", "open", "close", "volume"]).sort_values("date").reset_index(drop=True)
    if len(df) < 25:
        return None

    trade_ts = df["date"].dt.normalize()
    trade_dates_int = trade_ts.dt.strftime("%Y%m%d").astype(int).to_numpy(dtype=np.int32)

    cutoff_ns = []
    close_ns = []
    for d in trade_ts.dt.date.to_numpy():
        cutoff_local = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=9, minute=0, second=0, tz=NY_TZ)
        cutoff_ns.append(int(cutoff_local.tz_convert("UTC").value))
        close_local = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=16, minute=0, second=0, tz=NY_TZ)
        close_ns.append(int(close_local.tz_convert("UTC").value))
    cutoff_ns = np.asarray(cutoff_ns, dtype=np.int64)
    close_ns = np.asarray(close_ns, dtype=np.int64)
    signal_window_open_ns = np.empty_like(cutoff_ns)
    signal_window_open_ns[0] = np.iinfo(np.int64).min
    signal_window_open_ns[1:] = cutoff_ns[:-1]
    overnight_window_open_ns = np.empty_like(close_ns)
    overnight_window_open_ns[0] = np.iinfo(np.int64).min
    overnight_window_open_ns[1:] = close_ns[:-1]

    open_px = df["open"].to_numpy(dtype=np.float64)
    close_px = df["close"].to_numpy(dtype=np.float64)
    adjusted_close_px = df["adjusted_close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    curr_open_valid = np.isfinite(open_px) & (open_px > 0)
    adj_factor = np.ones(len(df), dtype=np.float64)
    valid_adj = np.isfinite(close_px) & (close_px > 0) & np.isfinite(adjusted_close_px) & (adjusted_close_px > 0)
    adj_factor[valid_adj] = adjusted_close_px[valid_adj] / close_px[valid_adj]
    adjusted_open_px = open_px * adj_factor

    prev_open = adjusted_open_px[:-1]
    next_open = adjusted_open_px[1:]
    prev_volume = volume[:-1]
    next_volume = volume[1:]
    ret_open_open = np.zeros(len(df), dtype=np.float64)
    valid_ret = (
        np.isfinite(prev_open)
        & np.isfinite(next_open)
        & (prev_open > 0)
        & (next_open > 0)
        & np.isfinite(prev_volume)
        & np.isfinite(next_volume)
        & (prev_volume > 0)
        & (next_volume > 0)
    )
    rets = np.zeros(len(prev_open), dtype=np.float64)
    rets[valid_ret] = (next_open[valid_ret] / prev_open[valid_ret]) - 1.0
    ret_open_open[:-1] = rets

    dollar_vol = close_px * volume
    adv20 = pd.Series(dollar_vol).rolling(20, min_periods=20).mean().shift(1).to_numpy(dtype=np.float64)
    prev_close = pd.Series(close_px).shift(1).to_numpy(dtype=np.float64)

    signal_elig = (
        curr_open_valid
        & np.isfinite(prev_close)
        & np.isfinite(adv20)
        & (prev_close >= float(min_price))
        & (adv20 >= float(min_adv_usd))
    )

    ret_s = pd.Series(ret_open_open.astype(np.float64), index=trade_ts.to_numpy())
    signal_elig_s = pd.Series(signal_elig.astype(bool), index=trade_ts.to_numpy())

    return SymbolState(
        symbol=symbol,
        trade_dates_int=trade_dates_int,
        cutoff_ns=cutoff_ns,
        signal_window_open_ns=signal_window_open_ns,
        overnight_window_open_ns=overnight_window_open_ns,
        ret_s=ret_s,
        signal_elig_s=signal_elig_s,
    )


def build_symbol_states(
    ohlcv_dir: Path,
    min_price: float,
    min_adv_usd: float,
    allowed_symbols: set[str] | None = None,
) -> dict[str, SymbolState]:
    parquet_root = ohlcv_dir / "parquet"
    if not parquet_root.is_dir():
        raise FileNotFoundError(f"Expected OHLCV parquet root: {parquet_root}")

    candidate_paths: list[Path] = []
    for exch in ALLOWED_EXCHANGES:
        exch_dir = parquet_root / exch
        if exch_dir.is_dir():
            candidate_paths.extend(sorted(exch_dir.glob("*.parquet")))

    out: dict[str, SymbolState] = {}
    progress = make_progress_bar(total=len(candidate_paths), desc="OHLCV files", unit="file")
    try:
        for path in candidate_paths:
            stem = path.stem.upper()
            if allowed_symbols is not None:
                stem_variants = {stem}
                if "." in stem:
                    base = stem.rsplit(".", 1)[0]
                    stem_variants.add(base)
                    stem_variants.add(f"{base}.US")
                if stem_variants.isdisjoint(allowed_symbols):
                    progress.update(1)
                    continue

            st = load_symbol_state(path, stem, min_price=min_price, min_adv_usd=min_adv_usd)
            if st is not None:
                out.setdefault(stem, st)
                if "." in stem:
                    base = stem.rsplit(".", 1)[0]
                    out.setdefault(base, st)
                    out.setdefault(f"{base}.US", st)
            progress.update(1)
    finally:
        progress.close()
    if not out:
        raise RuntimeError("No valid NYSE/NASDAQ symbol states built from OHLCV data.")
    return out


def load_linear_params(model_path: Path) -> tuple[np.ndarray, float, str]:
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(model_path, map_location="cpu")
    weight = payload.get("weight")
    bias = payload.get("bias")
    if weight is None or bias is None:
        raise ValueError(f"{model_path} does not contain closed-form linear params: weight/bias.")
    weight_arr = np.asarray(weight, dtype=np.float32)
    bias_val = float(bias)
    inference_mode = str(payload.get("inference_mode", "sum_embedding"))
    return weight_arr, bias_val, inference_mode


def load_encoder(encoder_path: Path) -> torch.nn.Module:
    try:
        payload = torch.load(encoder_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(encoder_path, map_location="cpu")
    state = payload["encoder_state_dict"]
    input_dim = int(payload["input_dim"])
    hidden_dim = int(payload["hidden_dim"])
    latent_dim = int(payload["latent_dim"])

    encoder = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_dim, latent_dim),
    )
    encoder.load_state_dict(state)
    encoder = encoder.cuda()
    encoder.eval()
    return encoder


def apply_linear(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return (x.astype(np.float32, copy=False) @ w.astype(np.float32, copy=False) + np.float32(b)).astype(
        np.float32, copy=False
    )


def finite_row_mask(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={arr.shape}")
    return np.isfinite(arr).all(axis=1)


def _link_matches_midnight_shift_domain(link: object) -> bool:
    text = str(link or "").strip().lower()
    if not text:
        return False
    host = urlparse(text).netloc.strip().lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return False
    return any(host == domain or host.endswith(f".{domain}") for domain in MIDNIGHT_SHIFT_DOMAINS)


def parse_news_timestamps_utc(values: pd.Series, links: pd.Series | None = None) -> pd.Series:
    try:
        ts = pd.to_datetime(values, utc=True, errors="coerce", format="ISO8601")
    except (TypeError, ValueError):
        ts = pd.to_datetime(values, utc=True, errors="coerce")

    if links is None:
        return ts

    midnight_mask = ts.notna() & (ts == ts.dt.normalize())
    if not bool(midnight_mask.any()):
        return ts

    domain_mask = links.map(_link_matches_midnight_shift_domain).astype(bool)
    shift_mask = midnight_mask & domain_mask.reindex(ts.index, fill_value=False)
    if bool(shift_mask.any()):
        ts = ts.copy()
        ts.loc[shift_mask] = ts.loc[shift_mask] + pd.Timedelta(days=1)
    return ts


def predict_group_scores(feat_sum: np.ndarray, news_count: np.ndarray, w: np.ndarray, b: float, inference_mode: str) -> np.ndarray:
    if inference_mode == "mean":
        x = feat_sum.astype(np.float32, copy=False) / news_count[:, None].astype(np.float32, copy=False)
        return apply_linear(x, w, b)
    if inference_mode == "sum_head":
        pred = feat_sum.astype(np.float32, copy=False) @ w.astype(np.float32, copy=False)
        pred = pred + news_count.astype(np.float32, copy=False) * np.float32(b)
        return pred.astype(np.float32, copy=False)
    if inference_mode == "sum_embedding":
        return apply_linear(feat_sum, w, b)
    raise ValueError(f"Unsupported inference_mode in model payload: {inference_mode}")


def in_bt_range(date_int: int, start_date: int, end_date: int) -> bool:
    return int(start_date) <= int(date_int) <= int(end_date)


def aggregate_predictions(
    *,
    chunk_pairs: list[tuple[Path, Path]],
    symbol_states: dict[str, SymbolState],
    model_kind: str,
    weight: np.ndarray,
    bias: float,
    inference_mode: str,
    encoder: torch.nn.Module | None,
    start_date: int,
    end_date: int,
    max_news_rows: int,
) -> pd.DataFrame:
    agg_sum: dict[tuple[int, str], np.ndarray] = {}
    agg_cnt: dict[tuple[int, str], int] = {}

    processed = 0
    with torch.inference_mode():
        for meta_path, emb_path in chunk_pairs:
            meta = pd.read_parquet(meta_path, columns=["symbol", "date", "link"])
            emb = np.load(emb_path, mmap_mode=None)
            if len(meta) != len(emb):
                raise ValueError(f"Row mismatch between {meta_path} and {emb_path}: {len(meta)} vs {len(emb)}")

            ts = parse_news_timestamps_utc(meta["date"], meta["link"])
            ts_ns = ts.view("int64").to_numpy()
            syms = meta["symbol"].fillna("").astype(str).str.upper().to_numpy()

            valid_rows: list[int] = []
            keys: list[tuple[int, str]] = []

            for i in range(len(meta)):
                symbol = syms[i]
                if not symbol:
                    continue
                st = symbol_states.get(symbol)
                if st is None:
                    continue
                trade_date_int = st.assign_trade_date_int(int(ts_ns[i]), overnight_only=bool(args.overnight_news))
                if trade_date_int is None:
                    continue
                if not in_bt_range(trade_date_int, start_date, end_date):
                    continue
                valid_rows.append(i)
                keys.append((int(trade_date_int), symbol))

            if not valid_rows:
                continue

            vec = emb[np.asarray(valid_rows, dtype=np.int64)].astype(np.float32, copy=False)
            if model_kind == "ae_linear":
                if encoder is None:
                    raise RuntimeError("encoder is required for ae_linear.")
                xb = torch.from_numpy(vec).cuda(non_blocking=True)
                vec = encoder(xb).detach().cpu().numpy().astype(np.float32, copy=False)

            finite_mask = finite_row_mask(vec)
            if not finite_mask.all():
                vec = vec[finite_mask]
                keys = [key for key, keep in zip(keys, finite_mask.tolist(), strict=True) if keep]
            if len(keys) == 0:
                continue

            for k, v in zip(keys, vec, strict=True):
                if k not in agg_sum:
                    agg_sum[k] = np.array(v, dtype=np.float32, copy=True)
                    agg_cnt[k] = 1
                else:
                    agg_sum[k] += v
                    agg_cnt[k] += 1

                processed += 1
                if max_news_rows > 0 and processed >= max_news_rows:
                    break

            if max_news_rows > 0 and processed >= max_news_rows:
                break

    rows = []
    feats = []
    for (trade_date_int, symbol), feat in agg_sum.items():
        rows.append(
            {
                "trade_date": int(trade_date_int),
                "symbol": symbol,
                "news_count": int(agg_cnt[(trade_date_int, symbol)]),
            }
        )
        feats.append(feat)
    if not rows:
        raise RuntimeError("No prediction rows generated. Check date range or input paths.")

    pred_df = pd.DataFrame(rows)
    feat_sum = np.stack(feats, axis=0).astype(np.float32, copy=False)
    news_count = pred_df["news_count"].to_numpy(dtype=np.float32)
    pred_df["pred"] = predict_group_scores(feat_sum, news_count, weight, bias, inference_mode)

    elig_flags = []
    target_ret = []
    for row in pred_df.itertuples(index=False):
        st = symbol_states[row.symbol]
        dt = yyyymmdd_to_ts(row.trade_date)
        elig = bool(st.signal_elig_s.get(dt, False))
        ret = st.ret_s.get(dt, np.nan)
        elig_flags.append(elig)
        target_ret.append(float(ret) if np.isfinite(ret) else np.nan)

    pred_df["signal_eligible"] = np.asarray(elig_flags, dtype=bool)
    pred_df["target_return"] = np.asarray(target_ret, dtype=np.float64)
    pred_df = pred_df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return pred_df


def build_targets_and_meta(
    pred_df: pd.DataFrame,
    symbols: pd.Index,
    trade_dates: pd.DatetimeIndex,
    q: float,
    min_news_pool: int,
) -> tuple[dict[pd.Timestamp, pd.Series], dict[pd.Timestamp, dict], pd.Series]:
    target_by_date: dict[pd.Timestamp, pd.Series] = {}
    meta_by_date: dict[pd.Timestamp, dict] = {}
    rebalance_dates: list[pd.Timestamp] = []
    groups = {int(k): grp for k, grp in pred_df.groupby("trade_date", sort=False)}

    for dt in trade_dates:
        trade_date_int = int(pd.Timestamp(dt).strftime("%Y%m%d"))
        grp = groups.get(trade_date_int, pred_df.iloc[0:0])
        eligible_grp = grp[grp["signal_eligible"]].copy()
        n_pool = len(eligible_grp)
        k = int(np.floor(float(q) * n_pool))
        w = pd.Series(0.0, index=symbols, dtype="float64")
        if n_pool >= int(min_news_pool) and k >= 1:
            eligible_grp = eligible_grp.sort_values("pred")
            short_syms = eligible_grp.head(k)["symbol"].to_list()
            long_syms = eligible_grp.tail(k)["symbol"].to_list()
            w.loc[short_syms] = -0.5 / float(k)
            w.loc[long_syms] = 0.5 / float(k)

        target_by_date[dt] = w
        meta_by_date[dt] = {
            "signal_period_end": dt,
            "n_eligible": int(n_pool),
            "n_pb_pool": int(n_pool),
        }
        rebalance_dates.append(dt)

    rebalance_mask = pd.Series(True, index=pd.DatetimeIndex(sorted(rebalance_dates)))
    return target_by_date, meta_by_date, rebalance_mask


def build_backtest_trade_dates(
    symbol_states: dict[str, SymbolState],
    start_date: int,
    end_date: int,
) -> pd.DatetimeIndex:
    trade_dates_int: set[int] = set()
    for st in symbol_states.values():
        vals = st.trade_dates_int
        mask = (vals >= int(start_date)) & (vals <= int(end_date))
        if mask.any():
            trade_dates_int.update(int(v) for v in vals[mask].tolist())
    return pd.DatetimeIndex(sorted(yyyymmdd_to_ts(v) for v in trade_dates_int))


def build_ret_and_elig_matrices(
    symbol_states: dict[str, SymbolState],
    symbols: pd.Index,
    trade_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, pd.Series]]:
    ret_cols = {}
    elig_cols = {}
    for s in symbols:
        st = symbol_states[s]
        ret_cols[s] = st.ret_s.reindex(trade_dates).astype("float64")
        elig_cols[s] = st.signal_elig_s.reindex(trade_dates).fillna(False).astype(bool)
    ret_d = pd.DataFrame(ret_cols, index=trade_dates).sort_index()
    elig_d = pd.DataFrame(elig_cols, index=trade_dates).sort_index()
    elig_by_date = {dt: elig_d.loc[dt] for dt in elig_d.index}
    return ret_d, elig_by_date


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    news_emb_dir = Path(args.news_emb_dir).expanduser().resolve()
    ohlcv_dir = Path(args.ohlcv_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    encoder_path = Path(args.ae_encoder_path).expanduser().resolve() if args.ae_encoder_path else None

    if args.model_kind == "ae_linear" and encoder_path is None:
        raise ValueError("--ae-encoder-path is required when --model-kind ae_linear.")

    symbol_states = build_symbol_states(
        ohlcv_dir=ohlcv_dir,
        min_price=float(args.min_price),
        min_adv_usd=float(args.min_adv_usd),
    )
    chunk_pairs = discover_chunk_pairs(news_emb_dir)

    weight, bias, inference_mode = load_linear_params(model_path)
    encoder = load_encoder(encoder_path) if args.model_kind == "ae_linear" else None

    pred_df = aggregate_predictions(
        chunk_pairs=chunk_pairs,
        symbol_states=symbol_states,
        model_kind=args.model_kind,
        weight=weight,
        bias=bias,
        inference_mode=inference_mode,
        encoder=encoder,
        start_date=int(args.start_date),
        end_date=int(args.end_date),
        max_news_rows=int(args.max_news_rows),
    )
    pred_df.to_parquet(out_dir / "predictions_backtest_window.parquet", index=False)

    bt_df = pred_df[pred_df["signal_eligible"]].copy()
    bt_df = bt_df[np.isfinite(bt_df["target_return"])]
    if bt_df.empty:
        raise RuntimeError("No eligible prediction rows for backtest after filters.")

    symbols = pd.Index(sorted(bt_df["symbol"].unique().tolist()))
    trade_dates = build_backtest_trade_dates(
        symbol_states=symbol_states,
        start_date=int(args.start_date),
        end_date=int(args.end_date),
    )

    ret_d, elig_by_date = build_ret_and_elig_matrices(
        symbol_states=symbol_states,
        symbols=symbols,
        trade_dates=trade_dates,
    )
    cal = pd.DataFrame(index=trade_dates)
    cal["period_end"] = trade_dates

    target_by_date, meta_by_date, rebalance_mask = build_targets_and_meta(
        pred_df=pred_df,
        symbols=symbols,
        trade_dates=trade_dates,
        q=float(args.long_short_quantile),
        min_news_pool=int(args.min_news_pool),
    )

    rebalance_info, daily_bt, w_daily = run_weight_execution_engine(
        ret_d=ret_d,
        cal=cal,
        symbols=symbols,
        rebalance_mask=rebalance_mask.reindex(trade_dates).fillna(False),
        target_by_date=target_by_date,
        meta_by_date=meta_by_date,
        elig_by_date=elig_by_date,
        cost_bps=float(args.cost_bps),
        vol_target_ann=0.0,
        period_col="period_end",
    )

    rebalance_info.to_parquet(out_dir / "rebalance_info.parquet", index=False)
    daily_bt.to_parquet(out_dir / "daily_backtest.parquet", index=False)
    w_daily.to_parquet(out_dir / "weights_daily.parquet")
    pnl_plot_path = save_pnl_plot(
        daily_bt,
        out_dir / "pnl.png",
        title=f"Backtest PnL: {args.model_kind}",
    )

    stats = performance_stats(
        pd.Series(daily_bt.set_index("trade_date")["net_ret"]).sort_index(),
        periods_per_year=252,
    )
    summary = {
        "model_kind": args.model_kind,
        "model_path": str(model_path),
        "ae_encoder_path": str(encoder_path) if encoder_path else None,
        "date_range": {"start": args.start_date, "end": args.end_date},
        "universe": {"exchanges": list(ALLOWED_EXCHANGES), "symbols_in_bt": int(len(symbols))},
        "signal_filters": {
            "min_price_prev_close": float(args.min_price),
            "min_adv20_usd_prev_day": float(args.min_adv_usd),
            "long_short_quantile": float(args.long_short_quantile),
            "min_news_pool": int(args.min_news_pool),
            "overnight_news": bool(args.overnight_news),
        },
        "cost_bps": float(args.cost_bps),
        "performance": stats,
        "artifacts": {
            "predictions": str(out_dir / "predictions_backtest_window.parquet"),
            "rebalance_info": str(out_dir / "rebalance_info.parquet"),
            "daily_backtest": str(out_dir / "daily_backtest.parquet"),
            "weights_daily": str(out_dir / "weights_daily.parquet"),
            "pnl_plot": str(pnl_plot_path),
        },
    }
    (out_dir / "backtest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["performance"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
