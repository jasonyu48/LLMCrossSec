from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_news_head_backtest import build_symbol_states, load_symbol_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export holdings, contributions, and news for abnormal LLM-response backtest days."
    )
    parser.add_argument(
        "--backtest-root",
        required=True,
        help="Directory containing aggregated_response_signals.parquet and strategy subdirectories.",
    )
    parser.add_argument(
        "--response-dir",
        required=True,
        help="Directory containing shard_*/responses_*.parquet from generate_news_llm_responses.py.",
    )
    parser.add_argument(
        "--ohlcv-dir",
        required=True,
        help="OHLCV directory used by the backtest.",
    )
    parser.add_argument(
        "--strategy",
        default="paper_like_positive_negative",
        help="Strategy subdirectory under backtest-root. Default: paper_like_positive_negative",
    )
    parser.add_argument(
        "--date",
        action="append",
        default=None,
        help="Trade date(s) to inspect in YYYYMMDD or YYYY-MM-DD. If omitted, inspect days with net_ret < -1.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Default: <backtest-root>/blowup_analysis/<strategy>",
    )
    return parser.parse_args()


def normalize_trade_date(raw: str) -> int:
    return int(str(raw).replace("-", ""))


def discover_dates(daily_bt: pd.DataFrame, explicit_dates: list[str] | None) -> list[int]:
    if explicit_dates:
        return sorted({normalize_trade_date(item) for item in explicit_dates})
    rows = daily_bt[daily_bt["net_ret"] < -1.0].copy()
    return sorted(rows["trade_date"].dt.strftime("%Y%m%d").astype(int).tolist())


def discover_response_files(response_dir: Path) -> list[Path]:
    shard_dirs = sorted(p for p in response_dir.iterdir() if p.is_dir() and p.name.startswith("shard_"))
    roots = shard_dirs or [response_dir]
    out: list[Path] = []
    for root in roots:
        out.extend(sorted(root.glob("responses_*.parquet")))
    if not out:
        raise FileNotFoundError(f"No responses_*.parquet found under {response_dir}")
    return out


def resolve_output_dir(backtest_root: Path, strategy: str, explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        return Path(explicit_output_dir).expanduser().resolve()
    return (backtest_root / "blowup_analysis" / strategy).resolve()


def resolve_ohlcv_file(symbol: str, ohlcv_dir: Path) -> Path | None:
    base = symbol.upper().replace(".US", "")
    parquet_root = ohlcv_dir / "parquet"
    preferred = [
        parquet_root / "NYSE" / f"{base}.NYSE.parquet",
        parquet_root / "NASDAQ" / f"{base}.NASDAQ.parquet",
    ]
    for path in preferred:
        if path.is_file():
            return path

    for exch in ("NYSE", "NASDAQ"):
        matches = sorted((parquet_root / exch).glob(f"{base}*.{exch}.parquet"))
        if matches:
            return matches[0]
    return None


def build_price_context(symbol: str, trade_date: int, ohlcv_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    path = resolve_ohlcv_file(symbol, ohlcv_dir)
    if path is None:
        return pd.DataFrame(), {"symbol": symbol, "trade_date": trade_date, "ohlcv_file": None}

    st = load_symbol_state(path, symbol, min_price=1.0, min_adv_usd=1_000_000.0)
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    date_ts = pd.Timestamp(str(trade_date))
    idx = df.index[df["date"] == date_ts]
    if len(idx) == 0:
        return pd.DataFrame(), {"symbol": symbol, "trade_date": trade_date, "ohlcv_file": str(path)}

    loc = int(idx[0])
    window = df.iloc[max(0, loc - 3) : min(len(df), loc + 4)].copy()
    keep_cols = [c for c in ["date", "open", "high", "low", "close", "adjusted_close", "volume"] if c in window.columns]
    window = window[keep_cols]

    current_open = float(df.iloc[loc]["open"])
    next_open = float(df.iloc[loc + 1]["open"]) if loc + 1 < len(df) else None
    raw_return = (next_open / current_open - 1.0) if next_open and current_open else None
    meta = {
        "symbol": symbol,
        "trade_date": trade_date,
        "ohlcv_file": str(path),
        "trade_open": current_open,
        "next_open": next_open,
        "raw_open_to_open_return": raw_return,
        "signal_eligible": None if st is None else bool(st.signal_elig_s.get(date_ts, False)),
    }
    return window, meta


def extract_relevant_news(
    *,
    response_files: list[Path],
    symbol_states: dict[str, object],
    target_dates: set[int],
    target_symbols: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for resp_path in response_files:
        cols = [
            "date",
            "symbol",
            "title",
            "link",
            "response_label",
            "response_explanation",
            "source_file",
            "source_row",
        ]
        df = pd.read_parquet(resp_path, columns=cols)
        df = df[df["symbol"].isin(target_symbols)]
        if df.empty:
            continue

        ts = pd.to_datetime(df["date"], utc=True)
        for row, t in zip(df.itertuples(index=False), ts):
            st = symbol_states.get(row.symbol)
            if st is None:
                continue
            trade_date = st.assign_trade_date_int(int(t.value))
            if trade_date not in target_dates:
                continue
            rows.append(
                {
                    "trade_date": int(trade_date),
                    "news_ts_utc": t.isoformat(),
                    "symbol": row.symbol,
                    "response_label": row.response_label,
                    "title": row.title,
                    "link": row.link,
                    "response_explanation": row.response_explanation,
                    "source_file": row.source_file,
                    "source_row": row.source_row,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["trade_date", "symbol", "news_ts_utc"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    backtest_root = Path(args.backtest_root).expanduser().resolve()
    response_dir = Path(args.response_dir).expanduser().resolve()
    ohlcv_dir = Path(args.ohlcv_dir).expanduser().resolve()
    strategy_dir = backtest_root / args.strategy
    out_dir = resolve_output_dir(backtest_root, args.strategy, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_bt = pd.read_parquet(strategy_dir / "daily_backtest.parquet")
    daily_bt["trade_date"] = pd.to_datetime(daily_bt["trade_date"])
    weights = pd.read_parquet(strategy_dir / "weights_daily.parquet")
    weights.index = pd.to_datetime(weights.index)
    signals = pd.read_parquet(backtest_root / "aggregated_response_signals.parquet")

    target_dates = discover_dates(daily_bt, args.date)
    if not target_dates:
        raise RuntimeError("No target dates found.")

    held_symbols: set[str] = set()
    short_symbols: set[str] = set()
    for trade_date in target_dates:
        ts = pd.Timestamp(str(trade_date))
        row = weights.loc[ts]
        nz = row[row != 0].copy()
        held_symbols.update(nz.index.tolist())
        short_symbols.update(nz[nz < 0].index.tolist())

    symbol_states = build_symbol_states(
        ohlcv_dir=ohlcv_dir,
        min_price=1.0,
        min_adv_usd=1_000_000.0,
        allowed_symbols=held_symbols | short_symbols,
    )
    response_files = discover_response_files(response_dir)
    short_news = extract_relevant_news(
        response_files=response_files,
        symbol_states=symbol_states,
        target_dates=set(target_dates),
        target_symbols=short_symbols,
    )

    summary_rows: list[dict[str, object]] = []
    for trade_date in target_dates:
        trade_ts = pd.Timestamp(str(trade_date))
        day_dir = out_dir / str(trade_date)
        day_dir.mkdir(parents=True, exist_ok=True)

        row = weights.loc[trade_ts]
        holdings = row[row != 0].rename("weight").to_frame()
        day_signals = signals[signals["trade_date"] == trade_date].copy().set_index("symbol")
        holdings["target_return"] = day_signals["target_return"]
        holdings["paper_score"] = day_signals["paper_score"]
        holdings["yes_count"] = day_signals["yes_count"]
        holdings["no_count"] = day_signals["no_count"]
        holdings["unknown_count"] = day_signals["unknown_count"]
        holdings["news_count"] = day_signals["news_count"]
        holdings["signal_eligible"] = day_signals["signal_eligible"]
        holdings["contribution"] = holdings["weight"] * holdings["target_return"]
        holdings = holdings.sort_values("contribution")
        holdings.to_csv(day_dir / "holdings.csv")

        longs = holdings[holdings["weight"] > 0].copy().sort_index()
        longs[["weight", "paper_score", "yes_count", "no_count", "news_count", "target_return", "contribution"]].to_csv(
            day_dir / "long_holdings.csv"
        )
        shorts = holdings[holdings["weight"] < 0].copy()
        shorts.to_csv(day_dir / "short_holdings.csv")

        (day_dir / "long_symbols.txt").write_text("\n".join(longs.index.tolist()) + "\n", encoding="utf-8")

        short_symbol_set = set(shorts.index.tolist())
        day_short_news = short_news[
            (short_news["trade_date"] == trade_date) & (short_news["symbol"].isin(short_symbol_set))
        ].copy()
        day_short_news.to_csv(day_dir / "short_news.csv", index=False)

        culprit = shorts.sort_values("contribution").index.tolist()
        culprit_rows: list[dict[str, object]] = []
        for symbol in culprit:
            price_context, meta = build_price_context(symbol, trade_date, ohlcv_dir)
            if not price_context.empty:
                price_context.to_csv(day_dir / f"{symbol.replace('.', '_')}_price_context.csv", index=False)
            culprit_rows.append(meta)
        pd.DataFrame(culprit_rows).to_csv(day_dir / "short_price_context.csv", index=False)

        day_bt = daily_bt[daily_bt["trade_date"] == trade_ts].iloc[0]
        summary_rows.append(
            {
                "trade_date": trade_date,
                "net_ret": float(day_bt["net_ret"]),
                "gross_ret": float(day_bt["gross_ret"]),
                "n_holdings": int(len(holdings)),
                "n_longs": int((holdings["weight"] > 0).sum()),
                "n_shorts": int((holdings["weight"] < 0).sum()),
                "worst_symbol": holdings.index[0],
                "worst_contribution": float(holdings.iloc[0]["contribution"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("trade_date")
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
