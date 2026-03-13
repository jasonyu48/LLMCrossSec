from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_TICKERS_FILE = Path("/home/jyu197/LLMCrossSec/us_universe_merged/us_nyse_nasdaq_common_stock.tickers.txt")
DEFAULT_EXCLUDE_TICKERS_FILE = Path(
    "/export/fs06/jyu197/eodhd/ohlcv_nyse_nasdaq_active_delisted_since_20200101_v2/tickers_without_data.csv"
)
DEFAULT_OHLCV_DIR = Path("/export/fs06/jyu197/eodhd/ohlcv_nyse_nasdaq_active_delisted_since_20200101_v2")
DEFAULT_METADATA_CSV = Path("/home/jyu197/LLMCrossSec/us_universe_merged/us_nyse_nasdaq_common_stock.metadata.csv")
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_SLEEP_SECONDS = 0.25
DEFAULT_MAX_RETRIES = 6
DEFAULT_RETRY_SLEEP_SECONDS = 15.0


class NullProgressBar:
    def update(self, _: int = 1) -> None:
        pass

    def set_description(self, _: str) -> None:
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
        description="Build daily historical market cap from yfinance shares outstanding and local OHLCV close prices."
    )
    parser.add_argument("--tickers-file", default=str(DEFAULT_TICKERS_FILE), help=f"Ticker list to request. Default: {DEFAULT_TICKERS_FILE}")
    parser.add_argument(
        "--exclude-tickers-file",
        default=str(DEFAULT_EXCLUDE_TICKERS_FILE),
        help="CSV file of tickers to exclude, e.g. OHLCV tickers_without_data.csv.",
    )
    parser.add_argument(
        "--metadata-csv",
        default=str(DEFAULT_METADATA_CSV),
        help="Universe metadata CSV used for local filtering of units, warrants, and rights.",
    )
    parser.add_argument("--ohlcv-dir", default=str(DEFAULT_OHLCV_DIR), help=f"OHLCV root directory. Default: {DEFAULT_OHLCV_DIR}")
    parser.add_argument(
        "--response-dir",
        default=None,
        help=(
            "Optional directory with shard_*/responses_*.parquet or responses_*.parquet. "
            "If provided, only tickers that actually appear in response/news data are requested."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory where per-ticker parquet files and state files are written.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help=f"Keep rows with date >= this value. Default: {DEFAULT_START_DATE}")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help=f"Optional sleep between ticker requests. Default: {DEFAULT_SLEEP_SECONDS}",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries for retryable yfinance errors such as rate limits. Default: {DEFAULT_MAX_RETRIES}",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=DEFAULT_RETRY_SLEEP_SECONDS,
        help=f"Base sleep in seconds before retrying a rate-limited request. Default: {DEFAULT_RETRY_SLEEP_SECONDS}",
    )
    parser.add_argument("--state-file", default=None, help="Optional custom path for crawl state JSON.")
    parser.add_argument("--summary-file", default=None, help="Optional custom path for summary JSON.")
    parser.add_argument("--reset-state", action="store_true", help="Delete any existing crawl state and start from scratch.")
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional cap on tickers processed for debugging. 0 means no cap.")
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=0.5,
        help="Require at least this fraction of trading days to have a filled shares value after forward-fill. Default: 0.5",
    )
    return parser.parse_args()


def normalize_ticker(value: str) -> str:
    return value.strip().upper()


def extract_exchange_from_ticker(ticker: str) -> str:
    if "." not in ticker:
        return ""
    return ticker.rsplit(".", 1)[1].strip().upper()


def extract_base_symbol(ticker: str) -> str:
    text = normalize_ticker(ticker)
    if "." not in text:
        return text
    return text.rsplit(".", 1)[0].strip().upper()


def to_yahoo_symbol(ticker: str) -> str:
    if "." not in ticker:
        return normalize_ticker(ticker)
    base, _exchange = ticker.rsplit(".", 1)
    return normalize_ticker(base)


def load_tickers(path: Path) -> list[str]:
    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        tickers.append(normalize_ticker(text.split(",")[0]))
    return tickers


def load_excluded_tickers(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError(f"{path} must contain a ticker column.")
    return {normalize_ticker(v) for v in df["ticker"].dropna().astype(str).tolist() if v.strip()}


def load_metadata(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame(columns=["full_ticker", "name", "type"])
    df = pd.read_csv(path, usecols=["full_ticker", "name", "type"])
    df["full_ticker"] = df["full_ticker"].fillna("").astype(str).str.upper().str.strip()
    df["name"] = df["name"].fillna("").astype(str)
    df["type"] = df["type"].fillna("").astype(str)
    df = df[df["full_ticker"] != ""].drop_duplicates(subset=["full_ticker"], keep="last").reset_index(drop=True)
    return df


def is_filtered_non_equity_name(name: str, ticker: str) -> bool:
    text = f"{ticker} {name}".lower()
    keywords = (
        " warrant",
        " warrants",
        " right",
        " rights",
        " unit",
        " units",
    )
    if any(keyword in text for keyword in keywords):
        return True
    ticker_upper = ticker.upper()
    return any(token in ticker_upper for token in ("-WT", "-WS", "-W", "-RT", "-RW", "-R", "-U", "-UN"))


def filter_tickers_by_metadata(
    tickers: list[str],
    metadata_df: pd.DataFrame,
) -> tuple[list[str], list[dict[str, str]]]:
    if metadata_df.empty:
        return tickers, []
    meta = metadata_df.set_index("full_ticker")
    kept: list[str] = []
    filtered_rows: list[dict[str, str]] = []
    for ticker in tickers:
        row = meta.loc[ticker] if ticker in meta.index else None
        if row is None:
            kept.append(ticker)
            continue
        name = str(row["name"])
        sec_type = str(row["type"])
        if is_filtered_non_equity_name(name=name, ticker=ticker):
            filtered_rows.append(
                {
                    "ticker": ticker,
                    "name": name,
                    "type": sec_type,
                    "reason": "filtered_unit_warrant_rights",
                }
            )
            continue
        kept.append(ticker)
    return kept, filtered_rows


def discover_response_files(response_dir: Path) -> list[Path]:
    shard_dirs = sorted(p for p in response_dir.iterdir() if p.is_dir() and p.name.startswith("shard_"))
    roots = shard_dirs or [response_dir]
    out: list[Path] = []
    for root in roots:
        out.extend(sorted(root.glob("responses_*.parquet")))
    if not out:
        raise FileNotFoundError(f"No responses_*.parquet files found under {response_dir}")
    return out


def collect_response_symbols(response_dir: Path) -> set[str]:
    symbols: set[str] = set()
    files = discover_response_files(response_dir)
    progress = make_progress_bar(total=len(files), desc="Response symbols", unit="file")
    try:
        for resp_path in files:
            df = pd.read_parquet(resp_path, columns=["symbol"])
            if not df.empty:
                vals = df["symbol"].fillna("").astype(str).str.upper()
                symbols.update(sym for sym in vals.unique().tolist() if sym)
            progress.update(1)
    finally:
        progress.close()
    return symbols


def filter_tickers_by_response_symbols(
    tickers: list[str],
    response_symbols: set[str],
) -> tuple[list[str], list[dict[str, str]]]:
    response_base_symbols = {extract_base_symbol(symbol) for symbol in response_symbols if symbol}
    kept: list[str] = []
    filtered_rows: list[dict[str, str]] = []
    for ticker in tickers:
        if extract_base_symbol(ticker) in response_base_symbols:
            kept.append(ticker)
        else:
            filtered_rows.append({"ticker": ticker, "reason": "filtered_no_response_news_data"})
    return kept, filtered_rows


def sanitize_filename(text: str) -> str:
    return text.replace("/", "_")


def ticker_output_path(output_dir: Path, ticker: str) -> Path:
    exchange = extract_exchange_from_ticker(ticker) or "UNKNOWN"
    exchange_dir = output_dir / "parquet" / exchange
    return exchange_dir / f"{sanitize_filename(ticker)}.parquet"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def default_state(output_dir: Path, tickers: list[str], start_date: str) -> dict[str, Any]:
    return {
        "version": 1,
        "config": {
            "start_date": start_date,
            "tickers_total": len(tickers),
        },
        "current_ticker_index": 0,
        "stats": {
            "tickers_completed": 0,
            "tickers_with_data": 0,
            "tickers_without_data": 0,
            "tickers_failed": 0,
            "tickers_missing_ohlcv": 0,
            "rows_written": 0,
        },
        "paths": {
            "output_dir": str(output_dir),
        },
    }


def ohlcv_path_for_ticker(ohlcv_dir: Path, ticker: str) -> Path:
    exchange = extract_exchange_from_ticker(ticker) or "UNKNOWN"
    return ohlcv_dir / "parquet" / exchange / f"{sanitize_filename(ticker)}.parquet"


def filter_tickers_with_ohlcv(ohlcv_dir: Path, tickers: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    kept: list[str] = []
    missing_rows: list[dict[str, str]] = []
    for ticker in tickers:
        ohlcv_path = ohlcv_path_for_ticker(ohlcv_dir, ticker)
        if ohlcv_path.is_file():
            kept.append(ticker)
        else:
            missing_rows.append(
                {
                    "ticker": ticker,
                    "exchange": extract_exchange_from_ticker(ticker),
                    "ohlcv_path": str(ohlcv_path),
                }
            )
    return kept, missing_rows


def load_ohlcv_close_series(ohlcv_path: Path, start_date: str) -> pd.DataFrame:
    df = pd.read_parquet(ohlcv_path, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    df = df[df["date"] >= pd.Timestamp(start_date)]
    return df


def is_retryable_yfinance_error(exc: Exception) -> bool:
    text = str(exc).strip().lower()
    return any(token in text for token in ("too many requests", "rate limited", "429", "try after a while"))


def fetch_shares_series(
    yahoo_symbol: str,
    start_date: str,
    *,
    max_retries: int,
    retry_sleep_seconds: float,
) -> pd.Series:
    ticker = yf.Ticker(yahoo_symbol)
    shares = None
    for attempt in range(int(max_retries) + 1):
        try:
            shares = ticker.get_shares_full(start=start_date)
            break
        except Exception as exc:
            if not is_retryable_yfinance_error(exc) or attempt >= int(max_retries):
                raise
            sleep_s = max(float(retry_sleep_seconds), 0.0) * (2 ** attempt)
            time.sleep(sleep_s)
    if shares is None:
        return pd.Series(dtype="float64")
    if not isinstance(shares, pd.Series):
        shares = pd.Series(shares)
    shares = pd.to_numeric(shares, errors="coerce")
    shares.index = pd.to_datetime(shares.index, errors="coerce").tz_localize(None).normalize()
    shares = shares[shares.index.notna()]
    shares = shares.dropna()
    shares = shares[~shares.index.duplicated(keep="last")]
    shares = shares.sort_index()
    return shares.astype("float64")


def build_market_cap_frame(
    *,
    ticker: str,
    yahoo_symbol: str,
    ohlcv_path: Path,
    start_date: str,
    min_coverage_ratio: float,
    max_retries: int,
    retry_sleep_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    close_df = load_ohlcv_close_series(ohlcv_path, start_date=start_date)
    if close_df.empty:
        raise ValueError("OHLCV has no usable close rows in range.")

    shares = fetch_shares_series(
        yahoo_symbol,
        start_date=start_date,
        max_retries=max_retries,
        retry_sleep_seconds=retry_sleep_seconds,
    )
    if shares.empty:
        raise ValueError("yfinance returned no shares history.")

    out = close_df.copy()
    aligned_shares = shares.reindex(pd.DatetimeIndex(out["date"])).ffill()
    out["shares_outstanding"] = aligned_shares.to_numpy()
    valid_count = int(out["shares_outstanding"].notna().sum())
    coverage_ratio = float(valid_count / len(out)) if len(out) > 0 else 0.0
    if coverage_ratio < float(min_coverage_ratio):
        raise ValueError(
            f"shares coverage too low after forward-fill: {coverage_ratio:.3f} < {float(min_coverage_ratio):.3f}"
        )

    out = out.dropna(subset=["shares_outstanding"]).copy()
    if out.empty:
        raise ValueError("No overlap between OHLCV dates and yfinance shares history after forward-fill.")

    out["market_cap"] = out["close"].astype("float64") * out["shares_outstanding"].astype("float64")
    out["ticker"] = ticker
    out["yahoo_symbol"] = yahoo_symbol
    out = out[["date", "ticker", "yahoo_symbol", "close", "shares_outstanding", "market_cap"]].reset_index(drop=True)

    meta = {
        "ticker": ticker,
        "yahoo_symbol": yahoo_symbol,
        "ohlcv_path": str(ohlcv_path),
        "rows_written": int(len(out)),
        "shares_raw_points": int(len(shares)),
        "coverage_ratio": coverage_ratio,
        "first_date": str(out["date"].min().date()),
        "last_date": str(out["date"].max().date()),
    }
    return out, meta


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers_file = Path(args.tickers_file).expanduser().resolve()
    exclude_tickers_file = Path(args.exclude_tickers_file).expanduser().resolve() if args.exclude_tickers_file else None
    metadata_csv = Path(args.metadata_csv).expanduser().resolve() if args.metadata_csv else None
    ohlcv_dir = Path(args.ohlcv_dir).expanduser().resolve()
    response_dir = Path(args.response_dir).expanduser().resolve() if args.response_dir else None

    tickers = load_tickers(tickers_file)
    original_ticker_count = len(tickers)
    excluded = load_excluded_tickers(exclude_tickers_file)
    tickers = [ticker for ticker in tickers if ticker not in excluded]
    excluded_ticker_count = original_ticker_count - len(tickers)

    metadata_df = load_metadata(metadata_csv)
    tickers, filtered_structure_rows = filter_tickers_by_metadata(tickers, metadata_df)

    response_symbol_count = 0
    filtered_no_news_rows: list[dict[str, str]] = []
    if response_dir is not None:
        response_symbols = collect_response_symbols(response_dir)
        response_symbol_count = int(len(response_symbols))
        tickers, filtered_no_news_rows = filter_tickers_by_response_symbols(tickers, response_symbols)

    tickers, missing_ohlcv_rows = filter_tickers_with_ohlcv(ohlcv_dir, tickers)
    if args.max_tickers > 0:
        tickers = tickers[: int(args.max_tickers)]
    if not tickers:
        raise SystemExit("No tickers left after applying exclusions.")

    state_path = Path(args.state_file).expanduser().resolve() if args.state_file else output_dir / "yfinance_market_cap_state.json"
    summary_path = Path(args.summary_file).expanduser().resolve() if args.summary_file else output_dir / "summary.json"
    failures_path = output_dir / "tickers_failed.csv"
    no_data_path = output_dir / "tickers_without_data.csv"
    coverage_path = output_dir / "coverage_summary.csv"
    missing_ohlcv_path = output_dir / "tickers_missing_ohlcv.csv"
    filtered_structure_path = output_dir / "tickers_filtered_unit_warrant_rights.csv"
    filtered_no_news_path = output_dir / "tickers_filtered_no_response_news.csv"

    if args.reset_state and state_path.exists():
        state_path.unlink()

    state = read_json(state_path) or default_state(output_dir=output_dir, tickers=tickers, start_date=args.start_date)
    state["stats"]["tickers_missing_ohlcv"] = int(len(missing_ohlcv_rows))
    failed_rows: list[dict[str, Any]] = []
    no_data_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    if filtered_structure_rows:
        pd.DataFrame(filtered_structure_rows).to_csv(filtered_structure_path, index=False)
    if filtered_no_news_rows:
        pd.DataFrame(filtered_no_news_rows).to_csv(filtered_no_news_path, index=False)
    if missing_ohlcv_rows:
        pd.DataFrame(missing_ohlcv_rows).to_csv(missing_ohlcv_path, index=False)

    progress = make_progress_bar(total=len(tickers), desc="Tickers", unit="ticker")
    start_idx = int(state.get("current_ticker_index", 0))
    if start_idx > 0:
        progress.update(start_idx)

    try:
        for idx in range(start_idx, len(tickers)):
            ticker = tickers[idx]
            yahoo_symbol = to_yahoo_symbol(ticker)
            progress.set_description(f"Ticker {ticker}")

            state["current_ticker_index"] = idx
            save_json(state_path, state)

            try:
                ohlcv_path = ohlcv_path_for_ticker(ohlcv_dir, ticker)
                out_df, meta = build_market_cap_frame(
                    ticker=ticker,
                    yahoo_symbol=yahoo_symbol,
                    ohlcv_path=ohlcv_path,
                    start_date=args.start_date,
                    min_coverage_ratio=float(args.min_coverage_ratio),
                    max_retries=int(args.max_retries),
                    retry_sleep_seconds=float(args.retry_sleep_seconds),
                )
                out_path = ticker_output_path(output_dir, ticker)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_df.to_parquet(out_path, index=False)

                coverage_rows.append(meta)
                state["stats"]["tickers_with_data"] += 1
                state["stats"]["rows_written"] += int(len(out_df))
            except ValueError as exc:
                no_data_rows.append(
                    {
                        "ticker": ticker,
                        "yahoo_symbol": yahoo_symbol,
                        "exchange": extract_exchange_from_ticker(ticker),
                        "reason": str(exc),
                    }
                )
                state["stats"]["tickers_without_data"] += 1
            except Exception as exc:
                failed_rows.append(
                    {
                        "ticker": ticker,
                        "yahoo_symbol": yahoo_symbol,
                        "exchange": extract_exchange_from_ticker(ticker),
                        "error": str(exc),
                    }
                )
                state["stats"]["tickers_failed"] += 1

            state["stats"]["tickers_completed"] += 1
            state["current_ticker_index"] = idx + 1
            save_json(state_path, state)

            if failed_rows:
                pd.DataFrame(failed_rows).to_csv(failures_path, index=False)
            if no_data_rows:
                pd.DataFrame(no_data_rows).to_csv(no_data_path, index=False)
            if coverage_rows:
                pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)

            progress.update(1)
            progress.set_postfix(
                completed=state["stats"]["tickers_completed"],
                with_data=state["stats"]["tickers_with_data"],
                without_data=state["stats"]["tickers_without_data"],
                failed=state["stats"]["tickers_failed"],
            )

            if float(args.sleep_seconds) > 0:
                time.sleep(float(args.sleep_seconds))
    finally:
        progress.close()

    summary = {
        "version": 1,
        "config": {
            "tickers_total_before_filters": int(original_ticker_count),
            "start_date": args.start_date,
            "tickers_total": len(tickers),
            "tickers_file": str(tickers_file),
            "exclude_tickers_file": str(exclude_tickers_file) if exclude_tickers_file else None,
            "metadata_csv": str(metadata_csv) if metadata_csv else None,
            "ohlcv_dir": str(ohlcv_dir),
            "response_dir": str(response_dir) if response_dir is not None else None,
            "excluded_ticker_count": int(excluded_ticker_count),
            "tickers_filtered_unit_warrant_rights": int(len(filtered_structure_rows)),
            "tickers_filtered_no_response_news": int(len(filtered_no_news_rows)),
            "tickers_missing_ohlcv_filtered": int(len(missing_ohlcv_rows)),
            "response_symbol_count": int(response_symbol_count),
            "min_coverage_ratio": float(args.min_coverage_ratio),
            "max_retries": int(args.max_retries),
            "retry_sleep_seconds": float(args.retry_sleep_seconds),
        },
        "stats": state["stats"],
        "outputs": {
            "output_dir": str(output_dir),
            "state_file": str(state_path),
            "summary_file": str(summary_path),
            "failures_csv": str(failures_path),
            "no_data_csv": str(no_data_path),
            "coverage_csv": str(coverage_path),
            "missing_ohlcv_csv": str(missing_ohlcv_path),
            "filtered_structure_csv": str(filtered_structure_path),
            "filtered_no_news_csv": str(filtered_no_news_path),
        },
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
