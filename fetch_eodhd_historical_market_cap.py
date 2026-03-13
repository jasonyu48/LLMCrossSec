from __future__ import annotations

import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_TOKEN_FILE = Path("/home/jyu197/LLMCrossSec/eodhd_api_token")
DEFAULT_ENDPOINT_TEMPLATE = "https://eodhd.com/api/historical-market-cap/{ticker}"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRY_ATTEMPTS = 8
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_MAX_REQUESTS_PER_MINUTE = 999
DEFAULT_TICKERS_FILE = Path("/home/jyu197/LLMCrossSec/us_universe_merged/us_nyse_nasdaq_common_stock.tickers.txt")
DEFAULT_EXCLUDE_TICKERS_FILE = Path(
    "/export/fs06/jyu197/eodhd/ohlcv_nyse_nasdaq_active_delisted_since_20200101_v2/tickers_without_data.csv"
)


class NullProgressBar:
    def update(self, _: int = 1) -> None:
        pass

    def set_description(self, _: str) -> None:
        pass

    def set_postfix(self, **_: object) -> None:
        pass

    def close(self) -> None:
        pass


class RequestRateLimiter:
    def __init__(self, max_requests_per_minute: float) -> None:
        self.interval_seconds = 0.0 if max_requests_per_minute <= 0 else 60.0 / max_requests_per_minute
        self.next_allowed_ts = 0.0

    def wait(self) -> None:
        if self.interval_seconds <= 0:
            return
        now = time.time()
        if now < self.next_allowed_ts:
            time.sleep(self.next_allowed_ts - now)
        self.next_allowed_ts = max(time.time(), self.next_allowed_ts) + self.interval_seconds


def make_progress_bar(total: int | None, desc: str, unit: str):
    if tqdm is None:
        return NullProgressBar()
    return tqdm(total=total, desc=desc, unit=unit, leave=False)


def resolve_api_token(explicit_token: str | None, token_file: str | None) -> str:
    if explicit_token:
        return explicit_token.strip()

    for env_name in ("EODHD_API_TOKEN", "API_EODHD_TOKEN"):
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()

    candidate_paths: list[Path] = []
    if token_file:
        candidate_paths.append(Path(token_file).expanduser().resolve())
    candidate_paths.append(DEFAULT_TOKEN_FILE)

    for path in candidate_paths:
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value

    raise SystemExit(
        "No EODHD API token found. Pass --api-token, set EODHD_API_TOKEN, "
        f"or create {DEFAULT_TOKEN_FILE}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download EODHD historical market cap by ticker with resume support.")
    parser.add_argument("--api-token", default=None, help="EODHD API token.")
    parser.add_argument(
        "--token-file",
        default=str(DEFAULT_TOKEN_FILE),
        help=f"Optional text file containing the EODHD API token. Default: {DEFAULT_TOKEN_FILE}",
    )
    parser.add_argument(
        "--tickers-file",
        default=str(DEFAULT_TICKERS_FILE),
        help=f"Ticker list to request. Default: {DEFAULT_TICKERS_FILE}",
    )
    parser.add_argument(
        "--exclude-tickers-file",
        default=str(DEFAULT_EXCLUDE_TICKERS_FILE),
        help="CSV file of tickers to exclude, e.g. OHLCV tickers_without_data.csv.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where per-ticker parquet files and state files are written.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help=f"Download rows with date >= this value. Default: {DEFAULT_START_DATE}")
    parser.add_argument("--end-date", default=None, help="Optional upper date bound in YYYY-MM-DD format. Default: today.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help=f"HTTP timeout in seconds. Default: {DEFAULT_TIMEOUT}")
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help=f"How many times to retry transient HTTP failures. Default: {DEFAULT_RETRY_ATTEMPTS}",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=DEFAULT_RETRY_BACKOFF,
        help=f"Base backoff for retries. Default: {DEFAULT_RETRY_BACKOFF}",
    )
    parser.add_argument(
        "--max-requests-per-minute",
        type=float,
        default=DEFAULT_MAX_REQUESTS_PER_MINUTE,
        help=f"Client-side rate limit across all HTTP attempts. Default: {DEFAULT_MAX_REQUESTS_PER_MINUTE}",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional sleep between ticker requests. Default: 0")
    parser.add_argument("--state-file", default=None, help="Optional custom path for crawl state JSON.")
    parser.add_argument("--summary-file", default=None, help="Optional custom path for summary JSON.")
    parser.add_argument("--reset-state", action="store_true", help="Delete any existing crawl state and start from scratch.")
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional cap on tickers processed for debugging. 0 means no cap.")
    return parser.parse_args()


def normalize_ticker(value: str) -> str:
    return value.strip().upper()


def extract_exchange_from_ticker(ticker: str) -> str:
    if "." not in ticker:
        return ""
    return ticker.rsplit(".", 1)[1].strip().upper()


def to_eodhd_market_cap_symbol(ticker: str) -> str:
    if "." not in ticker:
        return normalize_ticker(ticker)
    base, _exchange = ticker.rsplit(".", 1)
    return f"{normalize_ticker(base)}.US"


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


def default_state(output_dir: Path, tickers: list[str], start_date: str, end_date: str) -> dict[str, Any]:
    return {
        "version": 1,
        "config": {
            "start_date": start_date,
            "end_date": end_date,
            "tickers_total": len(tickers),
        },
        "current_ticker_index": 0,
        "stats": {
            "requests_made": 0,
            "tickers_completed": 0,
            "tickers_with_data": 0,
            "tickers_without_data": 0,
            "rows_downloaded": 0,
        },
        "paths": {
            "output_dir": str(output_dir),
        },
    }


def fetch_market_cap_rows(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    api_token: str,
    timeout: float,
    retry_attempts: int,
    retry_backoff_seconds: float,
    rate_limiter: RequestRateLimiter,
) -> Any:
    request_ticker = to_eodhd_market_cap_symbol(ticker)
    params: dict[str, str] = {
        "api_token": api_token,
        "fmt": "json",
        "from": start_date,
        "to": end_date,
    }
    url = DEFAULT_ENDPOINT_TEMPLATE.format(ticker=urllib.parse.quote(request_ticker))
    request_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        request_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "LLMCrossSec/fetch_eodhd_historical_market_cap.py",
        },
    )

    for attempt in range(1, retry_attempts + 1):
        try:
            rate_limiter.wait()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            is_retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            if attempt < retry_attempts and is_retryable:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
                continue
            raise RuntimeError(f"HTTP {exc.code} from EODHD for {request_ticker}: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt < retry_attempts:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
                continue
            raise RuntimeError(f"Failed to reach EODHD for {request_ticker}: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            if attempt < retry_attempts:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
                continue
            raise RuntimeError(f"Timed out while reading EODHD response for {request_ticker}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Response from EODHD for {request_ticker} was not valid JSON.") from exc

    raise AssertionError("unreachable")


def normalize_market_cap_rows(rows: Any, ticker: str) -> pd.DataFrame:
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected response type for {ticker}: {type(rows)}")

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized_rows.append(
            {
                "ticker": ticker,
                "requested_symbol": to_eodhd_market_cap_symbol(ticker),
                "date": str(row.get("date", "")),
                "market_cap": row.get("market_cap", row.get("marketCapitalization", row.get("MarketCapitalization"))),
            }
        )
    df = pd.DataFrame(normalized_rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce").dt.strftime("%Y-%m-%d")
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df = df.dropna(subset=["date", "market_cap"]).sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers_file = Path(args.tickers_file).expanduser().resolve()
    exclude_tickers_file = Path(args.exclude_tickers_file).expanduser().resolve() if args.exclude_tickers_file else None
    tickers = load_tickers(tickers_file)
    excluded = load_excluded_tickers(exclude_tickers_file)
    tickers = [ticker for ticker in tickers if ticker not in excluded]
    if args.max_tickers > 0:
        tickers = tickers[: int(args.max_tickers)]
    if not tickers:
        raise SystemExit("No tickers left after applying exclusions.")

    end_date = args.end_date or date.today().isoformat()
    api_token = resolve_api_token(args.api_token, args.token_file)
    state_path = Path(args.state_file).expanduser().resolve() if args.state_file else output_dir / "market_cap_state.json"
    summary_path = Path(args.summary_file).expanduser().resolve() if args.summary_file else output_dir / "summary.json"
    failures_path = output_dir / "tickers_failed.csv"
    no_data_path = output_dir / "tickers_without_data.csv"

    if args.reset_state and state_path.exists():
        state_path.unlink()

    state = read_json(state_path) or default_state(output_dir=output_dir, tickers=tickers, start_date=args.start_date, end_date=end_date)
    rate_limiter = RequestRateLimiter(float(args.max_requests_per_minute))

    progress = make_progress_bar(total=len(tickers), desc="Tickers", unit="ticker")
    start_idx = int(state.get("current_ticker_index", 0))
    if start_idx > 0:
        progress.update(start_idx)

    failed_rows: list[dict[str, Any]] = []
    no_data_rows: list[dict[str, Any]] = []
    try:
        for idx in range(start_idx, len(tickers)):
            ticker = tickers[idx]
            progress.set_description(f"Ticker {ticker}")
            state["current_ticker_index"] = idx
            save_json(state_path, state)

            try:
                rows = fetch_market_cap_rows(
                    ticker=ticker,
                    start_date=args.start_date,
                    end_date=end_date,
                    api_token=api_token,
                    timeout=float(args.timeout),
                    retry_attempts=int(args.retry_attempts),
                    retry_backoff_seconds=float(args.retry_backoff_seconds),
                    rate_limiter=rate_limiter,
                )
                state["stats"]["requests_made"] += 1
                df = normalize_market_cap_rows(rows, ticker=ticker)

                if df.empty:
                    state["stats"]["tickers_without_data"] += 1
                    no_data_rows.append(
                        {
                            "ticker": ticker,
                            "requested_symbol": to_eodhd_market_cap_symbol(ticker),
                            "exchange": extract_exchange_from_ticker(ticker),
                            "reason": "empty_response",
                            "requested_start_date": args.start_date,
                            "requested_end_date": end_date,
                        }
                    )
                else:
                    out_path = ticker_output_path(output_dir, ticker)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(out_path, index=False)
                    state["stats"]["tickers_with_data"] += 1
                    state["stats"]["rows_downloaded"] += int(len(df))
            except Exception as exc:
                failed_rows.append(
                    {
                        "ticker": ticker,
                        "requested_symbol": to_eodhd_market_cap_symbol(ticker),
                        "exchange": extract_exchange_from_ticker(ticker),
                        "error": str(exc),
                    }
                )

            state["stats"]["tickers_completed"] += 1
            state["current_ticker_index"] = idx + 1
            save_json(state_path, state)

            if failed_rows:
                pd.DataFrame(failed_rows).to_csv(failures_path, index=False)
            if no_data_rows:
                pd.DataFrame(no_data_rows).to_csv(no_data_path, index=False)

            progress.update(1)
            progress.set_postfix(
                completed=state["stats"]["tickers_completed"],
                with_data=state["stats"]["tickers_with_data"],
                without_data=state["stats"]["tickers_without_data"],
                failed=len(failed_rows),
            )

            if float(args.sleep_seconds) > 0:
                time.sleep(float(args.sleep_seconds))
    finally:
        progress.close()

    summary = {
        "version": 1,
        "config": {
            "start_date": args.start_date,
            "end_date": end_date,
            "tickers_total": len(tickers),
            "tickers_file": str(tickers_file),
            "exclude_tickers_file": str(exclude_tickers_file) if exclude_tickers_file else None,
        },
        "stats": state["stats"],
        "outputs": {
            "output_dir": str(output_dir),
            "state_file": str(state_path),
            "summary_file": str(summary_path),
            "failures_csv": str(failures_path),
            "no_data_csv": str(no_data_path),
        },
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
