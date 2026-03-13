from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
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
DEFAULT_ENDPOINT_TEMPLATE = "https://eodhd.com/api/eod/{ticker}"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRY_ATTEMPTS = 8
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_MAX_REQUESTS_PER_MINUTE = 999


class NullProgressBar:
    def update(self, _: int = 1) -> None:
        pass

    def set_description(self, _: str) -> None:
        pass

    def set_postfix(self, **_: object) -> None:
        pass

    def close(self) -> None:
        pass


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
    parser = argparse.ArgumentParser(
        description="Download EODHD daily OHLCV history by ticker with resume support."
    )
    parser.add_argument("--api-token", default=None, help="EODHD API token.")
    parser.add_argument(
        "--token-file",
        default=None,
        help=f"Optional text file containing the EODHD API token. Default fallback: {DEFAULT_TOKEN_FILE}",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        default=None,
        help="Ticker to download. Can be passed multiple times.",
    )
    parser.add_argument(
        "--tickers-file",
        action="append",
        default=None,
        help="Optional file containing tickers. Supports txt/csv/json. Can be passed multiple times.",
    )
    parser.add_argument(
        "--allowed-exchange",
        action="append",
        default=None,
        help="Optional exchange filter, e.g. --allowed-exchange NYSE --allowed-exchange NASDAQ.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-ticker parquet files and state files will be written.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Download rows with date >= this value. Default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional upper date bound in YYYY-MM-DD format. Default: today.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Optional custom path for crawl state JSON. Default: <output-dir>/ohlcv_state.json",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Optional custom path for summary JSON. Default: <output-dir>/summary.json",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds. Default: {DEFAULT_TIMEOUT}",
    )
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
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between ticker requests. Default: 0",
    )
    parser.add_argument(
        "--max-requests-per-minute",
        type=float,
        default=DEFAULT_MAX_REQUESTS_PER_MINUTE,
        help=(
            "Client-side rate limit across all HTTP attempts, including retries. "
            f"Default: {DEFAULT_MAX_REQUESTS_PER_MINUTE}"
        ),
    )
    parser.add_argument(
        "--adjusted",
        action="store_true",
        help="If set, request adjusted historical prices where supported by EODHD.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Delete any existing crawl state and start from scratch. Does not delete output files.",
    )
    return parser.parse_args()


def normalize_ticker(value: str) -> str:
    return value.strip().upper()


def normalize_exchange(value: str) -> str:
    return value.strip().upper()


def extract_exchange_from_ticker(ticker: str) -> str:
    if "." not in ticker:
        return ""
    return normalize_exchange(ticker.rsplit(".", 1)[1])


def to_eodhd_price_symbol(ticker: str) -> str:
    if "." not in ticker:
        return normalize_ticker(ticker)
    base, _exchange = ticker.rsplit(".", 1)
    return f"{normalize_ticker(base)}.US"


def load_tickers_from_text(path: Path) -> list[str]:
    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        tickers.append(normalize_ticker(text.split(",")[0]))
    return tickers


def load_tickers_from_csv(path: Path) -> list[str]:
    import csv

    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        original_fieldnames = reader.fieldnames or []
        normalized_to_original = {name.lower(): name for name in original_fieldnames}
        candidate_field = next(
            (name for name in ("ticker", "symbol", "code", "full_ticker") if name in normalized_to_original),
            None,
        )
        if candidate_field is None:
            raise ValueError(
                f"Could not find a ticker column in {path}. Expected one of: ticker, symbol, code, full_ticker."
            )
        original_field = normalized_to_original[candidate_field]
        out: list[str] = []
        for row in reader:
            value = row.get(original_field)
            if value and value.strip():
                out.append(normalize_ticker(value))
        return out


def load_tickers_from_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    for item in payload:
        if isinstance(item, str):
            value = item
        elif isinstance(item, dict):
            value = item.get("ticker") or item.get("symbol") or item.get("code") or item.get("full_ticker")
        else:
            value = None
        if value and str(value).strip():
            out.append(normalize_ticker(str(value)))
    return out


def filter_tickers_by_allowed_exchanges(
    tickers: list[str], allowed_exchanges: set[str] | None
) -> list[str]:
    if not allowed_exchanges:
        return tickers
    filtered: list[str] = []
    for ticker in tickers:
        exchange = extract_exchange_from_ticker(ticker)
        if exchange and exchange in allowed_exchanges:
            filtered.append(ticker)
    return filtered


def load_ticker_universe(args: argparse.Namespace) -> list[str]:
    tickers: list[str] = []
    for value in args.ticker or []:
        if value.strip():
            tickers.append(normalize_ticker(value))

    for raw_path in args.tickers_file or []:
        path = Path(raw_path).expanduser().resolve()
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            tickers.extend(load_tickers_from_csv(path))
        elif suffix == ".json":
            tickers.extend(load_tickers_from_json(path))
        else:
            tickers.extend(load_tickers_from_text(path))

    deduped: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        if ticker and ticker not in seen:
            deduped.append(ticker)
            seen.add(ticker)

    allowed_exchanges = (
        {normalize_exchange(value) for value in args.allowed_exchange or [] if value.strip()}
        if args.allowed_exchange
        else None
    )
    deduped = filter_tickers_by_allowed_exchanges(deduped, allowed_exchanges)

    if not deduped:
        raise SystemExit("No tickers provided. Pass --ticker and/or --tickers-file.")
    return deduped


def resolve_end_date(end_date: str | None) -> str:
    return end_date if end_date else date.today().isoformat()


class RequestRateLimiter:
    def __init__(self, max_requests_per_minute: float):
        if max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be positive")
        self.min_interval_seconds = 60.0 / max_requests_per_minute
        self.next_allowed_time = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self.next_allowed_time:
            time.sleep(self.next_allowed_time - now)
            now = time.monotonic()
        self.next_allowed_time = now + self.min_interval_seconds


def fetch_json(
    ticker: str,
    start_date: str,
    end_date: str,
    api_token: str,
    adjusted: bool,
    timeout: float,
    retry_attempts: int,
    retry_backoff_seconds: float,
    rate_limiter: RequestRateLimiter,
) -> Any:
    request_ticker = to_eodhd_price_symbol(ticker)
    params: dict[str, str | int] = {
        "api_token": api_token,
        "fmt": "json",
        "from": start_date,
        "to": end_date,
        "period": "d",
    }
    if adjusted:
        params["adjusted"] = 1

    url = DEFAULT_ENDPOINT_TEMPLATE.format(ticker=urllib.parse.quote(request_ticker))
    request_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        request_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "LLMCrossSec/fetch_eodhd_ohlcv_by_ticker.py",
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
            raise SystemExit(f"HTTP {exc.code} from EODHD for {request_ticker}:\n{detail}") from exc
        except urllib.error.URLError as exc:
            if attempt < retry_attempts:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
                continue
            raise SystemExit(f"Failed to reach EODHD for {request_ticker}: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            if attempt < retry_attempts:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
                continue
            raise SystemExit(f"Timed out while reading EODHD response for {request_ticker}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Response from EODHD for {request_ticker} was not valid JSON.") from exc

    raise AssertionError("unreachable")


def sanitize_filename(text: str) -> str:
    return text.replace("/", "_")


def ticker_output_path(output_dir: Path, ticker: str) -> Path:
    exchange = extract_exchange_from_ticker(ticker) or "UNKNOWN"
    exchange_dir = output_dir / "parquet" / exchange
    return exchange_dir / f"{sanitize_filename(ticker)}.parquet"


def normalize_ohlcv_rows(rows: list[dict[str, Any]], ticker: str) -> pd.DataFrame:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_rows.append(
            {
                "ticker": ticker,
                "requested_symbol": to_eodhd_price_symbol(ticker),
                "date": str(row.get("date", "")),
                "open": row.get("open", row.get("Open")),
                "high": row.get("high", row.get("High")),
                "low": row.get("low", row.get("Low")),
                "close": row.get("close", row.get("Close")),
                "adjusted_close": row.get("adjusted_close", row.get("Adjusted_Close", row.get("adjusted_close"))),
                "volume": row.get("volume", row.get("Volume")),
            }
        )
    df = pd.DataFrame(normalized_rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("open", "high", "low", "close", "adjusted_close", "volume"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def read_json(path: Path) -> dict | None:
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


def load_state(
    path: Path,
    output_dir: Path,
    tickers: list[str],
    start_date: str,
    end_date: str,
    reset_state: bool,
) -> dict[str, Any]:
    if reset_state and path.exists():
        path.unlink()
    payload = read_json(path)
    if payload is not None:
        return payload
    return default_state(output_dir, tickers, start_date, end_date)


def update_summary(path: Path, state: dict[str, Any], tickers: list[str]) -> None:
    idx = int(state.get("current_ticker_index", 0))
    summary = {
        "version": state.get("version"),
        "config": state.get("config"),
        "progress": {
            "current_ticker_index": idx,
            "current_ticker": tickers[idx] if idx < len(tickers) else None,
            "tickers_total": len(tickers),
            "tickers_remaining": max(0, len(tickers) - idx),
        },
        "stats": state.get("stats"),
    }
    save_json(path, summary)


def ensure_no_data_csv(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ticker",
                "requested_symbol",
                "exchange",
                "reason",
                "requested_start_date",
                "requested_end_date",
            ],
        )
        writer.writeheader()


def append_no_data_row(
    path: Path,
    *,
    ticker: str,
    start_date: str,
    end_date: str,
) -> None:
    ensure_no_data_csv(path)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ticker",
                "requested_symbol",
                "exchange",
                "reason",
                "requested_start_date",
                "requested_end_date",
            ],
        )
        writer.writerow(
            {
                "ticker": ticker,
                "requested_symbol": to_eodhd_price_symbol(ticker),
                "exchange": extract_exchange_from_ticker(ticker),
                "reason": "empty_response",
                "requested_start_date": start_date,
                "requested_end_date": end_date,
            }
        )


def make_progress_bar(total: int, initial: int):
    if tqdm is None:
        return NullProgressBar()
    return tqdm(total=total, initial=initial, desc="Tickers", unit="ticker")


def main() -> None:
    args = parse_args()
    api_token = resolve_api_token(args.api_token, args.token_file)
    tickers = load_ticker_universe(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = (
        Path(args.state_file).expanduser().resolve()
        if args.state_file
        else output_dir / "ohlcv_state.json"
    )
    summary_file = (
        Path(args.summary_file).expanduser().resolve()
        if args.summary_file
        else output_dir / "summary.json"
    )
    no_data_csv = output_dir / "tickers_without_data.csv"
    end_date = resolve_end_date(args.end_date)
    state = load_state(
        state_file,
        output_dir,
        tickers,
        args.start_date,
        end_date,
        args.reset_state,
    )
    rate_limiter = RequestRateLimiter(args.max_requests_per_minute)
    progress = make_progress_bar(len(tickers), int(state["current_ticker_index"]))

    try:
        for ticker_index in range(int(state["current_ticker_index"]), len(tickers)):
            ticker = tickers[ticker_index]
            progress.set_description(f"Ticker {ticker}")

            payload = fetch_json(
                ticker=ticker,
                start_date=args.start_date,
                end_date=end_date,
                api_token=api_token,
                adjusted=args.adjusted,
                timeout=args.timeout,
                retry_attempts=args.retry_attempts,
                retry_backoff_seconds=args.retry_backoff_seconds,
                rate_limiter=rate_limiter,
            )
            if not isinstance(payload, list):
                raise SystemExit(f"Unexpected response shape for {ticker}: {payload}")

            df = normalize_ohlcv_rows(payload, ticker=ticker)
            output_path = ticker_output_path(output_dir, ticker)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not df.empty:
                df.to_parquet(output_path, index=False)
                state["stats"]["tickers_with_data"] += 1
                state["stats"]["rows_downloaded"] += len(df)
            else:
                state["stats"]["tickers_without_data"] += 1
                append_no_data_row(
                    no_data_csv,
                    ticker=ticker,
                    start_date=args.start_date,
                    end_date=end_date,
                )

            state["stats"]["requests_made"] += 1
            state["stats"]["tickers_completed"] += 1
            state["current_ticker_index"] = ticker_index + 1
            save_json(state_file, state)
            update_summary(summary_file, state, tickers)
            progress.set_postfix(
                completed=state["stats"]["tickers_completed"],
                rows=state["stats"]["rows_downloaded"],
                with_data=state["stats"]["tickers_with_data"],
            )
            progress.update(1)

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    finally:
        progress.close()

    update_summary(summary_file, state, tickers)
    print(json.dumps(read_json(summary_file) or {}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
