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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_TOKEN_FILE = Path("/home/jyu197/LLMCrossSec/eodhd_api_token")
DEFAULT_ENDPOINT = "https://eodhd.com/api/news"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_LIMIT = 1000
DEFAULT_SHARD_SIZE = 2048
DEFAULT_RETRY_ATTEMPTS = 5
DEFAULT_RETRY_BACKOFF = 2.0
DEFAULT_MAX_REQUESTS_PER_MINUTE = 199


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
        description=(
            "Fetch EODHD news since a start date, keep only single-symbol articles, "
            "and save them into JSONL shards."
        )
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
        help="Ticker to crawl. Can be passed multiple times.",
    )
    parser.add_argument(
        "--tickers-file",
        default=None,
        help="Optional file containing the ticker universe. Supports txt/csv/json.",
    )
    parser.add_argument(
        "--allowed-exchange",
        action="append",
        default=None,
        help=(
            "Optional exchange filter applied to the input universe, e.g. "
            "--allowed-exchange NYSE --allowed-exchange NASDAQ."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where JSONL shards and crawl state will be saved.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Fetch news with date >= this value. Default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional upper date bound in YYYY-MM-DD format. Default: today.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Page size for each API request. Max supported by EODHD is {DEFAULT_LIMIT}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds. Default: 30",
    )
    parser.add_argument(
        "--max-records-per-file",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help=(
            "How many kept articles to store per finalized JSONL shard before rolling "
            f"to the next shard. Default: {DEFAULT_SHARD_SIZE}"
        ),
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Optional custom path for crawl state JSON. Default: <output-dir>/crawl_state.json",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Optional custom path for summary JSON. Default: <output-dir>/summary.json",
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
        help="Optional sleep between page requests. Default: 0",
    )
    parser.add_argument(
        "--max-requests-per-minute",
        type=float,
        default=DEFAULT_MAX_REQUESTS_PER_MINUTE,
        help=(
            "Client-side rate limit across all news API HTTP attempts, including retries. "
            f"Default: {DEFAULT_MAX_REQUESTS_PER_MINUTE}"
        ),
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Delete any existing crawl state and start from scratch. Does not delete output shards.",
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


def to_eodhd_news_symbol(ticker: str) -> str:
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
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        original_fieldnames = reader.fieldnames or []
        normalized_to_original = {name.lower(): name for name in original_fieldnames}
        fieldnames = list(normalized_to_original)
        candidate_field = next(
            (name for name in ("ticker", "symbol", "code") if name in fieldnames),
            None,
        )
        if candidate_field is None:
            raise ValueError(
                f"Could not find a ticker column in {path}. Expected one of: ticker, symbol, code."
            )
        tickers = []
        for row in reader:
            original_field = normalized_to_original[candidate_field]
            value = row.get(original_field)
            if value and value.strip():
                tickers.append(normalize_ticker(value))
        return tickers


def load_tickers_from_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tickers: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                value = item
            elif isinstance(item, dict):
                value = item.get("ticker") or item.get("symbol") or item.get("code")
            else:
                value = None
            if value and str(value).strip():
                tickers.append(normalize_ticker(str(value)))
        return tickers
    raise ValueError(f"Expected a JSON list in {path}")


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

    if args.tickers_file:
        path = Path(args.tickers_file).expanduser().resolve()
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
    if end_date:
        return end_date
    return date.today().isoformat()


def normalize_symbols(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_ticker(str(item)) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [normalize_ticker(value)]
    return []


def build_query_params(api_token: str, ticker: str, start_date: str, end_date: str, limit: int, offset: int) -> dict[str, str | int]:
    return {
        "api_token": api_token,
        "fmt": "json",
        "s": to_eodhd_news_symbol(ticker),
        "from": start_date,
        "to": end_date,
        "limit": limit,
        "offset": offset,
    }


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
    endpoint: str,
    params: dict[str, str | int],
    timeout: float,
    retry_attempts: int,
    retry_backoff_seconds: float,
    rate_limiter: RequestRateLimiter,
) -> Any:
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "LLMCrossSec/fetch_eodhd_single_symbol_news.py",
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
            raise SystemExit(f"HTTP {exc.code} from EODHD:\n{detail}") from exc
        except urllib.error.URLError as exc:
            if attempt < retry_attempts:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
                continue
            raise SystemExit(f"Failed to reach EODHD: {exc}") from exc
        except (TimeoutError, socket.timeout) as exc:
            if attempt < retry_attempts:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
                continue
            raise SystemExit(f"Timed out while reading EODHD response: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SystemExit("Response from EODHD was not valid JSON.") from exc

    raise AssertionError("unreachable")


def default_state(output_dir: Path, tickers: list[str], start_date: str, end_date: str, limit: int) -> dict[str, Any]:
    return {
        "version": 1,
        "config": {
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
            "tickers_total": len(tickers),
        },
        "current_ticker_index": 0,
        "current_offset": 0,
        "output_shard_index": 0,
        "output_shard_count": 0,
        "stats": {
            "pages_fetched": 0,
            "raw_articles": 0,
            "single_symbol_articles": 0,
            "kept_articles": 0,
            "multi_symbol_articles": 0,
            "zero_symbol_articles": 0,
            "other_single_symbol_articles": 0,
            "tickers_completed": 0,
        },
        "paths": {
            "output_dir": str(output_dir),
        },
    }


def load_state(path: Path, output_dir: Path, tickers: list[str], start_date: str, end_date: str, limit: int, reset_state: bool) -> dict[str, Any]:
    if reset_state and path.exists():
        path.unlink()
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return default_state(output_dir, tickers, start_date, end_date, limit)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def current_shard_path(output_dir: Path, shard_index: int) -> Path:
    return output_dir / f"news_{shard_index:05d}.jsonl"


def current_partial_shard_path(output_dir: Path, shard_index: int) -> Path:
    return output_dir / f"news_{shard_index:05d}.jsonl.part"


def finalize_shard(output_dir: Path, shard_index: int) -> None:
    partial_path = current_partial_shard_path(output_dir, shard_index)
    final_path = current_shard_path(output_dir, shard_index)
    if partial_path.is_file():
        partial_path.replace(final_path)


def flush_buffer(output_dir: Path, buffer: list[dict[str, Any]], state: dict[str, Any], max_records_per_file: int) -> None:
    if not buffer:
        return

    while buffer:
        shard_index = int(state["output_shard_index"])
        shard_count = int(state["output_shard_count"])
        remaining_capacity = max_records_per_file - shard_count
        chunk = buffer[:remaining_capacity]
        shard_path = current_partial_shard_path(output_dir, shard_index)
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        with shard_path.open("a", encoding="utf-8") as handle:
            for article in chunk:
                handle.write(json.dumps(article, ensure_ascii=False) + "\n")

        state["output_shard_count"] = shard_count + len(chunk)
        del buffer[: len(chunk)]

        if int(state["output_shard_count"]) >= max_records_per_file:
            finalize_shard(output_dir, shard_index)
            state["output_shard_index"] = shard_index + 1
            state["output_shard_count"] = 0


def transform_article(article: dict[str, Any], ticker: str) -> dict[str, Any]:
    symbols = normalize_symbols(article.get("symbols"))
    return {
        "symbol": to_eodhd_news_symbol(ticker),
        "source_ticker": ticker,
        "date": article.get("date"),
        "title": article.get("title"),
        "content": article.get("content"),
        "link": article.get("link"),
        "tags": article.get("tags"),
        "sentiment": article.get("sentiment"),
        "raw_symbols": symbols,
    }


def update_summary(path: Path, state: dict[str, Any], tickers: list[str]) -> None:
    summary = {
        "version": state.get("version"),
        "config": state.get("config"),
        "progress": {
            "current_ticker_index": state.get("current_ticker_index"),
            "current_ticker": tickers[state["current_ticker_index"]] if state["current_ticker_index"] < len(tickers) else None,
            "current_offset": state.get("current_offset"),
            "output_shard_index": state.get("output_shard_index"),
            "output_shard_count": state.get("output_shard_count"),
            "finalized_shards": state.get("output_shard_index"),
            "open_partial_shard": (
                f"news_{int(state['output_shard_index']):05d}.jsonl.part"
                if int(state.get("output_shard_count", 0)) > 0
                else None
            ),
        },
        "stats": state.get("stats"),
    }
    save_json(path, summary)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_producer_status(
    state: dict[str, Any],
    tickers: list[str],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    current_ticker_index = int(state.get("current_ticker_index", 0))
    current_ticker = tickers[current_ticker_index] if current_ticker_index < len(tickers) else None
    payload = {
        "status": status,
        "updated_at": utc_now_iso(),
        "current_ticker_index": current_ticker_index,
        "current_ticker": current_ticker,
        "current_offset": int(state.get("current_offset", 0)),
        "tickers_total": int(state.get("config", {}).get("tickers_total", len(tickers))),
        "tickers_completed": int(state.get("stats", {}).get("tickers_completed", 0)),
        "pages_fetched": int(state.get("stats", {}).get("pages_fetched", 0)),
        "raw_articles": int(state.get("stats", {}).get("raw_articles", 0)),
        "kept_articles": int(state.get("stats", {}).get("kept_articles", 0)),
        "finalized_shards": int(state.get("output_shard_index", 0)),
        "open_partial_shard": (
            f"news_{int(state['output_shard_index']):05d}.jsonl.part"
            if int(state.get("output_shard_count", 0)) > 0
            else None
        ),
    }
    if error:
        payload["error"] = error
    return payload


def update_producer_status(
    path: Path,
    state: dict[str, Any],
    tickers: list[str],
    status: str,
    error: str | None = None,
) -> None:
    save_json(path, build_producer_status(state=state, tickers=tickers, status=status, error=error))


def make_progress_bar(total: int, initial: int):
    if tqdm is None:
        return NullProgressBar()
    return tqdm(total=total, initial=initial, desc="Tickers", unit="ticker")


def main() -> None:
    args = parse_args()
    if args.limit < 1 or args.limit > DEFAULT_LIMIT:
        raise SystemExit(f"--limit must be between 1 and {DEFAULT_LIMIT}.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_file = (
        Path(args.state_file).expanduser().resolve()
        if args.state_file
        else output_dir / "crawl_state.json"
    )
    summary_file = (
        Path(args.summary_file).expanduser().resolve()
        if args.summary_file
        else output_dir / "summary.json"
    )
    status_file = output_dir / "producer_status.json"

    api_token = resolve_api_token(args.api_token, args.token_file)
    tickers = load_ticker_universe(args)
    end_date = resolve_end_date(args.end_date)
    rate_limiter = RequestRateLimiter(args.max_requests_per_minute)
    state = load_state(
        state_file,
        output_dir,
        tickers,
        args.start_date,
        end_date,
        args.limit,
        args.reset_state,
    )

    buffer: list[dict[str, Any]] = []
    progress = make_progress_bar(len(tickers), int(state["current_ticker_index"]))
    update_producer_status(status_file, state, tickers, status="running")

    try:
        for ticker_index in range(int(state["current_ticker_index"]), len(tickers)):
            ticker = tickers[ticker_index]
            news_symbol = to_eodhd_news_symbol(ticker)
            offset = int(state["current_offset"]) if ticker_index == int(state["current_ticker_index"]) else 0
            progress.set_description(f"Ticker {ticker}")

            while True:
                params = build_query_params(api_token, ticker, args.start_date, end_date, args.limit, offset)
                payload = fetch_json(
                    DEFAULT_ENDPOINT,
                    params,
                    args.timeout,
                    args.retry_attempts,
                    args.retry_backoff_seconds,
                    rate_limiter,
                )
                if not isinstance(payload, list):
                    raise SystemExit(f"Unexpected response shape for ticker {ticker}: {payload}")

                page_count = len(payload)
                state["stats"]["pages_fetched"] += 1
                state["stats"]["raw_articles"] += page_count

                for article in payload:
                    symbols = normalize_symbols(article.get("symbols"))
                    if not symbols:
                        state["stats"]["zero_symbol_articles"] += 1
                        continue
                    if len(symbols) != 1:
                        state["stats"]["multi_symbol_articles"] += 1
                        continue

                    state["stats"]["single_symbol_articles"] += 1
                    if symbols[0] != news_symbol:
                        state["stats"]["other_single_symbol_articles"] += 1
                        continue

                    buffer.append(transform_article(article, ticker))
                    state["stats"]["kept_articles"] += 1

                flush_buffer(output_dir, buffer, state, args.max_records_per_file)
                state["current_ticker_index"] = ticker_index
                state["current_offset"] = offset + args.limit if page_count == args.limit else 0
                save_json(state_file, state)
                update_summary(summary_file, state, tickers)
                update_producer_status(status_file, state, tickers, status="running")
                progress.set_postfix(
                    offset=offset,
                    pages=state["stats"]["pages_fetched"],
                    kept=state["stats"]["kept_articles"],
                )

                if page_count < args.limit:
                    break

                offset += args.limit
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            state["stats"]["tickers_completed"] += 1
            state["current_ticker_index"] = ticker_index + 1
            state["current_offset"] = 0
            save_json(state_file, state)
            update_summary(summary_file, state, tickers)
            update_producer_status(status_file, state, tickers, status="running")
            progress.update(1)

        flush_buffer(output_dir, buffer, state, args.max_records_per_file)
        if int(state["current_ticker_index"]) >= len(tickers) and int(state["output_shard_count"]) > 0:
            finalize_shard(output_dir, int(state["output_shard_index"]))
            state["output_shard_index"] = int(state["output_shard_index"]) + 1
            state["output_shard_count"] = 0
        save_json(state_file, state)
        update_summary(summary_file, state, tickers)
        update_producer_status(status_file, state, tickers, status="completed")
    except KeyboardInterrupt:
        save_json(state_file, state)
        update_summary(summary_file, state, tickers)
        update_producer_status(status_file, state, tickers, status="stopped")
        raise
    except BaseException as exc:
        save_json(state_file, state)
        update_summary(summary_file, state, tickers)
        update_producer_status(status_file, state, tickers, status="failed", error=str(exc))
        raise
    finally:
        progress.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
