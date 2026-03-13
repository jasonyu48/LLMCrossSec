from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_TOKEN_FILE = Path("/home/jyu197/LLMCrossSec/eodhd_api_token")
DEFAULT_ENDPOINT_TEMPLATE = "https://eodhd.com/api/exchange-symbol-list/{exchange}"


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
            "Fetch an approximate EODHD exchange universe and save both a ticker list "
            "and richer metadata."
        )
    )
    parser.add_argument("--api-token", default=None, help="EODHD API token.")
    parser.add_argument(
        "--token-file",
        default=None,
        help=f"Optional text file containing the EODHD API token. Default fallback: {DEFAULT_TOKEN_FILE}",
    )
    parser.add_argument(
        "--exchange",
        action="append",
        default=None,
        help="Exchange code for EODHD exchange-symbol-list. Can be passed multiple times. Default: NYSE",
    )
    parser.add_argument(
        "--security-type",
        default="common_stock",
        help="EODHD type filter. Default: common_stock",
    )
    parser.add_argument(
        "--include-delisted",
        action="store_true",
        help="Also fetch delisted tickers from the same exchange and merge them in.",
    )
    parser.add_argument(
        "--delisted-only",
        action="store_true",
        help="Write only tickers that appear in the delisted fetch and not in the active fetch.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional filename prefix. Default: <exchange>_<security-type>",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds. Default: 30",
    )
    return parser.parse_args()


def fetch_json(exchange: str, api_token: str, security_type: str, include_only_delisted: bool, timeout: float) -> Any:
    params: dict[str, str | int] = {
        "api_token": api_token,
        "fmt": "json",
        "type": security_type,
    }
    if include_only_delisted:
        params["delisted"] = 1

    url = DEFAULT_ENDPOINT_TEMPLATE.format(exchange=urllib.parse.quote(exchange))
    request_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        request_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "LLMCrossSec/fetch_eodhd_exchange_universe.py",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from EODHD:\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to reach EODHD: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit("Response from EODHD was not valid JSON.") from exc


def normalize_row(row: dict[str, Any], exchange: str, is_delisted_fetch: bool) -> dict[str, Any]:
    code = str(row.get("Code", "")).strip().upper()
    exchange_name = str(row.get("Exchange", "")).strip() or exchange
    return {
        "ticker": code,
        "full_ticker": f"{code}.{exchange_name}" if code else "",
        "name": row.get("Name"),
        "country": row.get("Country"),
        "exchange": exchange_name,
        "currency": row.get("Currency"),
        "type": row.get("Type"),
        "isin": row.get("Isin"),
        "is_delisted_fetch": is_delisted_fetch,
        "raw": row,
    }


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ticker: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = row.get("full_ticker") or row.get("ticker")
        if not ticker:
            continue

        existing = by_ticker.get(ticker)
        if existing is None:
            by_ticker[ticker] = row
            continue

        if existing["is_delisted_fetch"] and not row["is_delisted_fetch"]:
            by_ticker[ticker] = row

    return sorted(by_ticker.values(), key=lambda item: item["ticker"])


def filter_delisted_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_tickers = {(row.get("full_ticker") or row.get("ticker")) for row in rows if not row["is_delisted_fetch"]}
    filtered = [
        row
        for row in rows
        if row["is_delisted_fetch"] and (row.get("full_ticker") or row.get("ticker")) not in active_tickers
    ]
    return sorted(filtered, key=lambda item: item["ticker"])


def write_tickers_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(row["full_ticker"] for row in rows if row["full_ticker"]) + "\n", encoding="utf-8")


def write_metadata_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "ticker",
        "full_ticker",
        "name",
        "country",
        "exchange",
        "currency",
        "type",
        "isin",
        "is_delisted_fetch",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    if args.delisted_only:
        args.include_delisted = True
    exchanges = [str(exchange).strip().upper() for exchange in (args.exchange or ["NYSE"]) if str(exchange).strip()]
    if not exchanges:
        raise SystemExit("At least one --exchange value is required.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix or f"{'_'.join(exchange.lower() for exchange in exchanges)}_{args.security_type}"
    api_token = resolve_api_token(args.api_token, args.token_file)

    active_payload_by_exchange: dict[str, list[Any]] = {}
    delisted_payload_by_exchange: dict[str, list[Any] | None] = {}
    all_rows: list[dict[str, Any]] = []

    for exchange in exchanges:
        active_payload = fetch_json(exchange, api_token, args.security_type, False, args.timeout)
        if not isinstance(active_payload, list):
            raise SystemExit(f"Unexpected response shape for active payload from {exchange}: {active_payload}")
        active_payload_by_exchange[exchange] = active_payload
        all_rows.extend(normalize_row(row, exchange, False) for row in active_payload if isinstance(row, dict))

        delisted_payload = None
        if args.include_delisted:
            delisted_payload = fetch_json(exchange, api_token, args.security_type, True, args.timeout)
            if not isinstance(delisted_payload, list):
                raise SystemExit(f"Unexpected response shape for delisted payload from {exchange}: {delisted_payload}")
            all_rows.extend(normalize_row(row, exchange, True) for row in delisted_payload if isinstance(row, dict))
        delisted_payload_by_exchange[exchange] = delisted_payload

    deduped_rows = dedupe_rows(all_rows)
    output_rows = filter_delisted_only(all_rows) if args.delisted_only else deduped_rows

    tickers_txt = output_dir / f"{prefix}.tickers.txt"
    metadata_csv = output_dir / f"{prefix}.metadata.csv"
    raw_json = output_dir / f"{prefix}.raw.json"
    summary_json = output_dir / f"{prefix}.summary.json"

    write_tickers_txt(tickers_txt, output_rows)
    write_metadata_csv(metadata_csv, output_rows)

    raw_payload = {
        "exchanges": exchanges,
        "active_by_exchange": active_payload_by_exchange,
        "delisted_by_exchange": delisted_payload_by_exchange if args.include_delisted else None,
    }
    raw_json.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "exchanges": exchanges,
        "security_type": args.security_type,
        "include_delisted": args.include_delisted,
        "delisted_only": args.delisted_only,
        "active_rows": sum(len(payload) for payload in active_payload_by_exchange.values()),
        "delisted_rows": sum(
            len(payload) for payload in delisted_payload_by_exchange.values() if isinstance(payload, list)
        ),
        "deduped_rows": len(deduped_rows),
        "output_rows": len(output_rows),
        "rows_by_exchange": {
            exchange: {
                "active_rows": len(active_payload_by_exchange[exchange]),
                "delisted_rows": len(delisted_payload_by_exchange[exchange])
                if isinstance(delisted_payload_by_exchange[exchange], list)
                else 0,
            }
            for exchange in exchanges
        },
        "outputs": {
            "tickers_txt": str(tickers_txt),
            "metadata_csv": str(metadata_csv),
            "raw_json": str(raw_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
