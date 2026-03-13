from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TOKEN_FILE = Path("/home/jyu197/LLMCrossSec/eodhd_api_token")
DEFAULT_ENDPOINT = "https://eodhd.com/api/news"


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
        f"or create {DEFAULT_TOKEN_FILE}.\n"
        "For a limited test, you can also pass --api-token DEMO."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the EODHD news API and summarize how many articles come back "
            "in one request and whether they include ticker symbols."
        )
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="EODHD API token. If omitted, read from EODHD_API_TOKEN or eodhd_api_token file.",
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help=f"Optional text file containing the EODHD API token. Default fallback: {DEFAULT_TOKEN_FILE}",
    )
    parser.add_argument(
        "--symbol",
        default="AAPL.US",
        help="Ticker symbol passed as the `s` parameter. Default: AAPL.US",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional tag passed as the `t` parameter, e.g. technology.",
    )
    parser.add_argument(
        "--from-date",
        default=None,
        help="Optional lower date bound in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--to-date",
        default=None,
        help="Optional upper date bound in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of rows requested from EODHD. Docs say max is 1000. Default: 100",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset for pagination. Default: 0",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds. Default: 30",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save the final JSON response after any local filtering.",
    )
    parser.add_argument(
        "--exact-symbol-count",
        type=int,
        default=None,
        help=(
            "Optional local filter: keep only articles whose `symbols` array has "
            "exactly this many entries. Use 1 for single-symbol news."
        ),
    )
    parser.add_argument(
        "--print-sample",
        type=int,
        default=5,
        help="How many articles to print as a compact sample. Default: 5",
    )
    return parser.parse_args()


def build_query_params(args: argparse.Namespace, api_token: str) -> dict[str, str | int]:
    params: dict[str, str | int] = {
        "api_token": api_token,
        "fmt": "json",
        "limit": args.limit,
        "offset": args.offset,
    }
    if args.symbol:
        params["s"] = args.symbol
    if args.tag:
        params["t"] = args.tag
    if args.from_date:
        params["from"] = args.from_date
    if args.to_date:
        params["to"] = args.to_date
    if "s" not in params and "t" not in params:
        raise SystemExit("EODHD news API requires at least one of --symbol or --tag.")
    return params


def fetch_json(endpoint: str, params: dict[str, str | int], timeout: float) -> Any:
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "LLMCrossSec/probe_eodhd_news_api.py",
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
        raise SystemExit(f"Response was not valid JSON:\n{raw[:1000]}") from exc


def normalize_symbols(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def summarize_articles(articles: list[dict[str, Any]]) -> dict[str, Any]:
    symbol_counter: Counter[str] = Counter()
    with_symbols = 0
    without_symbols = 0
    with_sentiment = 0
    with_tags = 0

    for article in articles:
        symbols = normalize_symbols(article.get("symbols"))
        tags = article.get("tags")
        sentiment = article.get("sentiment")

        if symbols:
            with_symbols += 1
            symbol_counter.update(symbols)
        else:
            without_symbols += 1

        if isinstance(tags, list) and tags:
            with_tags += 1
        if isinstance(sentiment, dict) and sentiment:
            with_sentiment += 1

    return {
        "article_count": len(articles),
        "articles_with_symbols": with_symbols,
        "articles_without_symbols": without_symbols,
        "articles_with_tags": with_tags,
        "articles_with_sentiment": with_sentiment,
        "unique_symbols_found": len(symbol_counter),
        "top_symbols": symbol_counter.most_common(20),
    }


def filter_articles_by_exact_symbol_count(
    articles: list[dict[str, Any]], exact_symbol_count: int | None
) -> list[dict[str, Any]]:
    if exact_symbol_count is None:
        return articles
    return [
        article
        for article in articles
        if len(normalize_symbols(article.get("symbols"))) == exact_symbol_count
    ]


def print_sample_rows(articles: list[dict[str, Any]], sample_size: int) -> None:
    if sample_size <= 0:
        return
    print("\nSample articles:")
    for idx, article in enumerate(articles[:sample_size], start=1):
        symbols = normalize_symbols(article.get("symbols"))
        row = {
            "idx": idx,
            "date": article.get("date"),
            "title": article.get("title"),
            "symbols": symbols,
            "tag_count": len(article.get("tags") or []),
            "has_sentiment": bool(article.get("sentiment")),
        }
        print(json.dumps(row, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    api_token = resolve_api_token(args.api_token, args.token_file)
    params = build_query_params(args, api_token)
    payload = fetch_json(DEFAULT_ENDPOINT, params, args.timeout)

    if not isinstance(payload, list):
        print("Unexpected response shape from EODHD:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        raise SystemExit(1)

    filtered_payload = filter_articles_by_exact_symbol_count(payload, args.exact_symbol_count)
    raw_summary = summarize_articles(payload)
    filtered_summary = summarize_articles(filtered_payload)
    request_meta = {
        "endpoint": DEFAULT_ENDPOINT,
        "query": {
            key: ("***" if key == "api_token" else value)
            for key, value in params.items()
        },
        "local_filter": {
            "exact_symbol_count": args.exact_symbol_count,
        },
    }
    result_summary = {
        "raw_response": raw_summary,
        "after_local_filter": filtered_summary,
    }

    print(json.dumps(request_meta, indent=2, ensure_ascii=False))
    print(json.dumps(result_summary, indent=2, ensure_ascii=False))
    print_sample_rows(filtered_payload, args.print_sample)

    if args.save_json:
        output_path = Path(args.save_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(filtered_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved filtered response to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
