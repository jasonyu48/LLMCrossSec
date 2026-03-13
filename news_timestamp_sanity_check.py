from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_INPUT_DIR = Path(
    "/export/fs06/jyu197/eodhd/nyse_nasdaq_single_symbol_news_2020_active_delisted_dedup_bow"
)
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_HTML_BYTES = 2_000_000
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

PUBLISHED_META_KEYS = (
    "article:published_time",
    "og:article:published_time",
    "datepublished",
    "publishdate",
    "pubdate",
    "dc.date",
    "dc.date.issued",
    "parsely-pub-date",
)

MODIFIED_META_KEYS = (
    "article:modified_time",
    "og:article:modified_time",
    "datemodified",
    "lastmod",
    "last-modified",
    "dc.modified",
    "parsely-updated",
)

UPDATED_TEXT_PATTERN = re.compile(
    r"(?i)\b(?:updated|last updated|modified)\b[^0-9]{0,20}"
    r"(\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:?\d{2})?)"
)


@dataclass
class NewsRow:
    symbol: str
    source_ticker: str
    db_time_utc: datetime
    title: str
    link: str
    source_file: Path
    line_number: int
    raw_date: str


@dataclass
class PageTimestamps:
    published_time_utc: datetime | None
    modified_time_utc: datetime | None
    published_source: str | None
    modified_source: str | None
    final_url: str


class NullProgressBar:
    def update(self, _: int = 1) -> None:
        pass

    def set_postfix(self, **_: object) -> None:
        pass

    def close(self) -> None:
        pass


class MetadataHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta_tags: list[dict[str, str]] = []
        self.ld_json_blocks: list[str] = []
        self._capture_ld_json = False
        self._ld_json_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key.lower(): (value or "") for key, value in attrs}
        if tag.lower() == "meta":
            self.meta_tags.append(attr_map)
            return
        if tag.lower() == "script":
            script_type = attr_map.get("type", "").strip().lower()
            if script_type == "application/ld+json":
                self._capture_ld_json = True
                self._ld_json_parts = []

    def handle_data(self, data: str) -> None:
        if self._capture_ld_json:
            self._ld_json_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._capture_ld_json:
            self.ld_json_blocks.append("".join(self._ld_json_parts).strip())
            self._capture_ld_json = False
            self._ld_json_parts = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample news rows from the JSONL database, fetch article pages, "
            "extract published/modified timestamps, and print rows where db_time_utc "
            "is earlier than the page's published timestamp."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing news_*.jsonl shards. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"How many rows to sample. Default: {DEFAULT_SAMPLE_SIZE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling. Default: 0",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds. Default: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on scanned files for debugging. 0 means no cap.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on scanned rows for debugging. 0 means no cap.",
    )
    parser.add_argument(
        "--max-html-bytes",
        type=int,
        default=DEFAULT_MAX_HTML_BYTES,
        help=f"Max response bytes to read per page. Default: {DEFAULT_MAX_HTML_BYTES}",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print all sampled comparisons, not only suspicious rows.",
    )
    return parser.parse_args()


def make_progress_bar(total: int | None, desc: str, unit: str):
    if tqdm is None:
        return NullProgressBar()
    return tqdm(total=total, desc=desc, unit=unit, leave=False)


def parse_timestamp_utc(raw: str) -> datetime:
    text = str(raw).strip()
    if not text:
        raise ValueError("empty timestamp")

    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    iso_candidates = [text]
    if " " in text and "T" not in text:
        iso_candidates.append(text.replace(" ", "T", 1))

    for candidate in iso_candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

    if text.isdigit():
        if len(text) == 10:
            return datetime.fromtimestamp(int(text), tz=timezone.utc)
        if len(text) == 13:
            return datetime.fromtimestamp(int(text) / 1000.0, tz=timezone.utc)

    parsed = parsedate_to_datetime(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def discover_news_files(input_dir: Path, max_files: int) -> list[Path]:
    files = sorted(input_dir.glob("news_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No news_*.jsonl files found under {input_dir}")
    if max_files > 0:
        files = files[:max_files]
    return files


def parse_news_row(payload: dict[str, Any], source_file: Path, line_number: int) -> NewsRow | None:
    raw_date = str(payload.get("date", "")).strip()
    link = str(payload.get("link", "")).strip()
    if not raw_date or not link:
        return None
    try:
        db_time_utc = parse_timestamp_utc(raw_date)
    except ValueError:
        return None
    return NewsRow(
        symbol=str(payload.get("symbol", "")).strip(),
        source_ticker=str(payload.get("source_ticker", "")).strip(),
        db_time_utc=db_time_utc,
        title=str(payload.get("title", "")).strip(),
        link=link,
        source_file=source_file,
        line_number=line_number,
        raw_date=raw_date,
    )


def reservoir_sample_rows(
    files: list[Path],
    sample_size: int,
    seed: int,
    max_rows: int,
) -> tuple[list[NewsRow], int, int, int]:
    rng = random.Random(seed)
    sample: list[NewsRow] = []
    scanned_rows = 0
    valid_rows = 0
    bad_rows = 0
    progress = make_progress_bar(total=len(files), desc="Sampling news", unit="file")

    try:
        for path in files:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    scanned_rows += 1
                    if max_rows > 0 and scanned_rows > max_rows:
                        return sample, scanned_rows - 1, valid_rows, bad_rows

                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        bad_rows += 1
                        continue

                    row = parse_news_row(payload, source_file=path, line_number=line_number)
                    if row is None:
                        bad_rows += 1
                        continue

                    valid_rows += 1
                    if sample_size <= 0:
                        continue
                    if len(sample) < sample_size:
                        sample.append(row)
                        continue
                    slot = rng.randint(0, valid_rows - 1)
                    if slot < sample_size:
                        sample[slot] = row

            progress.update(1)
            progress.set_postfix(scanned_rows=scanned_rows, valid_rows=valid_rows, sample=len(sample))
    finally:
        progress.close()

    return sample, scanned_rows, valid_rows, bad_rows


def normalize_meta_key(value: str) -> str:
    return value.strip().lower()


def try_parse_candidate(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        raw = str(int(raw))
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return parse_timestamp_utc(text)
    except (TypeError, ValueError):
        return None


def extract_first_jsonld_timestamp(
    node: Any,
    target_keys: tuple[str, ...],
    path: str = "$",
) -> tuple[datetime | None, str | None]:
    normalized_targets = {key.lower() for key in target_keys}

    if isinstance(node, dict):
        for key, value in node.items():
            key_lower = key.lower()
            if key_lower in normalized_targets:
                parsed = try_parse_candidate(value)
                if parsed is not None:
                    return parsed, f"jsonld:{path}.{key}"
            child_dt, child_source = extract_first_jsonld_timestamp(value, target_keys, path=f"{path}.{key}")
            if child_dt is not None:
                return child_dt, child_source
        return None, None

    if isinstance(node, list):
        for idx, item in enumerate(node):
            child_dt, child_source = extract_first_jsonld_timestamp(item, target_keys, path=f"{path}[{idx}]")
            if child_dt is not None:
                return child_dt, child_source
        return None, None

    return None, None


def extract_meta_timestamp(meta_tags: list[dict[str, str]], keys: tuple[str, ...]) -> tuple[datetime | None, str | None]:
    target_keys = {normalize_meta_key(key) for key in keys}
    for attrs in meta_tags:
        key = (
            attrs.get("property")
            or attrs.get("name")
            or attrs.get("itemprop")
            or attrs.get("http-equiv")
            or ""
        )
        key = normalize_meta_key(key)
        if key not in target_keys:
            continue
        parsed = try_parse_candidate(attrs.get("content", ""))
        if parsed is not None:
            return parsed, f"meta:{key}"
    return None, None


def extract_text_timestamp(html: str, pattern: re.Pattern[str], label: str) -> tuple[datetime | None, str | None]:
    match = pattern.search(html)
    if not match:
        return None, None
    parsed = try_parse_candidate(match.group(1))
    if parsed is None:
        return None, None
    return parsed, label


def fetch_page_html(url: str, timeout: float, max_html_bytes: int) -> tuple[str, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw_bytes = response.read(max_html_bytes + 1)
        if len(raw_bytes) > max_html_bytes:
            raw_bytes = raw_bytes[:max_html_bytes]
        charset = response.headers.get_content_charset() or "utf-8"
        html = raw_bytes.decode(charset, errors="replace")
        return html, response.geturl()


def extract_page_timestamps(url: str, timeout: float, max_html_bytes: int) -> PageTimestamps:
    html, final_url = fetch_page_html(url, timeout=timeout, max_html_bytes=max_html_bytes)
    parser = MetadataHTMLParser()
    parser.feed(html)

    published_dt, published_source = extract_meta_timestamp(parser.meta_tags, PUBLISHED_META_KEYS)
    modified_dt, modified_source = extract_meta_timestamp(parser.meta_tags, MODIFIED_META_KEYS)

    if published_dt is None:
        for block_idx, block in enumerate(parser.ld_json_blocks):
            block = unescape(block).strip()
            if not block:
                continue
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                continue
            published_dt, published_source = extract_first_jsonld_timestamp(
                payload,
                ("datePublished",),
                path=f"$[{block_idx}]",
            )
            if published_dt is not None:
                break

    if modified_dt is None:
        for block_idx, block in enumerate(parser.ld_json_blocks):
            block = unescape(block).strip()
            if not block:
                continue
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                continue
            modified_dt, modified_source = extract_first_jsonld_timestamp(
                payload,
                ("dateModified",),
                path=f"$[{block_idx}]",
            )
            if modified_dt is not None:
                break

    if modified_dt is None:
        modified_dt, modified_source = extract_text_timestamp(
            html,
            UPDATED_TEXT_PATTERN,
            "html:updated_text_fallback",
        )

    return PageTimestamps(
        published_time_utc=published_dt,
        modified_time_utc=modified_dt,
        published_source=published_source,
        modified_source=modified_source,
        final_url=final_url,
    )


def format_dt(value: datetime | None) -> str:
    if value is None:
        return "None"
    return value.isoformat().replace("+00:00", "Z")


def format_hours(delta_seconds: float) -> str:
    return f"{delta_seconds / 3600.0:.2f}h"


def print_result(row: NewsRow, page: PageTimestamps, only_if_suspicious: bool) -> bool:
    published = page.published_time_utc
    suspicious = published is not None and row.db_time_utc < published
    if only_if_suspicious and not suspicious:
        return False

    if published is None:
        status = "unknown"
    elif suspicious:
        status = "suspicious"
    else:
        status = "ok"

    delta_seconds = None if published is None else (published - row.db_time_utc).total_seconds()
    print("=" * 80)
    print(f"title: {row.title}")
    print(f"symbol: {row.symbol} ({row.source_ticker})")
    print(f"db_time_utc: {format_dt(row.db_time_utc)}")
    print(f"published_time_utc: {format_dt(page.published_time_utc)}")
    print(f"published_source: {page.published_source}")
    print(f"modified_time_utc: {format_dt(page.modified_time_utc)}")
    print(f"modified_source: {page.modified_source}")
    print(f"status: {status}")
    if delta_seconds is not None:
        print(f"published_minus_db: {format_hours(delta_seconds)}")
        print(
            "published_minus_db_note: positive means db_time_utc is earlier than "
            "published_time_utc, which is suspicious"
        )
    print(f"link: {row.link}")
    print(f"final_url: {page.final_url}")
    print(f"sample_row: {row.source_file.name}:{row.line_number}")
    return suspicious


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    files = discover_news_files(input_dir, max_files=int(args.max_files))
    sample, scanned_rows, valid_rows, bad_rows = reservoir_sample_rows(
        files=files,
        sample_size=int(args.sample_size),
        seed=int(args.seed),
        max_rows=int(args.max_rows),
    )

    checked = 0
    suspicious_count = 0
    fetch_errors = 0
    no_published_count = 0
    progress = make_progress_bar(total=len(sample), desc="Checking pages", unit="page")

    try:
        for idx, row in enumerate(sample, start=1):
            try:
                page = extract_page_timestamps(
                    row.link,
                    timeout=float(args.timeout),
                    max_html_bytes=int(args.max_html_bytes),
                )
            except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
                fetch_errors += 1
                if args.print_all:
                    print("=" * 80)
                    print(f"title: {row.title}")
                    print(f"symbol: {row.symbol} ({row.source_ticker})")
                    print(f"db_time_utc: {format_dt(row.db_time_utc)}")
                    print("status: fetch_error")
                    print(f"error: {exc}")
                    print(f"link: {row.link}")
                    print(f"sample_row: {row.source_file.name}:{row.line_number}")
                progress.update(1)
                progress.set_postfix(
                    checked=checked,
                    suspicious=suspicious_count,
                    fetch_errors=fetch_errors,
                )
                continue

            checked += 1
            if page.published_time_utc is None:
                no_published_count += 1

            is_suspicious = print_result(
                row=row,
                page=page,
                only_if_suspicious=not bool(args.print_all),
            )
            if is_suspicious:
                suspicious_count += 1
            progress.update(1)
            progress.set_postfix(
                checked=checked,
                suspicious=suspicious_count,
                fetch_errors=fetch_errors,
            )
    finally:
        progress.close()

    if args.print_all or suspicious_count > 0:
        print("=" * 80)
        print("summary")
        print(f"input_dir: {input_dir}")
        print(f"files_scanned: {len(files)}")
        print(f"rows_scanned: {scanned_rows}")
        print(f"valid_rows_seen: {valid_rows}")
        print(f"bad_rows_skipped: {bad_rows}")
        print(f"sample_size_requested: {args.sample_size}")
        print(f"sample_size_checked: {len(sample)}")
        print(f"pages_fetched: {checked}")
        print(f"fetch_errors: {fetch_errors}")
        print(f"rows_without_published_time: {no_published_count}")
        print(f"suspicious_db_earlier_than_published: {suspicious_count}")


if __name__ == "__main__":
    main()
