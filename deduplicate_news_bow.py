from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import zlib
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
DEFAULT_BUCKET_COUNT = 256
DEFAULT_MAX_RECORDS_PER_FILE = 2048


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
            "Deduplicate raw EODHD single-symbol news using BoW cosine similarity within the same ticker "
            "over the prior five business days."
        )
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Directory containing finalized raw news_*.jsonl shards. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write deduplicated news_*.jsonl shards.")
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=100,
        help="Drop articles whose canonical title/body text is shorter than this many characters.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=100000,
        help="Drop articles whose canonical title/body text is longer than this many characters.",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.8,
        help="Drop if BoW cosine similarity to a prior kept article reaches this threshold.",
    )
    parser.add_argument(
        "--window-business-days",
        type=int,
        default=5,
        help="Compare against kept articles from the same ticker within this many prior business days.",
    )
    parser.add_argument(
        "--bucket-count",
        type=int,
        default=DEFAULT_BUCKET_COUNT,
        help="How many temporary ticker-hash buckets to partition raw articles into.",
    )
    parser.add_argument(
        "--max-records-per-file",
        type=int,
        default=DEFAULT_MAX_RECORDS_PER_FILE,
        help="How many kept articles to store per output news_*.jsonl shard.",
    )
    parser.add_argument(
        "--keep-temp-buckets",
        action="store_true",
        help="If set, keep temporary bucket files under output-dir/_tmp_buckets for inspection.",
    )
    return parser.parse_args()


def resolve_input_dirs(explicit_paths: list[str]) -> list[Path]:
    resolved = [Path(path).expanduser().resolve() for path in explicit_paths]
    if not resolved:
        raise FileNotFoundError("No input directories were provided.")
    for path in resolved:
        if not path.is_dir():
            raise FileNotFoundError(f"Could not find input directory: {path}")
    return resolved


def discover_jsonl_files(input_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_dir in input_dirs:
        files.extend(sorted(input_dir.glob("news_*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No finalized news_*.jsonl files found under: {input_dirs}")
    return files


def build_input_text(article: dict) -> str:
    title = str(article.get("title", "")).strip()
    content = str(article.get("content", "")).strip()
    if title and content:
        if title == content:
            return title
        return f"{title}\n\n{content}"
    return title or content


def tokenize_bow(text: str) -> Counter[str]:
    return Counter(TOKEN_RE.findall(text.lower()))


def bow_norm(counter: Counter[str]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in counter.values()))


def bow_cosine(a: Counter[str], a_norm: float, b: Counter[str], b_norm: float) -> float:
    if a_norm <= 0.0 or b_norm <= 0.0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for token, cnt in a.items():
        other = b.get(token)
        if other:
            dot += float(cnt) * float(other)
    return dot / (a_norm * b_norm) if dot > 0.0 else 0.0


def parse_article_timestamp(article: dict) -> pd.Timestamp | None:
    ts = pd.to_datetime(article.get("date"), utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def business_days_between(earlier: pd.Timestamp, later: pd.Timestamp) -> int:
    earlier_day = earlier.date().isoformat()
    later_day = later.date().isoformat()
    return int(np.busday_count(earlier_day, later_day))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class JsonlShardWriter:
    def __init__(self, output_dir: Path, max_records_per_file: int):
        self.output_dir = output_dir
        self.max_records_per_file = max_records_per_file
        self.shard_index = 0
        self.shard_count = 0
        self.total_written = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.handle = None

    def _ensure_open(self) -> None:
        if self.handle is not None:
            return
        path = self.output_dir / f"news_{self.shard_index:05d}.jsonl"
        self.handle = path.open("w", encoding="utf-8")

    def write(self, article: dict) -> None:
        if self.shard_count >= self.max_records_per_file:
            self.close_current()
            self.shard_index += 1
            self.shard_count = 0
        self._ensure_open()
        assert self.handle is not None
        self.handle.write(json.dumps(article, ensure_ascii=False) + "\n")
        self.shard_count += 1
        self.total_written += 1

    def close_current(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def close(self) -> None:
        self.close_current()

    @property
    def finalized_shards(self) -> int:
        if self.total_written == 0:
            return 0
        return self.shard_index + (1 if self.shard_count > 0 or self.handle is None else 0)


def partition_articles_to_buckets(
    jsonl_files: list[Path],
    temp_dir: Path,
    bucket_count: int,
    min_text_chars: int,
    max_text_chars: int,
) -> dict[str, int]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    bucket_paths = [temp_dir / f"bucket_{i:04d}.jsonl" for i in range(bucket_count)]
    handles = [path.open("w", encoding="utf-8") for path in bucket_paths]

    stats = {
        "raw_articles": 0,
        "valid_articles": 0,
        "dropped_short_text": 0,
        "dropped_long_text": 0,
        "dropped_missing_symbol": 0,
        "dropped_invalid_date": 0,
    }

    progress = make_progress_bar(total=len(jsonl_files), desc="Partitioning raw news", unit="file")
    try:
        for jsonl_path in jsonl_files:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for source_row, line in enumerate(handle):
                    stats["raw_articles"] += 1
                    article = json.loads(line)
                    symbol = str(article.get("symbol", "")).strip().upper()
                    if not symbol:
                        stats["dropped_missing_symbol"] += 1
                        continue
                    ts = parse_article_timestamp(article)
                    if ts is None:
                        stats["dropped_invalid_date"] += 1
                        continue
                    text = build_input_text(article)
                    if len(text) < int(min_text_chars):
                        stats["dropped_short_text"] += 1
                        continue
                    if len(text) > int(max_text_chars):
                        stats["dropped_long_text"] += 1
                        continue

                    bucket_idx = zlib.crc32(symbol.encode("utf-8")) % bucket_count
                    payload = {
                        "article": article,
                        "_symbol": symbol,
                        "_date": ts.isoformat(),
                        "_source_file": str(jsonl_path),
                        "_source_row": int(source_row),
                    }
                    handles[bucket_idx].write(json.dumps(payload, ensure_ascii=False) + "\n")
                    stats["valid_articles"] += 1
            progress.update(1)
    finally:
        progress.close()
        for handle in handles:
            handle.close()

    return stats


def process_bucket_files(
    temp_dir: Path,
    writer: JsonlShardWriter,
    cosine_threshold: float,
    window_business_days: int,
) -> dict[str, int]:
    stats = {
        "kept_articles": 0,
        "dropped_duplicate_articles": 0,
    }

    bucket_files = sorted(temp_dir.glob("bucket_*.jsonl"))
    progress = make_progress_bar(total=len(bucket_files), desc="Deduplicating buckets", unit="bucket")
    try:
        for bucket_path in bucket_files:
            by_symbol: dict[str, list[dict]] = defaultdict(list)
            with bucket_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    by_symbol[str(payload["_symbol"])].append(payload)

            for symbol_records in by_symbol.values():
                enriched = []
                for payload in symbol_records:
                    article = payload["article"]
                    ts = pd.Timestamp(payload["_date"])
                    text = build_input_text(article)
                    bow = tokenize_bow(text)
                    norm = bow_norm(bow)
                    enriched.append(
                        {
                            "article": article,
                            "ts": ts,
                            "bow": bow,
                            "norm": norm,
                            "source_file": str(payload["_source_file"]),
                            "source_row": int(payload["_source_row"]),
                        }
                    )

                enriched.sort(key=lambda row: (row["ts"], row["source_file"], row["source_row"]))
                kept_window: deque[dict] = deque()
                for row in enriched:
                    current_ts = row["ts"]
                    while kept_window and business_days_between(kept_window[0]["ts"], current_ts) > int(window_business_days):
                        kept_window.popleft()

                    is_dup = False
                    for prev in kept_window:
                        sim = bow_cosine(row["bow"], row["norm"], prev["bow"], prev["norm"])
                        if sim >= float(cosine_threshold):
                            is_dup = True
                            break

                    if is_dup:
                        stats["dropped_duplicate_articles"] += 1
                        continue

                    writer.write(row["article"])
                    kept_window.append(row)
                    stats["kept_articles"] += 1
            progress.update(1)
    finally:
        progress.close()
    return stats


def write_summary_and_status(output_dir: Path, input_dirs: list[Path], summary: dict, finalized_shards: int) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    producer_status = {
        "status": "completed",
        "updated_at": utc_now_iso(),
        "tickers_total": None,
        "tickers_completed": None,
        "pages_fetched": None,
        "raw_articles": int(summary["raw_articles"]),
        "kept_articles": int(summary["kept_articles"]),
        "finalized_shards": int(finalized_shards),
        "open_partial_shard": None,
        "input_dirs": [str(path) for path in input_dirs],
    }
    (output_dir / "producer_status.json").write_text(
        json.dumps(producer_status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_dirs = resolve_input_dirs(args.input_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = discover_jsonl_files(input_dirs)
    temp_dir = output_dir / "_tmp_buckets"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    partition_stats = partition_articles_to_buckets(
        jsonl_files=jsonl_files,
        temp_dir=temp_dir,
        bucket_count=int(args.bucket_count),
        min_text_chars=int(args.min_text_chars),
        max_text_chars=int(args.max_text_chars),
    )

    writer = JsonlShardWriter(output_dir=output_dir, max_records_per_file=int(args.max_records_per_file))
    try:
        dedup_stats = process_bucket_files(
            temp_dir=temp_dir,
            writer=writer,
            cosine_threshold=float(args.cosine_threshold),
            window_business_days=int(args.window_business_days),
        )
    finally:
        writer.close()
        if not args.keep_temp_buckets and temp_dir.exists():
            shutil.rmtree(temp_dir)

    summary = {
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "raw_files": len(jsonl_files),
        "raw_articles": int(partition_stats["raw_articles"]),
        "valid_articles_after_basic_filters": int(partition_stats["valid_articles"]),
        "kept_articles": int(dedup_stats["kept_articles"]),
        "dropped_short_text_articles": int(partition_stats["dropped_short_text"]),
        "dropped_long_text_articles": int(partition_stats["dropped_long_text"]),
        "dropped_missing_symbol_articles": int(partition_stats["dropped_missing_symbol"]),
        "dropped_invalid_date_articles": int(partition_stats["dropped_invalid_date"]),
        "dropped_duplicate_articles": int(dedup_stats["dropped_duplicate_articles"]),
        "min_text_chars": int(args.min_text_chars),
        "max_text_chars": int(args.max_text_chars),
        "cosine_threshold": float(args.cosine_threshold),
        "window_business_days": int(args.window_business_days),
        "bucket_count": int(args.bucket_count),
        "max_records_per_file": int(args.max_records_per_file),
        "finalized_shards": int(writer.finalized_shards),
    }
    write_summary_and_status(output_dir, input_dirs, summary, writer.finalized_shards)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
