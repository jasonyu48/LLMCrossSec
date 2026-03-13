from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


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
        description="Count deduplicated news rows on or after a cutoff timestamp with a progress bar."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing deduplicated news_*.jsonl shards.",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Inclusive cutoff timestamp, e.g. 2023-03-15 or 2023-03-15T00:00:00Z.",
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
    return parser.parse_args()


def parse_timestamp(raw: str) -> datetime:
    text = str(raw).strip()
    if len(text) == 10:
        text = f"{text}T00:00:00+00:00"
    elif text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def discover_news_files(input_dir: Path, max_files: int) -> list[Path]:
    files = sorted(input_dir.glob("news_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No news_*.jsonl files found under {input_dir}")
    if max_files > 0:
        files = files[:max_files]
    return files


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    cutoff = parse_timestamp(args.start_date)
    files = discover_news_files(input_dir, max_files=int(args.max_files))

    matched_rows = 0
    scanned_rows = 0
    bad_json_rows = 0
    bad_date_rows = 0
    files_scanned = 0

    progress = make_progress_bar(total=len(files), desc="Count news", unit="file")
    try:
        for path in files:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    scanned_rows += 1
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        bad_json_rows += 1
                        continue

                    raw_date = payload.get("date")
                    if not raw_date:
                        bad_date_rows += 1
                        continue

                    try:
                        news_dt = parse_timestamp(str(raw_date))
                    except ValueError:
                        bad_date_rows += 1
                        continue

                    if news_dt >= cutoff:
                        matched_rows += 1

                    if args.max_rows > 0 and scanned_rows >= int(args.max_rows):
                        break

            progress.update(1)
            files_scanned += 1
            progress.set_postfix(
                matched=matched_rows,
                scanned=scanned_rows,
                bad_json=bad_json_rows,
                bad_date=bad_date_rows,
            )
            if args.max_rows > 0 and scanned_rows >= int(args.max_rows):
                break
    finally:
        progress.close()

    print(f"input_dir: {input_dir}")
    print(f"start_date_inclusive_utc: {cutoff.isoformat()}")
    print(f"files_scanned: {files_scanned}")
    print(f"rows_scanned: {scanned_rows}")
    print(f"rows_matched: {matched_rows}")
    print(f"bad_json_rows: {bad_json_rows}")
    print(f"bad_date_rows: {bad_date_rows}")


if __name__ == "__main__":
    main()
