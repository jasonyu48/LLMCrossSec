from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


DEFAULT_HF_TOKEN_FILE = Path("/home/jyu197/LLMCrossSec/hf_token")
DEFAULT_MODEL_NAME_OR_PATH = "meta-llama/Llama-2-13b-hf"
DEFAULT_THRESHOLDS = [128, 256, 512, 1024, 2048, 4096, 8192, 10000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count token-length distribution for deduplicated news articles using the Llama 2 tokenizer. "
            "If dedup metadata does not store text, reconstruct text from source_file/source_row."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Deduplicated embedding directory with metadata_*.parquet files.")
    parser.add_argument("--output-dir", required=True, help="Directory to save token count summaries.")
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=DEFAULT_THRESHOLDS,
        help="Token-count thresholds to report exceedance rates for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="How many texts to tokenize at once.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on counted articles for debugging. 0 means no cap.",
    )
    parser.add_argument(
        "--hist-bins",
        nargs="+",
        type=int,
        default=[0, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        help="Upper bin edges for histogram CSV. Final bin is open-ended.",
    )
    return parser.parse_args()


def resolve_hf_token() -> str | None:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(env_name)
        if value:
            return value.strip()
    if DEFAULT_HF_TOKEN_FILE.is_file():
        value = DEFAULT_HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def discover_metadata_files(input_dir: Path) -> list[Path]:
    shard_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("shard_"))
    roots = shard_dirs or [input_dir]
    out: list[Path] = []
    for root in roots:
        out.extend(sorted(root.glob("metadata_*.parquet")))
    if not out:
        raise FileNotFoundError(f"No metadata_*.parquet files found under {input_dir}")
    return out


def build_input_text(article: dict) -> str:
    title = str(article.get("title", "")).strip()
    content = str(article.get("content", "")).strip()
    if title and content:
        return f"{title}\n\n{content}"
    return title or content


def load_texts_from_jsonl_rows(source_file: Path, row_ids: np.ndarray) -> list[str]:
    wanted = sorted(set(int(v) for v in row_ids.tolist()))
    wanted_set = set(wanted)
    loaded: dict[int, str] = {}
    with source_file.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if line_idx not in wanted_set:
                continue
            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                loaded[line_idx] = ""
                continue
            loaded[line_idx] = build_input_text(article)
            if len(loaded) == len(wanted_set):
                break
    return [loaded.get(int(v), "") for v in row_ids.tolist()]


def iter_text_batches(metadata_files: list[Path], batch_size: int, max_rows: int) -> tuple[list[str], int]:
    pending: list[str] = []
    yielded = 0

    for meta_path in metadata_files:
        meta = pd.read_parquet(meta_path)
        if "text" in meta.columns:
            texts = meta["text"].fillna("").astype(str)
        else:
            required = {"source_file", "source_row"}
            if not required.issubset(meta.columns):
                raise ValueError(f"{meta_path} needs either text column or source_file/source_row.")
            texts = pd.Series(index=meta.index, dtype="object")
            for source_file, group in meta.groupby("source_file", sort=False):
                path = Path(str(source_file))
                row_ids = pd.to_numeric(group["source_row"], errors="coerce").fillna(-1).astype(int).to_numpy()
                if path.suffix.lower() != ".jsonl":
                    raise ValueError(f"Unsupported source_file suffix for text reconstruction: {path}")
                loaded = load_texts_from_jsonl_rows(path, row_ids)
                texts.loc[group.index] = loaded
            texts = texts.fillna("").astype(str)

        for text in texts.tolist():
            pending.append(text)
            yielded += 1
            if max_rows > 0 and yielded >= max_rows:
                while pending:
                    yield pending[:batch_size], yielded
                    pending = pending[batch_size:]
                return
            if len(pending) >= batch_size:
                yield pending[:batch_size], yielded
                pending = pending[batch_size:]

    while pending:
        yield pending[:batch_size], yielded
        pending = pending[batch_size:]


def summarize_counts(counts: np.ndarray, thresholds: list[int]) -> dict:
    counts = counts.astype(np.int64, copy=False)
    q_levels = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    quantiles = np.quantile(counts, q_levels).tolist() if counts.size else []
    quantile_map = {f"q{int(q * 100):02d}" if q < 1 else "q100": float(v) for q, v in zip(q_levels, quantiles, strict=True)}

    threshold_rows = []
    threshold_fracs: dict[str, float] = {}
    for threshold in thresholds:
        n_ge = int((counts >= int(threshold)).sum())
        frac_ge = float(n_ge / counts.size) if counts.size else 0.0
        threshold_rows.append(
            {
                "threshold": int(threshold),
                "n_ge": n_ge,
                "frac_ge": frac_ge,
            }
        )
        threshold_fracs[f"frac_ge_{int(threshold)}"] = frac_ge

    return {
        "n_articles": int(counts.size),
        "mean": float(counts.mean()) if counts.size else 0.0,
        "std": float(counts.std(ddof=1)) if counts.size > 1 else 0.0,
        "min": int(counts.min()) if counts.size else 0,
        "max": int(counts.max()) if counts.size else 0,
        "quantiles": quantile_map,
        "thresholds": threshold_rows,
        "threshold_fracs": threshold_fracs,
    }


def write_histogram_csv(path: Path, counts: np.ndarray, hist_bins: list[int]) -> None:
    edges = sorted(set(int(v) for v in hist_bins))
    if not edges or edges[0] != 0:
        edges = [0] + edges

    rows: list[dict[str, object]] = []
    for idx in range(len(edges) - 1):
        left = edges[idx]
        right = edges[idx + 1]
        mask = (counts >= left) & (counts < right)
        n = int(mask.sum())
        rows.append(
            {
                "bin_left_inclusive": left,
                "bin_right_exclusive": right,
                "n_articles": n,
                "frac_articles": float(n / counts.size) if counts.size else 0.0,
            }
        )

    last_left = edges[-1]
    last_mask = counts >= last_left
    last_n = int(last_mask.sum())
    rows.append(
        {
            "bin_left_inclusive": last_left,
            "bin_right_exclusive": "",
            "n_articles": last_n,
            "frac_articles": float(last_n / counts.size) if counts.size else 0.0,
        }
    )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bin_left_inclusive", "bin_right_exclusive", "n_articles", "frac_articles"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_files = discover_metadata_files(input_dir)
    token = resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=token, use_fast=True)

    all_counts: list[np.ndarray] = []
    processed = 0
    for text_batch, seen_rows in iter_text_batches(
        metadata_files=metadata_files,
        batch_size=int(args.batch_size),
        max_rows=int(args.max_rows),
    ):
        encoded = tokenizer(
            text_batch,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        counts = np.asarray([len(ids) for ids in encoded["input_ids"]], dtype=np.int32)
        all_counts.append(counts)
        processed = seen_rows
        print(f"processed {processed} articles", flush=True)

    if not all_counts:
        raise RuntimeError("No articles were processed.")

    counts = np.concatenate(all_counts, axis=0).astype(np.int64, copy=False)
    summary = summarize_counts(counts, thresholds=[int(v) for v in args.thresholds])
    summary["input_dir"] = str(input_dir)
    summary["output_dir"] = str(output_dir)
    summary["model_name_or_path"] = str(args.model_name_or_path)
    summary["batch_size"] = int(args.batch_size)
    summary["max_rows"] = int(args.max_rows)
    summary["metadata_files"] = len(metadata_files)

    summary_path = output_dir / "llama2_token_count_summary.json"
    counts_path = output_dir / "llama2_token_counts.npy"
    hist_path = output_dir / "llama2_token_histogram.csv"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(counts_path, counts.astype(np.int32, copy=False))
    write_histogram_csv(hist_path, counts, hist_bins=[int(v) for v in args.hist_bins])

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved raw counts to {counts_path}")
    print(f"saved histogram to {hist_path}")


if __name__ == "__main__":
    main()
