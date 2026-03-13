from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_HF_TOKEN_FILE = Path("/home/jyu197/LLMCrossSec/hf_token")
DEFAULT_START_DATE = "2020-01-01T00:00:00Z"
DEFAULT_END_DATE = "2100-01-01T00:00:00Z"
DEFAULT_MODEL_NAME_OR_PATH = "meta-llama/Llama-2-13b-hf"
DEFAULT_POLL_SECONDS = 60.0
DEFAULT_PRODUCER_TIMEOUT_SECONDS = 1200.0


def resolve_input_dirs(explicit_paths: list[str] | None) -> list[Path]:
    if explicit_paths:
        resolved = [Path(path).expanduser().resolve() for path in explicit_paths]
        if resolved:
            return resolved
    raise FileNotFoundError("No input directories found. Pass one or more --input-dir /path/to/downloaded/eodhd/news.")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for deduplicated EODHD news JSONL shards."
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        default=None,
        help="Directory containing deduplicated finalized news_*.jsonl shards and producer_status.json. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for embedding chunks and metadata parquet files.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help="Embedding model name or local path. Default: meta-llama/Llama-2-13b-hf",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Keep rows with date >= start-date (inclusive). Default: 2018-01-01T00:00:00Z",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help="Keep rows with date <= end-date (inclusive). Default: 2100-01-01T00:00:00Z",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max total tokens passed to the model after truncation. Default: 2048",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="How many documents to process per outer loop batch.",
    )
    parser.add_argument(
        "--rows-per-chunk",
        type=int,
        default=2048,
        help="How many kept rows to store per output chunk.",
    )
    parser.add_argument(
        "--save-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Embedding dtype on disk.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["auto", "cpu", "cuda"],
        help="Torch device to use.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="Optional cap on kept rows for debugging. 0 means no cap.",
    )
    parser.add_argument(
        "--store-text",
        action="store_true",
        help="Also store full text in output metadata parquet.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of dataset shards to process in parallel. Default: 1",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="0-based shard index for this worker. Default: 0",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
        help=f"How often to poll for new finalized shards while producer is running. Default: {DEFAULT_POLL_SECONDS}",
    )
    parser.add_argument(
        "--producer-timeout-seconds",
        type=float,
        default=DEFAULT_PRODUCER_TIMEOUT_SECONDS,
        help=(
            "If producer_status.json says running but has not updated within this many seconds, "
            f"treat the producer as stalled. Default: {DEFAULT_PRODUCER_TIMEOUT_SECONDS}"
        ),
    )
    parser.add_argument(
        "--producer-status-file",
        action="append",
        default=None,
        help=(
            "Optional explicit path(s) to producer_status.json. If provided, the count must match "
            "the number of --input-dir arguments. Default: <input-dir>/producer_status.json"
        ),
    )
    parser.add_argument(
        "--embed-state-file",
        default=None,
        help="Optional explicit path for embed state JSON. Default: <output-dir>/embed_state.json",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_visible_cuda_devices() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        devices = [item.strip() for item in raw.split(",") if item.strip()]
        return devices
    return [str(i) for i in range(torch.cuda.device_count())]


def get_idle_visible_cuda_devices() -> list[str]:
    visible_devices = get_visible_cuda_devices()
    if len(visible_devices) <= 1:
        return visible_devices

    try:
        gpu_query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        process_query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return visible_devices

    gpu_records = []
    for line in gpu_query.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        index, uuid, memory_used = parts
        try:
            memory_used_mib = int(memory_used)
        except ValueError:
            continue
        gpu_records.append(
            {
                "index": index,
                "uuid": uuid,
                "memory_used_mib": memory_used_mib,
            }
        )

    busy_uuids = {line.strip() for line in process_query.splitlines() if line.strip()}

    idle_visible_devices: list[str] = []
    for visible_device in visible_devices:
        matched_record = next(
            (
                record
                for record in gpu_records
                if visible_device == record["index"]
                or visible_device == record["uuid"]
                or record["uuid"].startswith(visible_device)
            ),
            None,
        )
        if matched_record is None:
            idle_visible_devices.append(visible_device)
            continue

        has_compute_process = matched_record["uuid"] in busy_uuids
        has_material_memory_usage = matched_record["memory_used_mib"] > 64
        if not has_compute_process and not has_material_memory_usage:
            idle_visible_devices.append(visible_device)

    return idle_visible_devices


def maybe_launch_multi_gpu_workers(args: argparse.Namespace, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    if args.num_shards != 1 or args.shard_id != 0:
        return False

    visible_devices = get_visible_cuda_devices()
    idle_visible_devices = get_idle_visible_cuda_devices()
    if not idle_visible_devices:
        raise RuntimeError("No idle CUDA devices are currently available.")

    base_output_dir = Path(args.output_dir).expanduser().resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)
    worker_count = len(idle_visible_devices)
    worker_procs: list[subprocess.Popen] = []

    print(
        f"detected {len(visible_devices)} visible CUDA devices, "
        f"launching {worker_count} shard worker(s) on idle devices under {base_output_dir}"
    )

    try:
        for shard_id, visible_device in enumerate(idle_visible_devices):
            worker_output_dir = base_output_dir / f"shard_{shard_id:02d}"
            worker_cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--num-shards",
                str(worker_count),
                "--shard-id",
                str(shard_id),
                "--output-dir",
                str(worker_output_dir),
                "--device",
                "cuda",
            ]
            worker_env = os.environ.copy()
            worker_env["CUDA_VISIBLE_DEVICES"] = visible_device
            print(f"launching shard {shard_id}/{worker_count - 1} on CUDA_VISIBLE_DEVICES={visible_device}")
            worker_procs.append(subprocess.Popen(worker_cmd, env=worker_env))

        exit_codes = [proc.wait() for proc in worker_procs]
    except KeyboardInterrupt:
        for proc in worker_procs:
            if proc.poll() is None:
                proc.terminate()
        raise

    failed = [idx for idx, code in enumerate(exit_codes) if code != 0]
    if failed:
        raise RuntimeError(f"Shard workers failed for shard ids: {failed}")
    return True


def load_model_and_tokenizer(model_name_or_path: str, device: torch.device):
    token = resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    device_map = "auto" if device.type == "cuda" else None
    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "token": token,
        "device_map": device_map,
    }
    if device.type == "cuda":
        model_kwargs["attn_implementation"] = "sdpa"
    try:
        model = AutoModel.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
    except TypeError as exc:
        if model_kwargs.get("attn_implementation") != "sdpa":
            raise
        fallback_kwargs = dict(model_kwargs)
        fallback_kwargs.pop("attn_implementation", None)
        print(
            "[warn] transformers version/model class does not accept attn_implementation='sdpa'; "
            "falling back to default attention implementation."
        )
        model = AutoModel.from_pretrained(
            model_name_or_path,
            **fallback_kwargs,
        )
    model.eval()
    if device_map is None:
        model.to(device)
    return tokenizer, model


def get_embedding_numpy_dtype(device: torch.device) -> np.dtype:
    return np.float16 if device.type == "cuda" else np.float32


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype).expand(last_hidden_state.shape)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-6)
    return summed / counts


def sanitize_embedding_batch(embeddings: torch.Tensor) -> torch.Tensor:
    # Rare fp16 attention pathologies can emit non-finite values on a few rows.
    return torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)


class NullProgressBar:
    def update(self, _: int) -> None:
        pass

    def set_description(self, _: str) -> None:
        pass

    def set_postfix(self, **_: object) -> None:
        pass

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        pass


def make_progress_bar(total: int | None, desc: str):
    if tqdm is None:
        return NullProgressBar()
    return tqdm(total=total, desc=desc, unit="rows", leave=False)


def make_jsonl_progress_bar(total: int, initial: int):
    if tqdm is None:
        return NullProgressBar()
    return tqdm(total=total, initial=initial, desc="JSONL shards", unit="file")


def make_source_name(input_dir: Path, index: int) -> str:
    return f"source_{index:02d}_{input_dir.name}"


def build_input_sources(
    input_dirs: list[Path],
    producer_status_files: list[str] | None,
) -> list[dict[str, Path | str]]:
    if producer_status_files and len(producer_status_files) != len(input_dirs):
        raise ValueError("--producer-status-file count must match --input-dir count when provided.")

    sources: list[dict[str, Path | str]] = []
    for index, input_dir in enumerate(input_dirs):
        status_path = (
            Path(producer_status_files[index]).expanduser().resolve()
            if producer_status_files
            else input_dir / "producer_status.json"
        )
        sources.append(
            {
                "name": make_source_name(input_dir, index),
                "input_dir": input_dir,
                "producer_status_file": status_path,
            }
        )
    return sources


def collect_all_tasks(sources: list[dict[str, Path | str]]) -> list[dict[str, Path | str]]:
    tasks: list[dict[str, Path | str]] = []
    for source in sources:
        source_name = str(source["name"])
        input_dir = Path(source["input_dir"])
        for jsonl_path in sorted(input_dir.glob("news_*.jsonl")):
            tasks.append(
                {
                    "task_id": f"{source_name}/{jsonl_path.name}",
                    "source_name": source_name,
                    "jsonl_path": jsonl_path,
                    "producer_status_file": Path(source["producer_status_file"]),
                }
            )
    tasks.sort(key=lambda item: str(item["task_id"]))
    return tasks


def iter_assigned_tasks(
    sources: list[dict[str, Path | str]],
    *,
    num_shards: int,
    shard_id: int,
) -> Iterable[dict[str, Path | str]]:
    all_tasks = collect_all_tasks(sources)
    for file_idx, task in enumerate(all_tasks):
        if file_idx % num_shards == shard_id:
            yield task


def count_assigned_finalized_files(
    sources: list[dict[str, Path | str]],
    num_shards: int,
    shard_id: int,
) -> int:
    return sum(1 for _ in iter_assigned_tasks(sources, num_shards=num_shards, shard_id=shard_id))


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def clear_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def encode_texts_batched(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    *,
    max_length: int,
) -> np.ndarray:
    emb_dtype = get_embedding_numpy_dtype(device)
    if not texts:
        return np.empty((0, 0), dtype=emb_dtype)

    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    outputs = model(**encoded)
    pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
    pooled = sanitize_embedding_batch(pooled)
    return pooled.detach().cpu().to(dtype=outputs.last_hidden_state.dtype).numpy().astype(emb_dtype, copy=False)


def iter_finalized_jsonl_files(
    input_dir: Path,
    *,
    num_shards: int,
    shard_id: int,
) -> Iterable[Path]:
    file_idx = 0
    for jsonl_path in sorted(input_dir.glob("news_*.jsonl")):
        if file_idx % num_shards == shard_id:
            yield jsonl_path
        file_idx += 1


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def default_embed_state(
    input_dirs: list[Path],
    output_dir: Path,
    producer_status_files: list[Path],
) -> dict:
    return {
        "version": 2,
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "producer_status_files": [str(path) for path in producer_status_files],
        "processed_tasks": [],
        "current_task": None,
        "next_chunk_idx": 0,
        "stats": {
            "tasks_processed": 0,
            "scanned_total": 0,
            "kept_total": 0,
            "chunks_written": 0,
        },
    }


def migrate_legacy_embed_state(
    payload: dict,
    input_dirs: list[Path],
    output_dir: Path,
    producer_status_files: list[Path],
) -> dict:
    if "processed_tasks" in payload:
        return payload

    legacy_processed_files = payload.get("processed_files")
    legacy_current_file = payload.get("current_file")
    if not isinstance(legacy_processed_files, list):
        return payload

    current_sources = build_input_sources(input_dirs, [str(path) for path in producer_status_files])
    source_name_by_input_dir = {
        str(Path(source["input_dir"]).resolve()): str(source["name"])
        for source in current_sources
    }

    legacy_input_dir_raw = payload.get("input_dir")
    legacy_source_name: str | None = None
    if isinstance(legacy_input_dir_raw, str) and legacy_input_dir_raw.strip():
        legacy_input_dir = str(Path(legacy_input_dir_raw).expanduser().resolve())
        legacy_source_name = source_name_by_input_dir.get(legacy_input_dir)

    if legacy_source_name is None:
        if len(current_sources) == 1:
            legacy_source_name = str(current_sources[0]["name"])
        else:
            raise ValueError(
                "Legacy embed_state.json detected, but could not map its old input_dir to one of the "
                "current --input-dir values. Put the old input directory first or use a fresh output-dir."
            )

    migrated = default_embed_state(input_dirs, output_dir, producer_status_files)
    migrated["processed_tasks"] = [
        f"{legacy_source_name}/{file_name}"
        for file_name in legacy_processed_files
        if isinstance(file_name, str) and file_name.strip()
    ]
    migrated["current_task"] = (
        f"{legacy_source_name}/{legacy_current_file}"
        if isinstance(legacy_current_file, str) and legacy_current_file.strip()
        else None
    )
    migrated["next_chunk_idx"] = int(payload.get("next_chunk_idx", 0))
    migrated["stats"] = {
        "tasks_processed": len(migrated["processed_tasks"]),
        "scanned_total": int(payload.get("stats", {}).get("scanned_total", 0)),
        "kept_total": int(payload.get("stats", {}).get("kept_total", 0)),
        "chunks_written": int(payload.get("stats", {}).get("chunks_written", 0)),
    }
    migrated["migrated_from_legacy"] = True
    migrated["legacy_embed_state_version"] = payload.get("version")
    return migrated


def load_embed_state(
    path: Path,
    input_dirs: list[Path],
    output_dir: Path,
    producer_status_files: list[Path],
) -> dict:
    payload = read_json(path)
    if payload is not None:
        return migrate_legacy_embed_state(payload, input_dirs, output_dir, producer_status_files)
    return default_embed_state(input_dirs, output_dir, producer_status_files)


def save_embed_summary(path: Path, state: dict, producer_statuses: dict[str, dict | None]) -> None:
    summary = {
        "version": state.get("version"),
        "input_dirs": state.get("input_dirs"),
        "output_dir": state.get("output_dir"),
        "current_task": state.get("current_task"),
        "processed_tasks": len(state.get("processed_tasks", [])),
        "next_chunk_idx": state.get("next_chunk_idx"),
        "stats": state.get("stats"),
        "producer_statuses": producer_statuses,
    }
    save_json(path, summary)


def parse_article_date(raw_value: object) -> pd.Timestamp | None:
    if raw_value is None:
        return None
    ts = pd.to_datetime(raw_value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def normalize_symbols(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def strip_urls(text: str) -> str:
    text_without_urls = re.sub(r"https?://\S+|www\.\S+", "", text)
    return re.sub(r"[ \t]+\n", "\n", text_without_urls).strip()


def build_raw_article_text(article: dict) -> str:
    title = str(article.get("title", "")).strip()
    content = strip_urls(str(article.get("content", "")).strip())
    if title and content:
        if title == content:
            return title
        return f"{title}\n\n{content}"
    return title or content


def build_input_text(article: dict) -> str:
    return build_raw_article_text(article)


def build_extra_fields(article: dict) -> str:
    payload = {
        "ticker": str(article.get("symbol", "")).strip().upper(),
        "source_ticker": str(article.get("source_ticker", "")).strip().upper(),
        "link": str(article.get("link", "")).strip(),
        "title": str(article.get("title", "")).strip(),
        "tags": article.get("tags", []),
        "sentiment": article.get("sentiment", {}),
        "raw_symbols": normalize_symbols(article.get("raw_symbols")),
        "dataset": "eodhd_single_symbol_news",
        "source": "EODHD",
        "date_trading": str(article.get("date", "")).strip(),
    }
    return json.dumps(payload, ensure_ascii=False)


def build_metadata_row(
    article: dict,
    source_file: Path,
    source_row: int,
    text: str,
    *,
    store_text: bool,
) -> dict:
    out = {
        "date": str(article.get("date", "")),
        "subset": "eodhd_single_symbol_news",
        "symbol": str(article.get("symbol", "")).strip().upper(),
        "source_ticker": str(article.get("source_ticker", "")).strip().upper(),
        "title": str(article.get("title", "")),
        "link": str(article.get("link", "")),
        "source_file": str(source_file),
        "source_row": int(source_row),
        "text_char_len": len(text),
        "extra_fields": build_extra_fields(article),
    }
    if store_text:
        out["text"] = text
    return out


def load_articles_from_jsonl(
    jsonl_path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    limit_rows: int,
) -> tuple[list[str], list[dict], int]:
    texts: list[str] = []
    metas: list[dict] = []
    scanned_rows = 0

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for source_row, line in enumerate(handle):
            scanned_rows += 1
            article = json.loads(line)
            article_date = parse_article_date(article.get("date"))
            if article_date is None or article_date < start_date or article_date > end_date:
                continue

            text = build_input_text(article)
            metas.append(
                build_metadata_row(
                    article=article,
                    source_file=jsonl_path,
                    source_row=source_row,
                    text=text,
                    store_text=False,
                )
            )
            texts.append(text)
            if limit_rows and len(texts) >= limit_rows:
                break

    return texts, metas, scanned_rows


def refresh_metadata_store_text(metadata_rows: list[dict], texts: list[str], store_text: bool) -> list[dict]:
    if not store_text:
        return metadata_rows
    refreshed: list[dict] = []
    for meta, text in zip(metadata_rows, texts, strict=True):
        row = dict(meta)
        row["text"] = text
        refreshed.append(row)
    return refreshed


def write_embedding_chunks(
    output_dir: Path,
    metadata_rows: list[dict],
    embeddings: np.ndarray,
    start_chunk_idx: int,
    rows_per_chunk: int,
    save_dtype: str,
) -> int:
    chunk_idx = start_chunk_idx
    start = 0
    while start < len(metadata_rows):
        end = min(start + rows_per_chunk, len(metadata_rows))
        save_chunk(
            output_dir=output_dir,
            chunk_idx=chunk_idx,
            metadata_rows=metadata_rows[start:end],
            embeddings=embeddings[start:end],
            save_dtype=save_dtype,
        )
        chunk_idx += 1
        start = end
    return chunk_idx


def producer_status_is_stale(producer_status: dict, timeout_seconds: float) -> bool:
    updated_at = producer_status.get("updated_at")
    if not updated_at:
        return True
    updated_ts = pd.to_datetime(updated_at, utc=True, errors="coerce")
    if pd.isna(updated_ts):
        return True
    age_seconds = (pd.Timestamp.now(tz="UTC") - updated_ts).total_seconds()
    return age_seconds > timeout_seconds


def read_producer_statuses(sources: list[dict[str, Path | str]]) -> dict[str, dict | None]:
    return {
        str(source["name"]): read_json(Path(source["producer_status_file"]))
        for source in sources
    }


def process_jsonl_file(
    jsonl_path: Path,
    *,
    tokenizer,
    model,
    device: torch.device,
    args: argparse.Namespace,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    start_chunk_idx: int,
    global_kept_total: int,
    global_scanned_total: int,
) -> tuple[int, int, int]:
    remaining_limit = max(0, args.limit_rows - global_kept_total) if args.limit_rows else 0
    texts, metadata_rows, scanned_rows = load_articles_from_jsonl(
        jsonl_path=jsonl_path,
        start_date=start_date,
        end_date=end_date,
        limit_rows=remaining_limit,
    )
    metadata_rows = refresh_metadata_store_text(metadata_rows, texts, args.store_text)
    file_progress = make_progress_bar(total=len(texts), desc=jsonl_path.name)

    current_batch_size = args.batch_size
    embedded_batches: list[np.ndarray] = []
    kept_rows = 0

    try:
        start = 0
        while start < len(texts):
            text_batch = texts[start:start + current_batch_size]
            try:
                emb_batch = encode_texts_batched(
                    texts=text_batch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=args.max_length,
                )
            except Exception as exc:
                if not is_oom_error(exc):
                    raise

                old_batch_size = current_batch_size
                new_batch_size = max(1, current_batch_size // 2)
                if old_batch_size == new_batch_size:
                    raise RuntimeError(
                        "CUDA out of memory even with batch_size=1."
                    ) from exc

                print(
                    "OOM on embedding batch; retrying with smaller batch sizes: "
                    f"batch_size {old_batch_size} -> {new_batch_size}"
                )
                current_batch_size = new_batch_size
                clear_memory(device)
                file_progress.set_postfix(batch=current_batch_size)
                continue

            embedded_batches.append(emb_batch)
            kept_rows += len(text_batch)
            start += len(text_batch)
            file_progress.update(len(text_batch))
            file_progress.set_postfix(batch=current_batch_size)
    finally:
        file_progress.close()

    if not metadata_rows:
        return scanned_rows, 0, start_chunk_idx

    all_embeddings = np.concatenate(embedded_batches, axis=0)
    next_chunk_idx = write_embedding_chunks(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        metadata_rows=metadata_rows,
        embeddings=all_embeddings,
        start_chunk_idx=start_chunk_idx,
        rows_per_chunk=args.rows_per_chunk,
        save_dtype=args.save_dtype,
    )
    return scanned_rows, len(metadata_rows), next_chunk_idx


def save_chunk(
    output_dir: Path,
    chunk_idx: int,
    metadata_rows: list[dict],
    embeddings: np.ndarray,
    save_dtype: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_dtype = np.float16 if save_dtype == "float16" else np.float32
    emb_path = output_dir / f"embeddings_{chunk_idx:05d}.npy"
    meta_path = output_dir / f"metadata_{chunk_idx:05d}.parquet"

    np.save(emb_path, embeddings.astype(emb_dtype, copy=False))
    pd.DataFrame(metadata_rows).to_parquet(meta_path, index=False)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")

    device = resolve_device(args.device)
    if maybe_launch_multi_gpu_workers(args, device):
        return

    input_dirs = resolve_input_dirs(args.input_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    start_date = pd.Timestamp(args.start_date, tz="UTC")
    end_date = pd.Timestamp(args.end_date, tz="UTC")
    if end_date < start_date:
        raise ValueError("--end-date must be >= --start-date")
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Could not find input directory: {input_dir}")

    sources = build_input_sources(input_dirs, args.producer_status_file)
    producer_status_files = [Path(source["producer_status_file"]) for source in sources]
    embed_state_file = (
        Path(args.embed_state_file).expanduser().resolve()
        if args.embed_state_file
        else output_dir / "embed_state.json"
    )

    tokenizer, model = load_model_and_tokenizer(args.model_name_or_path, device)

    manifest = {
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "producer_status_files": [str(path) for path in producer_status_files],
        "model_name_or_path": args.model_name_or_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "rows_per_chunk": args.rows_per_chunk,
        "save_dtype": args.save_dtype,
        "device": str(device),
        "store_text": bool(args.store_text),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "poll_seconds": args.poll_seconds,
        "producer_timeout_seconds": args.producer_timeout_seconds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    state = load_embed_state(embed_state_file, input_dirs, output_dir, producer_status_files)
    processed_tasks = set(state.get("processed_tasks", []))
    kept_total = int(state.get("stats", {}).get("kept_total", 0))
    scanned_total = int(state.get("stats", {}).get("scanned_total", 0))
    chunk_idx = int(state.get("next_chunk_idx", 0))
    summary_path = output_dir / "summary.json"
    initial_total = count_assigned_finalized_files(sources, args.num_shards, args.shard_id)
    jsonl_progress = make_jsonl_progress_bar(total=max(initial_total, len(processed_tasks)), initial=len(processed_tasks))
    jsonl_progress.set_description(f"JSONL shard {args.shard_id + 1}/{args.num_shards}")
    try:
        while True:
            producer_statuses = read_producer_statuses(sources)
            assigned_tasks = list(
                iter_assigned_tasks(
                    sources,
                    num_shards=args.num_shards,
                    shard_id=args.shard_id,
                )
            )
            pending_tasks = [task for task in assigned_tasks if str(task["task_id"]) not in processed_tasks]
            produced_assigned_count = len(assigned_tasks)
            produced_global_count = sum(
                int((producer_statuses.get(str(source["name"])) or {}).get("finalized_shards", 0))
                for source in sources
            )

            if tqdm is not None:
                jsonl_progress.total = max(produced_assigned_count, len(processed_tasks))
                jsonl_progress.n = len(processed_tasks)
                jsonl_progress.refresh()
            jsonl_progress.set_postfix(
                processed=len(processed_tasks),
                produced=produced_assigned_count,
                producer_total=produced_global_count,
                current=state.get("current_task") or "-",
            )

            if pending_tasks:
                for task in pending_tasks:
                    if args.limit_rows and kept_total >= args.limit_rows:
                        break

                    task_id = str(task["task_id"])
                    jsonl_path = Path(task["jsonl_path"])
                    source_name = str(task["source_name"])
                    state["current_task"] = task_id
                    save_json(embed_state_file, state)
                    save_embed_summary(summary_path, state, producer_statuses)
                    jsonl_progress.set_postfix(
                        processed=len(processed_tasks),
                        produced=produced_assigned_count,
                        producer_total=produced_global_count,
                        current=task_id,
                    )

                    scanned_rows, kept_rows, next_chunk_idx = process_jsonl_file(
                        jsonl_path,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        args=args,
                        start_date=start_date,
                        end_date=end_date,
                        start_chunk_idx=chunk_idx,
                        global_kept_total=kept_total,
                        global_scanned_total=scanned_total,
                    )
                    scanned_total += scanned_rows
                    kept_total += kept_rows
                    processed_tasks.add(task_id)
                    state["processed_tasks"] = sorted(processed_tasks)
                    state["current_task"] = None
                    state["next_chunk_idx"] = next_chunk_idx
                    state["stats"]["tasks_processed"] = len(processed_tasks)
                    state["stats"]["scanned_total"] = scanned_total
                    state["stats"]["kept_total"] = kept_total
                    state["stats"]["chunks_written"] = next_chunk_idx
                    chunk_idx = next_chunk_idx
                    save_json(embed_state_file, state)
                    save_embed_summary(summary_path, state, producer_statuses)
                    jsonl_progress.update(1)
                    jsonl_progress.set_postfix(
                        processed=len(processed_tasks),
                        produced=produced_assigned_count,
                        producer_total=produced_global_count,
                        current="-",
                    )

                if args.limit_rows and kept_total >= args.limit_rows:
                    break
                continue

            if not producer_statuses:
                break

            save_embed_summary(summary_path, state, producer_statuses)
            producer_states = {
                source_name: str((status or {}).get("status", "")).strip().lower()
                for source_name, status in producer_statuses.items()
            }

            if all(state_value == "completed" for state_value in producer_states.values()):
                break
            if any(state_value == "failed" for state_value in producer_states.values()):
                failed_sources = [
                    source_name
                    for source_name, state_value in producer_states.items()
                    if state_value == "failed"
                ]
                details = {
                    source_name: (producer_statuses.get(source_name) or {}).get("error", "unknown error")
                    for source_name in failed_sources
                }
                raise RuntimeError(f"producer failed: {details}")
            if all(state_value in {"completed", "stopped"} for state_value in producer_states.values()):
                break

            stale_running_sources = [
                source_name
                for source_name, state_value in producer_states.items()
                if state_value == "running"
                and producer_status_is_stale(producer_statuses[source_name] or {}, args.producer_timeout_seconds)
            ]
            if stale_running_sources:
                raise RuntimeError(
                    "producer_status.json has not updated within the configured timeout for sources: "
                    f"{stale_running_sources}"
                )

            time.sleep(args.poll_seconds)
    finally:
        jsonl_progress.close()

    save_embed_summary(summary_path, state, read_producer_statuses(sources))
    print(json.dumps(read_json(summary_path) or {}, indent=2))


if __name__ == "__main__":
    main()
