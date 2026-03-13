from __future__ import annotations

import argparse
import gc
import json
import numpy as np
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Iterable

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_HF_TOKEN_FILE = Path("/home/jyu197/LLMCrossSec/hf_token")
DEFAULT_START_DATE = "2023-07-19T00:00:00Z"
DEFAULT_END_DATE = "2100-01-01T00:00:00Z"
DEFAULT_MODEL_NAME_OR_PATH = "meta-llama/Llama-2-13b-chat-hf"
DEFAULT_POLL_SECONDS = 60.0
DEFAULT_PRODUCER_TIMEOUT_SECONDS = 1200.0
DEFAULT_SYSTEM_PROMPT = "You are a financial expert."
DEFAULT_USER_INSTRUCTION = (
    'Answer "YES" if good news, "NO" if bad news, or "UNKNOWN" if uncertain in the first line. '
    "Then elaborate with one short and concise sentence on the next line."
)
DEFAULT_POST_NEWS_INSTRUCTION = "The above is the {subject}. Please choose a trading position."
DEFAULT_PROMPT_V2_TEMPLATE = (
    "Choose a trading position for {ticker} for the next trading day after the {subject} below is "
    "released.\n\n"
    'Your first word must be exactly one of: "LONG", "SHORT", or "INSUFFICIENT_INFORMATION".\n'
    'Choose "LONG" if you would take a long position because the {subject} is likely to move '
    "{ticker}'s stock price up over the next trading day.\n"
    'Choose "SHORT" if you would take a short position because the {subject} is likely to move '
    "{ticker}'s stock price down over the next trading day.\n"
    'Choose "INSUFFICIENT_INFORMATION" if the {subject} does not provide enough stock-specific '
    "evidence for a directional trading position.\n\n"
    "Focus only on information materially relevant to {ticker}'s near-term stock return. Ignore "
    "generic market commentary, promotional language, and background information.\n\n"
    "{news}"
)


def resolve_input_dirs(explicit_paths: list[str] | None) -> list[Path]:
    if explicit_paths:
        resolved = [Path(path).expanduser().resolve() for path in explicit_paths]
        if resolved:
            return resolved
    raise FileNotFoundError("No input directories found. Pass one or more --input-dir /path/to/deduplicated/news.")


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
        description="Generate LLM classification responses for deduplicated EODHD news JSONL shards."
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
        help="Directory for response parquet chunks and state files.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help=f"Generative model name or local path. Default: {DEFAULT_MODEL_NAME_OR_PATH}",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Keep rows with date >= start-date (inclusive). Default: {DEFAULT_START_DATE}",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help=f"Keep rows with date <= end-date (inclusive). Default: {DEFAULT_END_DATE}",
    )
    parser.add_argument(
        "--news-truncate-chars",
        type=int,
        default=1500,
        help="Truncate news content to this many characters before building the prompt. Default: 1500",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
        help="Maximum number of generated tokens per article. Default: 16",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="How many prompts to process per outer loop batch.",
    )
    parser.add_argument(
        "--rows-per-chunk",
        type=int,
        default=2048,
        help="How many rows to store per output parquet chunk.",
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
        "--store-prompt",
        action="store_true",
        help="Store the rendered prompt in the output parquet.",
    )
    parser.add_argument(
        "--store-truncated-news",
        action="store_true",
        help="Store the truncated news text in the output parquet.",
    )
    parser.add_argument(
        "--store-pre-response-embedding",
        action="store_true",
        help="Store the final-layer hidden state of the last prompt token before generation begins.",
    )
    parser.add_argument(
        "--pre-response-embedding-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Dtype used when saving pre-response embeddings to .npy chunks.",
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
        "--response-state-file",
        default=None,
        help="Optional explicit path for response state JSON. Default: <output-dir>/response_state.json",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt text for the model.",
    )
    parser.add_argument(
        "--user-instruction",
        default=DEFAULT_USER_INSTRUCTION,
        help="Instruction prefix placed before title and truncated news in the user prompt.",
    )
    parser.add_argument(
        "--post-news-instruction",
        default=DEFAULT_POST_NEWS_INSTRUCTION,
        help="Instruction suffix appended after title and truncated news in the user prompt.",
    )
    parser.add_argument(
        "--promptV2",
        action="store_true",
        help="Use the stricter next-trading-day LONG/SHORT/INSUFFICIENT_INFORMATION prompt.",
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
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if device.type == "cuda" else None
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "token": token,
        "device_map": device_map,
    }
    if device.type == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            print("loading model in 8-bit mode")
        except Exception as exc:
            model_kwargs["torch_dtype"] = torch.float16
            print(
                "[warn] could not enable 8-bit loading; falling back to fp16. "
                f"reason: {type(exc).__name__}: {exc}"
            )
    else:
        model_kwargs["torch_dtype"] = torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
    except TypeError:
        fallback_kwargs = dict(model_kwargs)
        removed = []
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **fallback_kwargs,
            )
        except TypeError:
            fallback_kwargs.pop("quantization_config", None)
            fallback_kwargs.setdefault("torch_dtype", torch.float16 if device.type == "cuda" else torch.float32)
            removed.append("quantization_config")
            print(
                "[warn] transformers/model class does not support some optimized loading args; "
                f"retrying without {', '.join(removed)}."
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **fallback_kwargs,
            )
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        try:
            model.generation_config.max_length = None
        except Exception:
            pass
        for attr_name in ("temperature", "top_p", "top_k", "typical_p"):
            try:
                setattr(model.generation_config, attr_name, None)
            except Exception:
                pass
    if device_map is None:
        model.to(device)
    return tokenizer, model


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


def get_output_numpy_dtype(dtype_name: str) -> np.dtype:
    return np.float16 if dtype_name == "float16" else np.float32


def get_causal_lm_base_model(model):
    base_model = getattr(model, "base_model", None)
    if base_model is not None and base_model is not model:
        return base_model
    base_model_prefix = getattr(model, "base_model_prefix", "")
    if base_model_prefix:
        candidate = getattr(model, base_model_prefix, None)
        if candidate is not None and candidate is not model:
            return candidate
    raise RuntimeError("Could not locate the causal LM base model for pre-response embedding extraction.")


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def default_response_state(
    input_dirs: list[Path],
    output_dir: Path,
    producer_status_files: list[Path],
) -> dict:
    return {
        "version": 1,
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
            "parse_failed_total": 0,
            "chunks_written": 0,
        },
    }


def load_response_state(
    path: Path,
    input_dirs: list[Path],
    output_dir: Path,
    producer_status_files: list[Path],
) -> dict:
    payload = read_json(path)
    if payload is not None:
        return payload
    return default_response_state(input_dirs, output_dir, producer_status_files)


def save_response_summary(path: Path, state: dict, producer_statuses: dict[str, dict | None]) -> None:
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


def parse_article_date(raw_value: object) -> pd.Timestamp | None:
    if raw_value is None:
        return None
    ts = pd.to_datetime(raw_value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def truncate_text_chars(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return text[:max_chars]


def strip_urls(text: str) -> str:
    text_without_urls = re.sub(r"https?://\S+|www\.\S+", "", text)
    return re.sub(r"[ \t]+\n", "\n", text_without_urls).strip()


def classify_title_only(title: str, raw_content: str, cleaned_content: str) -> bool:
    if not title:
        return False
    return not cleaned_content or title == raw_content or title == cleaned_content


def get_prompt_v2_subject(is_title_only: bool) -> str:
    return "news title" if is_title_only else "news"


def build_prompt_components(article: dict, args: argparse.Namespace) -> dict[str, str]:
    title = str(article.get("title", "")).strip()
    content = str(article.get("content", "")).strip()
    content_without_urls = strip_urls(content)
    is_title_only = classify_title_only(title, content, content_without_urls)
    truncated_news = "" if is_title_only else truncate_text_chars(content_without_urls, int(args.news_truncate_chars))
    body = f"{title}\n\n{truncated_news}".strip()
    if args.promptV2:
        ticker = str(article.get("symbol", "")).strip().upper() or "the stock"
        subject = get_prompt_v2_subject(is_title_only)
        user_prompt = DEFAULT_PROMPT_V2_TEMPLATE.format(ticker=ticker, subject=subject, news=body)
    else:
        if args.post_news_instruction:
            subject = get_prompt_v2_subject(is_title_only)
            body = f"{body}\n\n{args.post_news_instruction.format(subject=subject)}".strip()
        user_prompt = f"{args.user_instruction}\n{body}".strip()
    return {
        "title": title,
        "truncated_news": truncated_news,
        "is_title_only": is_title_only,
        "user_prompt": user_prompt,
        "system_prompt": str(args.system_prompt),
    }


def render_chat_prompt(system_prompt: str, user_prompt: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        "<s>[INST] <<SYS>>\n"
        f"{system_prompt}\n"
        "<</SYS>>\n\n"
        f"{user_prompt} [/INST]"
    )


def parse_response_text(raw_response: str) -> tuple[str, str, bool]:
    lines = [line.strip() for line in raw_response.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    if not lines:
        return "UNKNOWN", "", False

    first_line = lines[0].upper()
    earliest_match: tuple[int, str, str] | None = None
    for pattern, candidate, mapped_label in (
        (r"\bINSUF[A-Z_]*", "INSUF", "UNKNOWN"),
        (r"\bSHORT\b", "SHORT", "NO"),
        (r"\bLONG\b", "LONG", "YES"),
    ):
        match = re.search(pattern, first_line)
        if match is None:
            continue
        candidate_match = (match.start(), candidate, mapped_label)
        if earliest_match is None or candidate_match[0] < earliest_match[0]:
            earliest_match = candidate_match

    if earliest_match is None:
        label = "UNKNOWN"
        parsed_prefix = None
    else:
        _pos, parsed_prefix, label = earliest_match
    explanation = ""
    parsed_ok = parsed_prefix is not None
    return label, explanation, parsed_ok


def build_response_row(
    article: dict,
    source_file: Path,
    source_row: int,
    prompt_parts: dict[str, str],
    rendered_prompt: str,
    raw_response: str,
    *,
    store_prompt: bool,
    store_truncated_news: bool,
) -> dict:
    label, explanation, parsed_ok = parse_response_text(raw_response)
    row = {
        "date": str(article.get("date", "")),
        "symbol": str(article.get("symbol", "")).strip().upper(),
        "source_ticker": str(article.get("source_ticker", "")).strip().upper(),
        "title": prompt_parts["title"],
        "is_title_only": bool(prompt_parts["is_title_only"]),
        "link": str(article.get("link", "")),
        "source_file": str(source_file),
        "source_row": int(source_row),
        "news_char_len": len(str(article.get("content", "")).strip()),
        "truncated_news_char_len": len(prompt_parts["truncated_news"]),
        "prompt_char_len": len(rendered_prompt),
        "response_label": label,
        "response_explanation": explanation,
        "response_raw": raw_response,
        "response_parsed_ok": bool(parsed_ok),
    }
    if store_prompt:
        row["prompt"] = rendered_prompt
    if store_truncated_news:
        row["truncated_news"] = prompt_parts["truncated_news"]
    return row


def load_articles_from_jsonl(
    jsonl_path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    limit_rows: int,
    tokenizer,
    args: argparse.Namespace,
) -> tuple[list[str], list[dict], list[tuple[dict, int, dict]], int]:
    prompts: list[str] = []
    prompt_metas: list[tuple[dict, int, dict]] = []
    metadata_rows: list[dict] = []
    scanned_rows = 0

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for source_row, line in enumerate(handle):
            scanned_rows += 1
            article = json.loads(line)
            article_date = parse_article_date(article.get("date"))
            if article_date is None or article_date < start_date or article_date > end_date:
                continue

            prompt_parts = build_prompt_components(article, args)
            rendered_prompt = render_chat_prompt(
                prompt_parts["system_prompt"],
                prompt_parts["user_prompt"],
                tokenizer,
            )
            prompts.append(rendered_prompt)
            prompt_metas.append((article, source_row, prompt_parts))
            metadata_rows.append(
                {
                    "source_file": str(jsonl_path),
                    "source_row": int(source_row),
                }
            )
            if limit_rows and len(prompts) >= limit_rows:
                break

    return prompts, metadata_rows, prompt_metas, scanned_rows


@torch.inference_mode()
def generate_responses_batched(
    prompts: list[str],
    tokenizer,
    model,
    device: torch.device,
    *,
    max_new_tokens: int,
    store_pre_response_embedding: bool,
    pre_response_embedding_dtype: str,
) -> tuple[list[str], np.ndarray | None]:
    if not prompts:
        empty_embeddings = np.empty((0, 0), dtype=get_output_numpy_dtype(pre_response_embedding_dtype))
        return [], empty_embeddings if store_pre_response_embedding else None

    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    base_model = get_causal_lm_base_model(model)
    prefill_outputs = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    last_prompt_hidden = prefill_outputs.last_hidden_state[:, -1, :]

    pre_response_embeddings = None
    if store_pre_response_embedding:
        pre_response_embeddings = (
            last_prompt_hidden.detach()
            .cpu()
            .to(dtype=torch.float32)
            .numpy()
            .astype(get_output_numpy_dtype(pre_response_embedding_dtype), copy=False)
        )

    batch_size = int(input_ids.shape[0])
    if max_new_tokens <= 0:
        return [""] * batch_size, pre_response_embeddings

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Could not locate LM output embeddings for manual greedy decoding.")

    next_token_logits = lm_head(last_prompt_hidden)
    past_key_values = prefill_outputs.past_key_values
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Tokenizer must define pad_token_id or eos_token_id.")

    decode_state_device = next_token_logits.device
    finished = torch.zeros(batch_size, dtype=torch.bool, device=decode_state_device)
    generated_token_batches: list[torch.Tensor] = []

    for _ in range(max_new_tokens):
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        if finished.device != next_tokens.device:
            finished = finished.to(next_tokens.device)
        if eos_token_id is not None:
            eos_fill = torch.full_like(next_tokens, eos_token_id)
            next_tokens = torch.where(finished, eos_fill, next_tokens)
            finished = finished | (next_tokens == eos_token_id)
        generated_token_batches.append(next_tokens.detach().cpu())
        if bool(torch.all(finished)):
            break

        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        decode_outputs = model(
            input_ids=next_tokens.to(input_ids.device).unsqueeze(-1),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        next_token_logits = decode_outputs.logits[:, -1, :]
        past_key_values = decode_outputs.past_key_values

    generated_only = torch.stack(generated_token_batches, dim=1)
    responses = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
    return responses, pre_response_embeddings


def write_response_chunks(
    output_dir: Path,
    rows: list[dict],
    pre_response_embeddings: np.ndarray | None,
    start_chunk_idx: int,
    rows_per_chunk: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_idx = start_chunk_idx
    start = 0
    while start < len(rows):
        end = min(start + rows_per_chunk, len(rows))
        out_path = output_dir / f"responses_{chunk_idx:05d}.parquet"
        pd.DataFrame(rows[start:end]).to_parquet(out_path, index=False)
        if pre_response_embeddings is not None:
            emb_path = output_dir / f"pre_response_embeddings_{chunk_idx:05d}.npy"
            np.save(emb_path, pre_response_embeddings[start:end], allow_pickle=False)
        chunk_idx += 1
        start = end
    return chunk_idx


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
) -> tuple[int, int, int, int]:
    remaining_limit = max(0, args.limit_rows - global_kept_total) if args.limit_rows else 0
    prompts, _metadata_rows, prompt_metadatas, scanned_rows = load_articles_from_jsonl(
        jsonl_path=jsonl_path,
        start_date=start_date,
        end_date=end_date,
        limit_rows=remaining_limit,
        tokenizer=tokenizer,
        args=args,
    )

    file_progress = make_progress_bar(total=len(prompts), desc=jsonl_path.name)
    current_batch_size = args.batch_size
    response_rows: list[dict] = []
    pre_response_embedding_batches: list[np.ndarray] = []
    parse_failed_count = 0

    try:
        start = 0
        while start < len(prompts):
            prompt_batch = prompts[start:start + current_batch_size]
            meta_batch = prompt_metadatas[start:start + current_batch_size]
            try:
                response_batch, pre_response_embedding_batch = generate_responses_batched(
                    prompts=prompt_batch,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    store_pre_response_embedding=bool(args.store_pre_response_embedding),
                    pre_response_embedding_dtype=args.pre_response_embedding_dtype,
                )
            except Exception as exc:
                if not is_oom_error(exc):
                    raise

                old_batch_size = current_batch_size
                new_batch_size = max(1, current_batch_size // 2)
                if old_batch_size == new_batch_size:
                    raise RuntimeError("CUDA out of memory even with batch_size=1.") from exc

                print(
                    "OOM on generation batch; retrying with smaller batch sizes: "
                    f"batch_size {old_batch_size} -> {new_batch_size}"
                )
                current_batch_size = new_batch_size
                clear_memory(device)
                file_progress.set_postfix(batch=current_batch_size)
                continue

            for rendered_prompt, response_text, meta in zip(prompt_batch, response_batch, meta_batch, strict=True):
                article, source_row, prompt_parts = meta
                response_row = build_response_row(
                    article=article,
                    source_file=jsonl_path,
                    source_row=source_row,
                    prompt_parts=prompt_parts,
                    rendered_prompt=rendered_prompt,
                    raw_response=response_text.strip(),
                    store_prompt=bool(args.store_prompt),
                    store_truncated_news=bool(args.store_truncated_news),
                )
                response_rows.append(response_row)
                if not bool(response_row["response_parsed_ok"]):
                    parse_failed_count += 1
            if pre_response_embedding_batch is not None:
                pre_response_embedding_batches.append(pre_response_embedding_batch)

            start += len(prompt_batch)
            file_progress.update(len(prompt_batch))
            parse_failed_pct = (100.0 * parse_failed_count / len(response_rows)) if response_rows else 0.0
            file_progress.set_postfix(batch=current_batch_size, parse_failed_pct=f"{parse_failed_pct:.2f}%")
    finally:
        file_progress.close()

    if not response_rows:
        return scanned_rows, 0, parse_failed_count, start_chunk_idx

    pre_response_embeddings = (
        np.concatenate(pre_response_embedding_batches, axis=0)
        if pre_response_embedding_batches
        else None
    )
    next_chunk_idx = write_response_chunks(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        rows=response_rows,
        pre_response_embeddings=pre_response_embeddings,
        start_chunk_idx=start_chunk_idx,
        rows_per_chunk=args.rows_per_chunk,
    )
    return scanned_rows, len(response_rows), parse_failed_count, next_chunk_idx


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
    response_state_file = (
        Path(args.response_state_file).expanduser().resolve()
        if args.response_state_file
        else output_dir / "response_state.json"
    )

    tokenizer, model = load_model_and_tokenizer(args.model_name_or_path, device)

    manifest = {
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "producer_status_files": [str(path) for path in producer_status_files],
        "model_name_or_path": args.model_name_or_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "news_truncate_chars": args.news_truncate_chars,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "rows_per_chunk": args.rows_per_chunk,
        "device": str(device),
        "store_prompt": bool(args.store_prompt),
        "store_truncated_news": bool(args.store_truncated_news),
        "store_pre_response_embedding": bool(args.store_pre_response_embedding),
        "pre_response_embedding_dtype": args.pre_response_embedding_dtype,
        "promptV2": bool(args.promptV2),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "poll_seconds": args.poll_seconds,
        "producer_timeout_seconds": args.producer_timeout_seconds,
        "system_prompt": args.system_prompt,
        "user_instruction": args.user_instruction,
        "post_news_instruction": args.post_news_instruction,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    state = load_response_state(response_state_file, input_dirs, output_dir, producer_status_files)
    processed_tasks = set(state.get("processed_tasks", []))
    kept_total = int(state.get("stats", {}).get("kept_total", 0))
    scanned_total = int(state.get("stats", {}).get("scanned_total", 0))
    parse_failed_total = int(state.get("stats", {}).get("parse_failed_total", 0))
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
            global_parse_failed_pct = (100.0 * parse_failed_total / kept_total) if kept_total else 0.0
            jsonl_progress.set_postfix(
                processed=len(processed_tasks),
                produced=produced_assigned_count,
                producer_total=produced_global_count,
                parse_failed_pct=f"{global_parse_failed_pct:.2f}%",
                current=state.get("current_task") or "-",
            )

            if pending_tasks:
                for task in pending_tasks:
                    if args.limit_rows and kept_total >= args.limit_rows:
                        break

                    task_id = str(task["task_id"])
                    jsonl_path = Path(task["jsonl_path"])
                    state["current_task"] = task_id
                    save_json(response_state_file, state)
                    save_response_summary(summary_path, state, producer_statuses)
                    jsonl_progress.set_postfix(
                        processed=len(processed_tasks),
                        produced=produced_assigned_count,
                        producer_total=produced_global_count,
                        parse_failed_pct=f"{global_parse_failed_pct:.2f}%",
                        current=task_id,
                    )

                    scanned_rows, kept_rows, parse_failed_rows, next_chunk_idx = process_jsonl_file(
                        jsonl_path,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        args=args,
                        start_date=start_date,
                        end_date=end_date,
                        start_chunk_idx=chunk_idx,
                        global_kept_total=kept_total,
                    )
                    scanned_total += scanned_rows
                    kept_total += kept_rows
                    parse_failed_total += parse_failed_rows
                    processed_tasks.add(task_id)
                    state["processed_tasks"] = sorted(processed_tasks)
                    state["current_task"] = None
                    state["next_chunk_idx"] = next_chunk_idx
                    state["stats"]["tasks_processed"] = len(processed_tasks)
                    state["stats"]["scanned_total"] = scanned_total
                    state["stats"]["kept_total"] = kept_total
                    state["stats"]["parse_failed_total"] = parse_failed_total
                    state["stats"]["chunks_written"] = next_chunk_idx
                    chunk_idx = next_chunk_idx
                    save_json(response_state_file, state)
                    save_response_summary(summary_path, state, producer_statuses)
                    global_parse_failed_pct = (100.0 * parse_failed_total / kept_total) if kept_total else 0.0
                    jsonl_progress.update(1)
                    jsonl_progress.set_postfix(
                        processed=len(processed_tasks),
                        produced=produced_assigned_count,
                        producer_total=produced_global_count,
                        parse_failed_pct=f"{global_parse_failed_pct:.2f}%",
                        current="-",
                    )

                if args.limit_rows and kept_total >= args.limit_rows:
                    break
                continue

            if not producer_statuses:
                break

            save_response_summary(summary_path, state, producer_statuses)
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

    save_response_summary(summary_path, state, read_producer_statuses(sources))
    print(json.dumps(read_json(summary_path) or {}, indent=2))


if __name__ == "__main__":
    main()
