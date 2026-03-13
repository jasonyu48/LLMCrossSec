from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MODES = ["mean", "article", "sum_head"]
DEFAULT_L2S = [1e-6, 1e-4, 1e-2, 0.1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run train_news_return_heads.py over a grid of linear-train-mode and linear-l2 values, "
            "then collect metrics into a summary table."
        )
    )
    parser.add_argument("--news-emb-dir", required=True)
    parser.add_argument("--ohlcv-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=DEFAULT_MODES)
    parser.add_argument("--linear-l2s", nargs="+", type=float, default=DEFAULT_L2S)
    parser.add_argument("--train-start", default="20200101")
    parser.add_argument("--train-end", default="20230717")
    parser.add_argument("--val-start", default="20230719")
    parser.add_argument("--val-end", default="20260306")
    parser.add_argument("--min-ic-universe", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true", help="Skip a run if metrics.json already exists.")
    parser.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Extra args passed through to train_news_return_heads.py. Prefix with --extra-train-args.",
    )
    return parser.parse_args()


def format_l2_tag(value: float) -> str:
    text = f"{value:.12g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()

    workspace = Path(__file__).resolve().parent
    train_script = workspace / "train_news_return_heads.py"
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    failed_runs: list[dict[str, object]] = []

    total_runs = len(args.modes) * len(args.linear_l2s)
    run_idx = 0

    for mode in args.modes:
        for l2 in args.linear_l2s:
            run_idx += 1
            l2_tag = format_l2_tag(float(l2))
            run_name = f"mode_{mode}__l2_{l2_tag}"
            out_dir = output_root / run_name
            metrics_path = out_dir / "metrics.json"

            if args.skip_existing and metrics_path.is_file():
                print(f"[{run_idx}/{total_runs}] skipping existing {run_name}")
                payload = load_json(metrics_path)
                linear_metrics = payload.get("metrics", {}).get("linear", {})
                summary_rows.append(build_summary_row(run_name, mode, float(l2), out_dir, linear_metrics))
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(train_script),
                "--news-emb-dir",
                str(Path(args.news_emb_dir).expanduser().resolve()),
                "--ohlcv-dir",
                str(Path(args.ohlcv_dir).expanduser().resolve()),
                "--output-dir",
                str(out_dir),
                "--heads",
                "linear",
                "--linear-train-mode",
                mode,
                "--linear-l2",
                str(l2),
                "--min-ic-universe",
                str(args.min_ic_universe),
                "--train-start",
                str(args.train_start),
                "--train-end",
                str(args.train_end),
                "--val-start",
                str(args.val_start),
                "--val-end",
                str(args.val_end),
                "--seed",
                str(args.seed),
            ]
            if args.extra_train_args:
                cmd.extend(args.extra_train_args)

            print(f"[{run_idx}/{total_runs}] running {run_name}")
            print(" ".join(shell_quote(part) for part in cmd))
            start = time.time()
            result = subprocess.run(cmd, cwd=workspace)
            elapsed = time.time() - start

            if result.returncode != 0:
                failed_runs.append(
                    {
                        "run_name": run_name,
                        "mode": mode,
                        "linear_l2": float(l2),
                        "returncode": int(result.returncode),
                        "elapsed_sec": elapsed,
                        "output_dir": str(out_dir),
                    }
                )
                print(f"[{run_idx}/{total_runs}] failed {run_name} in {elapsed:.1f}s")
                continue

            if not metrics_path.is_file():
                failed_runs.append(
                    {
                        "run_name": run_name,
                        "mode": mode,
                        "linear_l2": float(l2),
                        "returncode": 0,
                        "elapsed_sec": elapsed,
                        "output_dir": str(out_dir),
                        "error": "metrics.json not found after successful run",
                    }
                )
                print(f"[{run_idx}/{total_runs}] missing metrics.json for {run_name}")
                continue

            payload = load_json(metrics_path)
            linear_metrics = payload.get("metrics", {}).get("linear", {})
            row = build_summary_row(run_name, mode, float(l2), out_dir, linear_metrics)
            row["elapsed_sec"] = elapsed
            summary_rows.append(row)
            print(
                f"[{run_idx}/{total_runs}] done {run_name} "
                f"val_ic={row['val_ic_mean']} val_ir={row['val_ic_ir']} elapsed={elapsed:.1f}s"
            )

    summary_rows.sort(
        key=lambda row: (
            safe_sort_float(row.get("val_ic_mean")),
            safe_sort_float(row.get("val_ic_ir")),
        ),
        reverse=True,
    )

    summary_json = output_root / "grid_search_summary.json"
    summary_csv = output_root / "grid_search_summary.csv"
    failed_json = output_root / "grid_search_failures.json"

    summary_payload = {
        "config": {
            "modes": list(args.modes),
            "linear_l2s": [float(v) for v in args.linear_l2s],
            "train_start": str(args.train_start),
            "train_end": str(args.train_end),
            "val_start": str(args.val_start),
            "val_end": str(args.val_end),
            "min_ic_universe": int(args.min_ic_universe),
            "seed": int(args.seed),
            "extra_train_args": list(args.extra_train_args or []),
        },
        "results": summary_rows,
        "n_succeeded": len(summary_rows),
        "n_failed": len(failed_runs),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    write_summary_csv(summary_csv, summary_rows)
    failed_json.write_text(json.dumps(failed_runs, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"wrote summary json: {summary_json}")
    print(f"wrote summary csv:  {summary_csv}")
    print(f"wrote failures:     {failed_json}")
    if summary_rows:
        best = summary_rows[0]
        print(
            f"best by val_ic_mean: {best['run_name']} "
            f"(mode={best['mode']}, l2={best['linear_l2']}, val_ic={best['val_ic_mean']}, val_ir={best['val_ic_ir']})"
        )


def build_summary_row(
    run_name: str,
    mode: str,
    l2: float,
    output_dir: Path,
    linear_metrics: dict,
) -> dict[str, object]:
    train_rank_ic = linear_metrics.get("train_rank_ic", {})
    val_rank_ic = linear_metrics.get("val_rank_ic", {})
    train_pearson_ic = linear_metrics.get("train_pearson_ic", {})
    val_pearson_ic = linear_metrics.get("val_pearson_ic", {})
    train_zero = linear_metrics.get("train_rank_ic_with_no_news_zero", {})
    val_zero = linear_metrics.get("val_rank_ic_with_no_news_zero", {})
    train_pearson_zero = linear_metrics.get("train_pearson_ic_with_no_news_zero", {})
    val_pearson_zero = linear_metrics.get("val_pearson_ic_with_no_news_zero", {})
    return {
        "run_name": run_name,
        "mode": mode,
        "linear_l2": l2,
        "output_dir": str(output_dir),
        "train_ic_mean": train_rank_ic.get("ic_mean"),
        "train_ic_ir": train_rank_ic.get("ic_ir"),
        "val_ic_mean": val_rank_ic.get("ic_mean"),
        "val_ic_ir": val_rank_ic.get("ic_ir"),
        "train_pearson_ic_mean": train_pearson_ic.get("ic_mean"),
        "train_pearson_ic_ir": train_pearson_ic.get("ic_ir"),
        "val_pearson_ic_mean": val_pearson_ic.get("ic_mean"),
        "val_pearson_ic_ir": val_pearson_ic.get("ic_ir"),
        "train_ic_zero_mean": train_zero.get("ic_mean"),
        "train_ic_zero_ir": train_zero.get("ic_ir"),
        "val_ic_zero_mean": val_zero.get("ic_mean"),
        "val_ic_zero_ir": val_zero.get("ic_ir"),
        "train_pearson_ic_zero_mean": train_pearson_zero.get("ic_mean"),
        "train_pearson_ic_zero_ir": train_pearson_zero.get("ic_ir"),
        "val_pearson_ic_zero_mean": val_pearson_zero.get("ic_mean"),
        "val_pearson_ic_zero_ir": val_pearson_zero.get("ic_ir"),
        "n_train_samples": linear_metrics.get("n_train_samples"),
        "n_val_samples": linear_metrics.get("n_val_samples"),
        "n_train_fit_samples": linear_metrics.get("n_train_fit_samples"),
    }


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "run_name",
        "mode",
        "linear_l2",
        "val_ic_mean",
        "val_ic_ir",
        "val_pearson_ic_mean",
        "val_pearson_ic_ir",
        "val_ic_zero_mean",
        "val_ic_zero_ir",
        "val_pearson_ic_zero_mean",
        "val_pearson_ic_zero_ir",
        "train_ic_mean",
        "train_ic_ir",
        "train_pearson_ic_mean",
        "train_pearson_ic_ir",
        "train_ic_zero_mean",
        "train_ic_zero_ir",
        "train_pearson_ic_zero_mean",
        "train_pearson_ic_zero_ir",
        "n_train_samples",
        "n_val_samples",
        "n_train_fit_samples",
        "elapsed_sec",
        "output_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def safe_sort_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if out != out:
        return float("-inf")
    return out


def shell_quote(text: str) -> str:
    if not text:
        return "''"
    if all(ch.isalnum() or ch in "._/-=:" for ch in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
