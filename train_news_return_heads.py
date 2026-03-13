from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_CACHE_DIR = Path("/export/fs06/jyu197/eodhd/.cache/train_news_return_heads")
CACHE_VERSION = 2
MIDNIGHT_SHIFT_DOMAINS = ("investorplace.com", "fool.com")


class NullProgressBar:
    def update(self, _: int = 1) -> None:
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
            "Train news-embedding return heads with 9:00 America/New_York signal cutoff "
            "and evaluate Rank IC."
        )
    )
    parser.add_argument("--news-emb-dir", required=True, help="Directory with metadata_*.parquet and embeddings_*.npy.")
    parser.add_argument("--ohlcv-dir", required=True, help="Directory produced by fetch_eodhd_ohlcv_by_ticker.py.")
    parser.add_argument("--output-dir", required=True, help="Output directory for models, metrics and predictions.")
    parser.add_argument(
        "--heads",
        nargs="+",
        choices=["linear", "ae_linear"],
        default=["linear", "ae_linear"],
        help="Which heads to train.",
    )
    parser.add_argument("--train-start", default="20200101", help="Train split start, inclusive, YYYYMMDD.")
    parser.add_argument("--train-end", default="20211231", help="Train split end, inclusive, YYYYMMDD.")
    parser.add_argument("--val-start", default="20220101", help="Val split start, inclusive, YYYYMMDD.")
    parser.add_argument("--val-end", default="20220901", help="Val split end, inclusive, YYYYMMDD.")
    parser.add_argument(
        "--linear-l2",
        type=float,
        default=1e-4,
        help="L2 coefficient for closed-form ridge regression in mean/article modes.",
    )
    parser.add_argument(
        "--linear-train-mode",
        choices=["mean", "article", "sum_head"],
        default="mean",
        help=(
            "How to map same-day same-symbol news into the supervised linear target. "
            "'mean' fits head(mean(embeddings))->r, "
            "'article' fits head(embedding)->same r for every article, "
            "'sum_head' fits sum(head(embedding))->r with Adam."
        ),
    )
    parser.add_argument("--adam-epochs", type=int, default=10, help="Epochs for sum_head Adam training.")
    parser.add_argument("--adam-lr", type=float, default=1e-4, help="Learning rate for sum_head Adam training.")
    parser.add_argument("--adam-batch-size", type=int, default=4096, help="Batch size for sum_head Adam training.")
    parser.add_argument(
        "--adam-weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay for sum_head training.",
    )
    parser.add_argument("--ae-latent-dim", type=int, default=128)
    parser.add_argument("--ae-hidden-dim", type=int, default=512)
    parser.add_argument("--ae-epochs", type=int, default=8)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--ae-batch-size", type=int, default=4096)
    parser.add_argument(
        "--max-train-news-for-ae",
        type=int,
        default=1200000,
        help="Cap train news rows used to train AE (0 means no cap).",
    )
    parser.add_argument("--min-ic-universe", type=int, default=20, help="Min symbols per day to compute daily IC.")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"Directory for reusable preprocessing caches. Default: {DEFAULT_CACHE_DIR}",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable disk caches for daily target universe and raw grouped features.",
    )
    parser.add_argument(
        "--overnight-news",
        action="store_true",
        help="If set, only use news between the previous trading day's close and the current trading day 9:00 America/New_York cutoff.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def chunk_pair_signature(chunk_pairs: list[tuple[Path, Path]], news_emb_dir: Path) -> list[dict[str, object]]:
    sigs: list[dict[str, object]] = []
    for meta_path, emb_path in chunk_pairs:
        sigs.append(
            {
                "meta_rel": str(meta_path.relative_to(news_emb_dir)),
                "meta_size": int(meta_path.stat().st_size),
                "meta_mtime_ns": int(meta_path.stat().st_mtime_ns),
                "emb_rel": str(emb_path.relative_to(news_emb_dir)),
                "emb_size": int(emb_path.stat().st_size),
                "emb_mtime_ns": int(emb_path.stat().st_mtime_ns),
            }
        )
    return sigs


def build_daily_target_universe_cache_key(ohlcv_dir: Path, cfg: argparse.Namespace, provider: "OhlcvProvider") -> str:
    payload = {
        "version": CACHE_VERSION,
        "kind": "daily_target_universe",
        "ohlcv_dir": str(ohlcv_dir.resolve()),
        "ohlcv_file_count": int(len(set(provider.file_map.values()))),
        "train_start": str(cfg.train_start),
        "train_end": str(cfg.train_end),
        "val_start": str(cfg.val_start),
        "val_end": str(cfg.val_end),
    }
    return make_cache_key(payload)


def build_group_feature_cache_key(
    *,
    news_emb_dir: Path,
    ohlcv_dir: Path,
    chunk_pairs: list[tuple[Path, Path]],
    cfg: argparse.Namespace,
    collect_train_news: bool,
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "kind": "group_features_raw",
        "news_emb_dir": str(news_emb_dir.resolve()),
        "ohlcv_dir": str(ohlcv_dir.resolve()),
        "train_start": str(cfg.train_start),
        "train_end": str(cfg.train_end),
        "val_start": str(cfg.val_start),
        "val_end": str(cfg.val_end),
        "collect_train_news": bool(collect_train_news),
        "max_train_news_for_ae": int(cfg.max_train_news_for_ae),
        "overnight_news": bool(cfg.overnight_news),
        "chunk_pairs": chunk_pair_signature(chunk_pairs, news_emb_dir),
    }
    return make_cache_key(payload)


def load_or_build_daily_target_universe_cached(
    *,
    provider: "OhlcvProvider",
    cfg: argparse.Namespace,
    cache_dir: Path,
    enabled: bool,
) -> dict[str, dict[int, list[tuple[str, float]]]]:
    if not enabled:
        return build_daily_target_universe_by_split(provider, cfg)

    cache_root = cache_dir / "daily_target_universe"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = build_daily_target_universe_cache_key(provider.ohlcv_dir, cfg, provider)
    cache_path = cache_root / f"{cache_key}.pkl"
    meta_path = cache_root / f"{cache_key}.json"
    if cache_path.is_file():
        print(f"[cache] loading daily target universe from {cache_path}")
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    payload = build_daily_target_universe_by_split(provider, cfg)
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path.write_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "kind": "daily_target_universe",
                "cache_path": str(cache_path),
                "ohlcv_dir": str(provider.ohlcv_dir.resolve()),
                "train_start": str(cfg.train_start),
                "train_end": str(cfg.train_end),
                "val_start": str(cfg.val_start),
                "val_end": str(cfg.val_end),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[cache] saved daily target universe to {cache_path}")
    return payload


def load_or_build_raw_group_features_cached(
    *,
    news_emb_dir: Path,
    ohlcv_dir: Path,
    chunk_pairs: list[tuple[Path, Path]],
    provider: "OhlcvProvider",
    cfg: argparse.Namespace,
    collect_train_news: bool,
    cache_dir: Path,
    enabled: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    if not enabled:
        return aggregate_group_features(
            chunk_pairs=chunk_pairs,
            provider=provider,
            cfg=cfg,
            transform_batch=None,
            collect_train_news=collect_train_news,
            progress_desc="Aggregating raw news chunks",
        )

    cache_root = cache_dir / "group_features_raw"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = build_group_feature_cache_key(
        news_emb_dir=news_emb_dir,
        ohlcv_dir=ohlcv_dir,
        chunk_pairs=chunk_pairs,
        cfg=cfg,
        collect_train_news=collect_train_news,
    )
    cache_path = cache_root / cache_key
    group_df_path = cache_path / "group_df.parquet"
    x_raw_path = cache_path / "x_raw.npy"
    ae_train_news_path = cache_path / "ae_train_news.npy"
    meta_path = cache_path / "meta.json"

    if group_df_path.is_file() and x_raw_path.is_file() and (not collect_train_news or ae_train_news_path.is_file()):
        print(f"[cache] loading raw group features from {cache_path}")
        group_df = pd.read_parquet(group_df_path)
        x_raw = np.load(x_raw_path, mmap_mode=None)
        ae_train_news = np.load(ae_train_news_path, mmap_mode=None) if collect_train_news else None
        return group_df, x_raw, ae_train_news

    group_df, x_raw, ae_train_news = aggregate_group_features(
        chunk_pairs=chunk_pairs,
        provider=provider,
        cfg=cfg,
        transform_batch=None,
        collect_train_news=collect_train_news,
        progress_desc="Aggregating raw news chunks",
    )
    cache_path.mkdir(parents=True, exist_ok=True)
    group_df.to_parquet(group_df_path, index=False)
    np.save(x_raw_path, x_raw, allow_pickle=False)
    if collect_train_news and ae_train_news is not None:
        np.save(ae_train_news_path, ae_train_news, allow_pickle=False)
    meta_path.write_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "kind": "group_features_raw",
                "cache_dir": str(cache_path),
                "news_emb_dir": str(news_emb_dir.resolve()),
                "ohlcv_dir": str(ohlcv_dir.resolve()),
                "train_start": str(cfg.train_start),
                "train_end": str(cfg.train_end),
                "val_start": str(cfg.val_start),
                "val_end": str(cfg.val_end),
                "collect_train_news": bool(collect_train_news),
                "max_train_news_for_ae": int(cfg.max_train_news_for_ae),
                "n_group_rows": int(len(group_df)),
                "x_raw_shape": list(map(int, x_raw.shape)),
                "ae_train_news_shape": list(map(int, ae_train_news.shape)) if ae_train_news is not None else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[cache] saved raw group features to {cache_path}")
    return group_df, x_raw, ae_train_news


def discover_chunk_pairs(news_emb_dir: Path) -> list[tuple[Path, Path]]:
    shard_dirs = sorted(p for p in news_emb_dir.iterdir() if p.is_dir() and p.name.startswith("shard_"))
    roots = shard_dirs or [news_emb_dir]
    pairs: list[tuple[Path, Path]] = []
    for root in roots:
        metadata_paths = sorted(root.glob("metadata_*.parquet"))
        if metadata_paths:
            meta_prefix = "metadata_"
            emb_prefix = "embeddings_"
        else:
            metadata_paths = sorted(root.glob("responses_*.parquet"))
            meta_prefix = "responses_"
            emb_prefix = "pre_response_embeddings_"

        for meta_path in metadata_paths:
            emb_name = meta_path.name.replace(meta_prefix, emb_prefix).replace(".parquet", ".npy")
            emb_path = root / emb_name
            if not emb_path.is_file():
                raise FileNotFoundError(f"Missing embedding file for {meta_path}: {emb_path}")
            pairs.append((meta_path, emb_path))
    if not pairs:
        raise FileNotFoundError(
            f"No compatible metadata/embedding chunk pairs found under {news_emb_dir}. "
            "Expected either metadata_*.parquet + embeddings_*.npy or "
            "responses_*.parquet + pre_response_embeddings_*.npy."
        )
    return pairs


def split_of_date(date_int: int, cfg: argparse.Namespace) -> str | None:
    if int(cfg.train_start) <= date_int <= int(cfg.train_end):
        return "train"
    if int(cfg.val_start) <= date_int <= int(cfg.val_end):
        return "val"
    return None


@dataclass
class SymbolIndex:
    symbol: str
    cutoff_ns: np.ndarray
    signal_window_open_ns: np.ndarray
    overnight_window_open_ns: np.ndarray
    trade_dates_int: np.ndarray
    target_by_date: dict[int, float]

    def assign_news_to_trade_date(self, ts_ns: int, cfg: argparse.Namespace) -> tuple[int, float, str] | None:
        idx = int(np.searchsorted(self.cutoff_ns, ts_ns, side="left"))
        if idx <= 0 or idx >= len(self.trade_dates_int):
            return None
        window_open_ns = self.overnight_window_open_ns if bool(cfg.overnight_news) else self.signal_window_open_ns
        if ts_ns <= int(window_open_ns[idx]):
            return None
        trade_date = int(self.trade_dates_int[idx])
        target = self.target_by_date.get(trade_date)
        if target is None or not np.isfinite(target):
            return None
        split = split_of_date(trade_date, cfg)
        if split is None:
            return None
        return trade_date, float(target), split


class OhlcvProvider:
    def __init__(self, ohlcv_dir: Path):
        self.ohlcv_dir = ohlcv_dir
        self.file_map = self._build_file_map()
        self.cache: dict[str, SymbolIndex | None] = {}

    def _build_file_map(self) -> dict[str, Path]:
        parquet_root = self.ohlcv_dir / "parquet"
        if not parquet_root.is_dir():
            raise FileNotFoundError(f"Expected OHLCV parquet root: {parquet_root}")
        out: dict[str, Path] = {}
        for path in parquet_root.glob("*/*.parquet"):
            stem = path.stem.upper()
            # OHLCV files are named like AAPL.NASDAQ.parquet / AA.NYSE.parquet.
            # News symbols are like AAPL.US, so map both variants.
            out.setdefault(stem, path)
            if "." in stem:
                base = stem.rsplit(".", 1)[0]
                out.setdefault(base, path)
                out.setdefault(f"{base}.US", path)
        return out

    def get(self, symbol: str) -> SymbolIndex | None:
        key = symbol.upper()
        if key in self.cache:
            return self.cache[key]
        path = self.file_map.get(key)
        if path is None:
            self.cache[key] = None
            return None
        idx = self._load_symbol_index(path, key)
        self.cache[key] = idx
        return idx

    def _load_symbol_index(self, path: Path, symbol: str) -> SymbolIndex | None:
        df = pd.read_parquet(path, columns=["date", "open"])
        if df.empty:
            return None
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df = df.dropna(subset=["date", "open"]).sort_values("date").reset_index(drop=True)
        if len(df) < 2:
            return None

        dates = df["date"].dt.date.to_numpy()
        open_px = df["open"].to_numpy(dtype=np.float64)
        trade_dates_int = np.array([int(d.strftime("%Y%m%d")) for d in dates], dtype=np.int32)

        cutoff_ns = []
        close_ns = []
        for d in dates:
            cutoff_local = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=9, minute=0, second=0, tz=NY_TZ)
            cutoff_ns.append(int(cutoff_local.tz_convert("UTC").value))
            close_local = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=16, minute=0, second=0, tz=NY_TZ)
            close_ns.append(int(close_local.tz_convert("UTC").value))
        cutoff_ns_arr = np.asarray(cutoff_ns, dtype=np.int64)
        close_ns_arr = np.asarray(close_ns, dtype=np.int64)
        signal_window_open_ns = np.empty_like(cutoff_ns_arr)
        signal_window_open_ns[0] = np.iinfo(np.int64).min
        signal_window_open_ns[1:] = cutoff_ns_arr[:-1]
        overnight_window_open_ns = np.empty_like(close_ns_arr)
        overnight_window_open_ns[0] = np.iinfo(np.int64).min
        overnight_window_open_ns[1:] = close_ns_arr[:-1]

        target_by_date: dict[int, float] = {}
        prev_open = open_px[:-1]
        next_open = open_px[1:]
        valid = np.isfinite(prev_open) & np.isfinite(next_open) & (prev_open > 0) & (next_open > 0)
        rets = np.full(len(prev_open), np.nan, dtype=np.float64)
        rets[valid] = (next_open[valid] / prev_open[valid]) - 1.0
        for i, ret in enumerate(rets):
            target_by_date[int(trade_dates_int[i])] = float(ret)

        return SymbolIndex(
            symbol=symbol,
            cutoff_ns=cutoff_ns_arr,
            signal_window_open_ns=signal_window_open_ns,
            overnight_window_open_ns=overnight_window_open_ns,
            trade_dates_int=trade_dates_int,
            target_by_date=target_by_date,
        )


def _fit_to_batch(emb_batch: np.ndarray, transform_batch: callable | None) -> np.ndarray:
    arr = emb_batch.astype(np.float32, copy=False)
    if transform_batch is None:
        return arr
    return transform_batch(arr)


def finite_row_mask(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={arr.shape}")
    return np.isfinite(arr).all(axis=1)


def _link_matches_midnight_shift_domain(link: object) -> bool:
    text = str(link or "").strip().lower()
    if not text:
        return False
    host = urlparse(text).netloc.strip().lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return False
    return any(host == domain or host.endswith(f".{domain}") for domain in MIDNIGHT_SHIFT_DOMAINS)


def parse_news_timestamps_utc(values: pd.Series, links: pd.Series | None = None) -> pd.Series:
    try:
        ts = pd.to_datetime(values, utc=True, errors="coerce", format="ISO8601")
    except (TypeError, ValueError):
        ts = pd.to_datetime(values, utc=True, errors="coerce")

    if links is None:
        return ts

    midnight_mask = ts.notna() & (ts == ts.dt.normalize())
    if not bool(midnight_mask.any()):
        return ts

    domain_mask = links.map(_link_matches_midnight_shift_domain).astype(bool)
    shift_mask = midnight_mask & domain_mask.reindex(ts.index, fill_value=False)
    if bool(shift_mask.any()):
        ts = ts.copy()
        ts.loc[shift_mask] = ts.loc[shift_mask] + pd.Timedelta(days=1)
    return ts


def aggregate_group_features(
    *,
    chunk_pairs: list[tuple[Path, Path]],
    provider: OhlcvProvider,
    cfg: argparse.Namespace,
    transform_batch: callable | None = None,
    collect_train_news: bool = False,
    progress_desc: str = "Aggregating chunks",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    group_sum: dict[tuple[str, int], np.ndarray] = {}
    group_count: dict[tuple[str, int], int] = {}
    group_target: dict[tuple[str, int], float] = {}
    group_split: dict[tuple[str, int], str] = {}
    ae_train_news_rows: list[np.ndarray] = []
    max_rows = int(cfg.max_train_news_for_ae)
    total_rows = 0
    valid_symbol_rows = 0
    valid_ts_rows = 0
    with_ohlcv_rows = 0
    assigned_rows = 0

    pbar = make_progress_bar(total=len(chunk_pairs), desc=progress_desc, unit="chunk")
    try:
        for meta_path, emb_path in chunk_pairs:
            meta = pd.read_parquet(meta_path, columns=["symbol", "date", "link"])
            emb = np.load(emb_path, mmap_mode=None)
            if len(meta) != len(emb):
                raise ValueError(f"Row mismatch between {meta_path} and {emb_path}: {len(meta)} vs {len(emb)}")

            symbols = meta["symbol"].fillna("").astype(str).str.upper().to_numpy()
            ts = parse_news_timestamps_utc(meta["date"], meta["link"])
            ts_ns = ts.view("int64")
            total_rows += len(meta)

            valid_idx: list[int] = []
            valid_keys: list[tuple[str, int]] = []
            valid_splits: list[str] = []
            valid_targets: list[float] = []
            for i in range(len(meta)):
                symbol = symbols[i]
                if not symbol:
                    continue
                valid_symbol_rows += 1
                ts_i = int(ts_ns[i])
                if ts_i <= 0:
                    continue
                valid_ts_rows += 1
                sym_idx = provider.get(symbol)
                if sym_idx is None:
                    continue
                with_ohlcv_rows += 1
                assigned = sym_idx.assign_news_to_trade_date(ts_i, cfg)
                if assigned is None:
                    continue
                assigned_rows += 1
                trade_date, target, split = assigned
                key = (symbol, int(trade_date))
                valid_idx.append(i)
                valid_keys.append(key)
                valid_splits.append(split)
                valid_targets.append(float(target))

            if not valid_idx:
                pbar.update(1)
                continue

            emb_valid = _fit_to_batch(emb[np.asarray(valid_idx, dtype=np.int64)], transform_batch)
            finite_mask = finite_row_mask(emb_valid)
            if not finite_mask.all():
                emb_valid = emb_valid[finite_mask]
                valid_keys = [key for key, keep in zip(valid_keys, finite_mask.tolist(), strict=True) if keep]
                valid_splits = [split for split, keep in zip(valid_splits, finite_mask.tolist(), strict=True) if keep]
                valid_targets = [target for target, keep in zip(valid_targets, finite_mask.tolist(), strict=True) if keep]

            for key, split, target, vec in zip(valid_keys, valid_splits, valid_targets, emb_valid, strict=True):
                if key not in group_sum:
                    group_sum[key] = np.array(vec, dtype=np.float32, copy=True)
                    group_count[key] = 1
                    group_target[key] = target
                    group_split[key] = split
                else:
                    group_sum[key] += vec
                    group_count[key] += 1

                if collect_train_news and split == "train":
                    if max_rows > 0 and len(ae_train_news_rows) >= max_rows:
                        continue
                    ae_train_news_rows.append(np.array(vec, dtype=np.float32, copy=True))
            pbar.update(1)
    finally:
        pbar.close()

    rows = []
    feats = []
    for (symbol, trade_date), feat in group_sum.items():
        rows.append(
            {
                "symbol": symbol,
                "trade_date": int(trade_date),
                "split": group_split[(symbol, trade_date)],
                "target_return": float(group_target[(symbol, trade_date)]),
                "news_count": int(group_count[(symbol, trade_date)]),
            }
        )
        feats.append(feat)
    if not rows:
        raise RuntimeError(
            "No grouped samples created. "
            f"total_rows={total_rows}, valid_symbol_rows={valid_symbol_rows}, valid_ts_rows={valid_ts_rows}, "
            f"with_ohlcv_rows={with_ohlcv_rows}, assigned_rows={assigned_rows}. "
            "Check paths/symbol alignment/split dates."
        )

    group_df = pd.DataFrame(rows).sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    feat_mat = np.stack(feats, axis=0).astype(np.float32, copy=False)

    train_news_arr = None
    if collect_train_news and ae_train_news_rows:
        train_news_arr = np.stack(ae_train_news_rows, axis=0).astype(np.float32, copy=False)

    return group_df, feat_mat, train_news_arr


def fit_linear_closed_form_from_stats(
    xtx: np.ndarray,
    xty: np.ndarray,
    *,
    l2: float,
) -> tuple[np.ndarray, float]:
    reg = np.eye(xtx.shape[0], dtype=np.float64) * float(l2)
    reg[-1, -1] = 0.0
    lhs = xtx + reg
    try:
        beta = np.linalg.solve(lhs, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ xty
    w = beta[:-1].astype(np.float32, copy=False)
    b = float(beta[-1])
    return w, b


def fit_linear_closed_form(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    l2: float,
) -> tuple[np.ndarray, float]:
    x = x_train.astype(np.float64, copy=False)
    y = y_train.astype(np.float64, copy=False)
    ones = np.ones((x.shape[0], 1), dtype=np.float64)
    x_aug = np.concatenate([x, ones], axis=1)

    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=np.float64) * float(l2)
    reg[-1, -1] = 0.0  # Do not regularize bias.
    lhs = xtx + reg
    rhs = x_aug.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    w = beta[:-1].astype(np.float32, copy=False)
    b = float(beta[-1])
    return w, b


def build_group_feature_matrix(group_df: pd.DataFrame, feat_sum: np.ndarray, inference_mode: str) -> np.ndarray:
    if inference_mode == "mean":
        counts = group_df["news_count"].to_numpy(dtype=np.float32)
        return feat_sum.astype(np.float32, copy=False) / counts[:, None]
    if inference_mode == "sum_embedding":
        return feat_sum.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported grouped feature mode: {inference_mode}")


def predict_group_scores(
    group_df: pd.DataFrame,
    feat_sum: np.ndarray,
    w: np.ndarray,
    b: float,
    *,
    inference_mode: str,
) -> np.ndarray:
    if inference_mode == "sum_head":
        counts = group_df["news_count"].to_numpy(dtype=np.float32)
        pred = feat_sum.astype(np.float32, copy=False) @ w.astype(np.float32, copy=False) + counts * np.float32(b)
        return pred.astype(np.float32, copy=False)
    x = build_group_feature_matrix(group_df, feat_sum, inference_mode)
    return predict_linear_closed_form(x, w, b)


def fit_linear_article_closed_form_streaming(
    *,
    chunk_pairs: list[tuple[Path, Path]],
    provider: OhlcvProvider,
    cfg: argparse.Namespace,
    l2: float,
    transform_batch: callable | None = None,
    progress_desc: str,
) -> tuple[np.ndarray, float, int]:
    xtx: np.ndarray | None = None
    xty: np.ndarray | None = None
    n_train_samples = 0

    pbar = make_progress_bar(total=len(chunk_pairs), desc=progress_desc, unit="chunk")
    try:
        for meta_path, emb_path in chunk_pairs:
            meta = pd.read_parquet(meta_path, columns=["symbol", "date", "link"])
            emb = np.load(emb_path, mmap_mode=None)
            if len(meta) != len(emb):
                raise ValueError(f"Row mismatch between {meta_path} and {emb_path}: {len(meta)} vs {len(emb)}")

            symbols = meta["symbol"].fillna("").astype(str).str.upper().to_numpy()
            ts = parse_news_timestamps_utc(meta["date"], meta["link"])
            ts_ns = ts.view("int64")

            valid_idx: list[int] = []
            valid_targets: list[float] = []
            for i in range(len(meta)):
                symbol = symbols[i]
                if not symbol:
                    continue
                ts_i = int(ts_ns[i])
                if ts_i <= 0:
                    continue
                sym_idx = provider.get(symbol)
                if sym_idx is None:
                    continue
                assigned = sym_idx.assign_news_to_trade_date(ts_i, cfg)
                if assigned is None:
                    continue
                _trade_date, target, split = assigned
                if split != "train":
                    continue
                valid_idx.append(i)
                valid_targets.append(float(target))

            if not valid_idx:
                pbar.update(1)
                continue

            x_train = _fit_to_batch(emb[np.asarray(valid_idx, dtype=np.int64)], transform_batch)
            finite_mask = finite_row_mask(x_train)
            if not finite_mask.all():
                x_train = x_train[finite_mask]
                y_train = np.asarray(
                    [target for target, keep in zip(valid_targets, finite_mask.tolist(), strict=True) if keep],
                    dtype=np.float64,
                )
            else:
                y_train = np.asarray(valid_targets, dtype=np.float64)
            if x_train.size == 0:
                pbar.update(1)
                continue
            x_train = x_train.astype(np.float64, copy=False)
            ones = np.ones((x_train.shape[0], 1), dtype=np.float64)
            x_aug = np.concatenate([x_train, ones], axis=1)

            if xtx is None:
                xtx = x_aug.T @ x_aug
                xty = x_aug.T @ y_train
            else:
                xtx += x_aug.T @ x_aug
                xty += x_aug.T @ y_train
            n_train_samples += int(x_train.shape[0])
            pbar.update(1)
    finally:
        pbar.close()

    if xtx is None or xty is None or n_train_samples == 0:
        raise RuntimeError("No article-level training samples were accumulated.")
    w, b = fit_linear_closed_form_from_stats(xtx, xty, l2=l2)
    return w, b, n_train_samples


class SumHeadLinearModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(input_dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, feat_sum: torch.Tensor, news_count: torch.Tensor) -> torch.Tensor:
        return feat_sum @ self.weight + news_count * self.bias


def fit_sum_head_adam(
    *,
    x_train_sum: np.ndarray,
    news_count_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    progress_desc: str,
) -> tuple[np.ndarray, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SumHeadLinearModel(input_dim=int(x_train_sum.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    x_t = torch.from_numpy(x_train_sum.astype(np.float32, copy=False))
    count_t = torch.from_numpy(news_count_train.astype(np.float32, copy=False))
    y_t = torch.from_numpy(y_train.astype(np.float32, copy=False))
    ds = TensorDataset(x_t, count_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)

    model.train()
    epoch_iter = tqdm(range(epochs), desc=progress_desc, leave=False) if tqdm is not None else range(epochs)
    for _ in epoch_iter:
        running_loss = 0.0
        seen = 0
        for xb, cb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, cb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_n = int(yb.shape[0])
            running_loss += float(loss.detach().item()) * batch_n
            seen += batch_n
        if tqdm is not None and seen > 0:
            epoch_iter.set_postfix(loss=f"{running_loss / seen:.6f}")
    model.eval()

    w = model.weight.detach().cpu().numpy().astype(np.float32, copy=False)
    b = float(model.bias.detach().cpu().item())
    return w, b


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def fit_autoencoder(
    train_news: np.ndarray,
    *,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> AutoEncoder:
    model = AutoEncoder(
        input_dim=train_news.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    x_t = torch.from_numpy(train_news.astype(np.float32, copy=False))
    ds = TensorDataset(x_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)

    model.train()
    epoch_iter = tqdm(range(epochs), desc="AE epochs", leave=False) if tqdm is not None else range(epochs)
    for _ in epoch_iter:
        for (xb,) in dl:
            xb = xb.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_linear_closed_form(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return (x.astype(np.float32, copy=False) @ w.astype(np.float32, copy=False) + np.float32(b)).astype(
        np.float32, copy=False
    )


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    sx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    sy = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    xstd = sx.std(ddof=1)
    ystd = sy.std(ddof=1)
    if xstd == 0 or ystd == 0:
        return np.nan
    return float(np.corrcoef(sx, sy)[0, 1])


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return np.nan
    x = x[valid]
    y = y[valid]
    xstd = x.std(ddof=1)
    ystd = y.std(ddof=1)
    if xstd == 0 or ystd == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def correlation_summary(
    pred_df: pd.DataFrame,
    min_universe: int,
    corr_fn: callable,
    *,
    progress_desc: str | None = None,
) -> dict[str, float | int]:
    ic_vals: list[float] = []
    grouped = list(pred_df.groupby("trade_date", sort=True))
    pbar = make_progress_bar(total=len(grouped), desc=progress_desc or "Correlation summary", unit="day")
    try:
        for _, grp in grouped:
            if len(grp) < min_universe:
                pbar.update(1)
                continue
            ic = corr_fn(
                grp["pred"].to_numpy(dtype=np.float64),
                grp["target_return"].to_numpy(dtype=np.float64),
            )
            if np.isfinite(ic):
                ic_vals.append(float(ic))
            pbar.update(1)
    finally:
        pbar.close()
    arr = np.asarray(ic_vals, dtype=np.float64)
    if arr.size == 0:
        return {
            "days": 0,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
        }
    std = float(arr.std(ddof=1)) if arr.size > 1 else np.nan
    return {
        "days": int(arr.size),
        "ic_mean": float(arr.mean()),
        "ic_std": std,
        "ic_ir": float(arr.mean() / std) if np.isfinite(std) and std > 0 else np.nan,
    }


def rank_ic_summary(pred_df: pd.DataFrame, min_universe: int) -> dict[str, float | int]:
    return correlation_summary(pred_df, min_universe, spearman_corr, progress_desc="Rank IC")


def pearson_ic_summary(pred_df: pd.DataFrame, min_universe: int) -> dict[str, float | int]:
    return correlation_summary(pred_df, min_universe, pearson_corr, progress_desc="Pearson IC")


def build_daily_target_universe_by_split(
    provider: OhlcvProvider,
    cfg: argparse.Namespace,
) -> dict[str, dict[int, list[tuple[str, float]]]]:
    out: dict[str, dict[int, list[tuple[str, float]]]] = {"train": {}, "val": {}}
    seen_paths: set[Path] = set()
    unique_paths = sorted(set(provider.file_map.values()))
    pbar = make_progress_bar(total=len(unique_paths), desc="Daily target universe", unit="symbol")
    try:
        for path in unique_paths:
            if path in seen_paths:
                pbar.update(1)
                continue
            seen_paths.add(path)
            stem = path.stem.upper()
            base = stem.rsplit(".", 1)[0] if "." in stem else stem
            canonical_symbol = f"{base}.US"
            sym_idx = provider.get(canonical_symbol) or provider.get(stem)
            if sym_idx is None:
                pbar.update(1)
                continue
            for date_int, target in sym_idx.target_by_date.items():
                if not np.isfinite(target):
                    continue
                split = split_of_date(int(date_int), cfg)
                if split is None:
                    continue
                out[split].setdefault(int(date_int), []).append((canonical_symbol, float(target)))
            pbar.update(1)
    finally:
        pbar.close()
    return out


def rank_ic_summary_with_no_news_zero(
    pred_df: pd.DataFrame,
    daily_target_universe: dict[int, list[tuple[str, float]]],
    min_universe: int,
) -> dict[str, float | int]:
    pred_map_by_date: dict[int, dict[str, float]] = {}
    for date_int, grp in pred_df.groupby("trade_date", sort=False):
        pred_map_by_date[int(date_int)] = {
            str(sym): float(pred)
            for sym, pred in zip(grp["symbol"].tolist(), grp["pred"].tolist(), strict=True)
        }

    ic_vals: list[float] = []
    sorted_dates = sorted(daily_target_universe.keys())
    pbar = make_progress_bar(total=len(sorted_dates), desc="Rank IC no-news-zero", unit="day")
    try:
        for date_int in sorted_dates:
            items = daily_target_universe[date_int]
            if len(items) < int(min_universe):
                pbar.update(1)
                continue
            pred_map = pred_map_by_date.get(int(date_int), {})
            x = np.asarray([float(pred_map.get(sym, 0.0)) for sym, _ in items], dtype=np.float64)
            y = np.asarray([float(ret) for _, ret in items], dtype=np.float64)
            valid = np.isfinite(x) & np.isfinite(y)
            if int(valid.sum()) < int(min_universe):
                pbar.update(1)
                continue
            x = x[valid]
            y = y[valid]
            ic = spearman_corr(x, y)
            if np.isfinite(ic):
                ic_vals.append(float(ic))
            pbar.update(1)
    finally:
        pbar.close()

    arr = np.asarray(ic_vals, dtype=np.float64)
    if arr.size == 0:
        return {
            "days": 0,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
        }
    std = float(arr.std(ddof=1)) if arr.size > 1 else np.nan
    return {
        "days": int(arr.size),
        "ic_mean": float(arr.mean()),
        "ic_std": std,
        "ic_ir": float(arr.mean() / std) if np.isfinite(std) and std > 0 else np.nan,
    }


def pearson_ic_summary_with_no_news_zero(
    pred_df: pd.DataFrame,
    daily_target_universe: dict[int, list[tuple[str, float]]],
    min_universe: int,
) -> dict[str, float | int]:
    pred_map_by_date: dict[int, dict[str, float]] = {}
    for date_int, grp in pred_df.groupby("trade_date", sort=False):
        pred_map_by_date[int(date_int)] = {
            str(sym): float(pred)
            for sym, pred in zip(grp["symbol"].tolist(), grp["pred"].tolist(), strict=True)
        }

    ic_vals: list[float] = []
    sorted_dates = sorted(daily_target_universe.keys())
    pbar = make_progress_bar(total=len(sorted_dates), desc="Pearson IC no-news-zero", unit="day")
    try:
        for date_int in sorted_dates:
            items = daily_target_universe[date_int]
            if len(items) < int(min_universe):
                pbar.update(1)
                continue
            pred_map = pred_map_by_date.get(int(date_int), {})
            x = np.asarray([float(pred_map.get(sym, 0.0)) for sym, _ in items], dtype=np.float64)
            y = np.asarray([float(ret) for _, ret in items], dtype=np.float64)
            valid = np.isfinite(x) & np.isfinite(y)
            if int(valid.sum()) < int(min_universe):
                pbar.update(1)
                continue
            x = x[valid]
            y = y[valid]
            ic = pearson_corr(x, y)
            if np.isfinite(ic):
                ic_vals.append(float(ic))
            pbar.update(1)
    finally:
        pbar.close()

    arr = np.asarray(ic_vals, dtype=np.float64)
    if arr.size == 0:
        return {
            "days": 0,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
        }
    std = float(arr.std(ddof=1)) if arr.size > 1 else np.nan
    return {
        "days": int(arr.size),
        "ic_mean": float(arr.mean()),
        "ic_std": std,
        "ic_ir": float(arr.mean() / std) if np.isfinite(std) and std > 0 else np.nan,
    }


def build_model_artifact_payload(
    *,
    w: np.ndarray,
    b: float,
    input_dim: int,
    linear_train_mode: str,
    inference_mode: str,
    solver: str,
    l2: float | None = None,
    requires_encoder: bool = False,
    extra: dict | None = None,
) -> dict:
    payload = {
        "weight": w,
        "bias": b,
        "input_dim": int(input_dim),
        "target_return": "open_to_open",
        "linear_train_mode": linear_train_mode,
        "inference_mode": inference_mode,
        "solver": solver,
        "requires_encoder": bool(requires_encoder),
    }
    if l2 is not None:
        payload["l2"] = float(l2)
    if extra:
        payload.update(extra)
    return payload


def train_and_score_linear_head(
    *,
    head_name: str,
    group_df: pd.DataFrame,
    feat_sum: np.ndarray,
    chunk_pairs: list[tuple[Path, Path]],
    provider: OhlcvProvider,
    cfg: argparse.Namespace,
    daily_target_universe_by_split: dict[str, dict[int, list[tuple[str, float]]]],
    transform_batch: callable | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    train_mode = str(cfg.linear_train_mode)
    inference_mode = "mean" if train_mode in {"mean", "article"} else "sum_head"

    if train_mode == "mean":
        x_group = build_group_feature_matrix(group_df, feat_sum, "mean")
        x_train, y_train, idx_train = split_xy(group_df, x_group, "train")
        print(f"[{head_name}] fitting closed-form ridge on {len(x_train)} grouped-train samples...")
        w, b = fit_linear_closed_form(x_train=x_train, y_train=y_train, l2=cfg.linear_l2)
        print(f"[{head_name}] fit done.")
        fit_sample_count = int(len(x_train))
        solver = "closed_form_ridge"
        solver_extra = {"l2": float(cfg.linear_l2)}
    elif train_mode == "article":
        print(f"[{head_name}] fitting article-level closed-form ridge...")
        w, b, fit_sample_count = fit_linear_article_closed_form_streaming(
            chunk_pairs=chunk_pairs,
            provider=provider,
            cfg=cfg,
            l2=cfg.linear_l2,
            transform_batch=transform_batch,
            progress_desc=f"{head_name} article ridge",
        )
        print(f"[{head_name}] fit done on {fit_sample_count} article-level train samples.")
        solver = "closed_form_ridge"
        solver_extra = {"l2": float(cfg.linear_l2)}
    elif train_mode == "sum_head":
        x_train_sum, y_train, idx_train = split_xy(group_df, feat_sum, "train")
        news_count_train = group_df.loc[group_df["split"].eq("train"), "news_count"].to_numpy(dtype=np.float32)
        print(f"[{head_name}] fitting sum(head(embedding)) model with Adam on {len(x_train_sum)} grouped-train samples...")
        w, b = fit_sum_head_adam(
            x_train_sum=x_train_sum,
            news_count_train=news_count_train,
            y_train=y_train,
            epochs=cfg.adam_epochs,
            lr=cfg.adam_lr,
            batch_size=cfg.adam_batch_size,
            weight_decay=cfg.adam_weight_decay,
            progress_desc=f"{head_name} sum_head Adam",
        )
        print(f"[{head_name}] fit done.")
        fit_sample_count = int(len(x_train_sum))
        solver = "adam_sum_head"
        solver_extra = {
            "adam_epochs": int(cfg.adam_epochs),
            "adam_lr": float(cfg.adam_lr),
            "adam_batch_size": int(cfg.adam_batch_size),
            "adam_weight_decay": float(cfg.adam_weight_decay),
        }
    else:
        raise ValueError(f"Unsupported linear_train_mode: {train_mode}")

    pred_all = predict_group_scores(group_df, feat_sum, w, b, inference_mode=inference_mode)
    train_mask = group_df["split"].eq("train").to_numpy()
    val_mask = group_df["split"].eq("val").to_numpy()

    pred_df_train = group_df.loc[train_mask].copy()
    pred_df_train["pred"] = pred_all[train_mask]
    pred_df_train["model"] = head_name

    pred_df_val = group_df.loc[val_mask].copy()
    pred_df_val["pred"] = pred_all[val_mask]
    pred_df_val["model"] = head_name

    metrics = {
        "linear_train_mode": train_mode,
        "inference_mode": inference_mode,
        "train_rank_ic": rank_ic_summary(pred_df_train, cfg.min_ic_universe),
        "val_rank_ic": rank_ic_summary(pred_df_val, cfg.min_ic_universe),
        "train_pearson_ic": pearson_ic_summary(pred_df_train, cfg.min_ic_universe),
        "val_pearson_ic": pearson_ic_summary(pred_df_val, cfg.min_ic_universe),
        "train_rank_ic_with_no_news_zero": rank_ic_summary_with_no_news_zero(
            pred_df_train,
            daily_target_universe_by_split["train"],
            cfg.min_ic_universe,
        ),
        "val_rank_ic_with_no_news_zero": rank_ic_summary_with_no_news_zero(
            pred_df_val,
            daily_target_universe_by_split["val"],
            cfg.min_ic_universe,
        ),
        "train_pearson_ic_with_no_news_zero": pearson_ic_summary_with_no_news_zero(
            pred_df_train,
            daily_target_universe_by_split["train"],
            cfg.min_ic_universe,
        ),
        "val_pearson_ic_with_no_news_zero": pearson_ic_summary_with_no_news_zero(
            pred_df_val,
            daily_target_universe_by_split["val"],
            cfg.min_ic_universe,
        ),
        "n_train_samples": int(len(pred_df_train)),
        "n_val_samples": int(len(pred_df_val)),
        "n_train_fit_samples": int(fit_sample_count),
    }
    metrics.update(solver_extra)

    artifact_payload = build_model_artifact_payload(
        w=w,
        b=b,
        input_dim=int(feat_sum.shape[1]),
        linear_train_mode=train_mode,
        inference_mode=inference_mode,
        solver=solver,
        l2=float(cfg.linear_l2) if solver == "closed_form_ridge" else None,
        requires_encoder=(head_name == "ae_linear"),
        extra=solver_extra if solver != "closed_form_ridge" else None,
    )

    pred_df = pd.concat([pred_df_train, pred_df_val], ignore_index=True)
    return pred_df, metrics, artifact_payload


def split_xy(group_df: pd.DataFrame, feat_mat: np.ndarray, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = group_df["split"].eq(split).to_numpy()
    x = feat_mat[mask]
    y = group_df.loc[mask, "target_return"].to_numpy(dtype=np.float32)
    idx = np.flatnonzero(mask)
    return x, y, idx


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    news_emb_dir = Path(cfg.news_emb_dir).expanduser().resolve()
    ohlcv_dir = Path(cfg.ohlcv_dir).expanduser().resolve()
    out_dir = Path(cfg.output_dir).expanduser().resolve()
    cache_dir = Path(cfg.cache_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_enabled = not bool(cfg.disable_cache)
    if cache_enabled:
        cache_dir.mkdir(parents=True, exist_ok=True)

    chunk_pairs = discover_chunk_pairs(news_emb_dir)
    provider = OhlcvProvider(ohlcv_dir=ohlcv_dir)
    daily_target_universe_by_split = load_or_build_daily_target_universe_cached(
        provider=provider,
        cfg=cfg,
        cache_dir=cache_dir,
        enabled=cache_enabled,
    )

    need_ae = "ae_linear" in cfg.heads
    group_df, x_raw, ae_train_news = load_or_build_raw_group_features_cached(
        news_emb_dir=news_emb_dir,
        ohlcv_dir=ohlcv_dir,
        chunk_pairs=chunk_pairs,
        provider=provider,
        cfg=cfg,
        collect_train_news=need_ae,
        cache_dir=cache_dir,
        enabled=cache_enabled,
    )

    predictions_all: list[pd.DataFrame] = []
    metrics: dict[str, dict] = {}

    if "linear" in cfg.heads:
        pred_df_linear, metrics["linear"], linear_payload = train_and_score_linear_head(
            head_name="linear",
            group_df=group_df,
            feat_sum=x_raw,
            chunk_pairs=chunk_pairs,
            provider=provider,
            cfg=cfg,
            daily_target_universe_by_split=daily_target_universe_by_split,
            transform_batch=None,
        )
        predictions_all.append(pred_df_linear)
        torch.save(linear_payload, out_dir / "linear_head.pt")

    if "ae_linear" in cfg.heads:
        if ae_train_news is None or len(ae_train_news) == 0:
            raise RuntimeError("No train news rows were collected for AE training.")
        ae_model = fit_autoencoder(
            train_news=ae_train_news,
            hidden_dim=cfg.ae_hidden_dim,
            latent_dim=cfg.ae_latent_dim,
            epochs=cfg.ae_epochs,
            lr=cfg.ae_lr,
            batch_size=cfg.ae_batch_size,
        )
        for p in ae_model.encoder.parameters():
            p.requires_grad = False
        ae_model.encoder.eval()

        @torch.inference_mode()
        def encode_batch(arr: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(arr.astype(np.float32, copy=False)).cuda(non_blocking=True)
            z = ae_model.encoder(x).detach().cpu().numpy().astype(np.float32, copy=False)
            return z

        @torch.inference_mode()
        def encode_matrix(arr: np.ndarray, batch_size: int) -> np.ndarray:
            outs: list[np.ndarray] = []
            total_batches = (len(arr) + batch_size - 1) // batch_size if batch_size > 0 else 0
            pbar = make_progress_bar(total=total_batches, desc="Encoding aggregated matrix", unit="batch")
            try:
                for start in range(0, len(arr), batch_size):
                    end = min(start + batch_size, len(arr))
                    xb = torch.from_numpy(arr[start:end].astype(np.float32, copy=False)).cuda(non_blocking=True)
                    zb = ae_model.encoder(xb).detach().cpu().numpy().astype(np.float32, copy=False)
                    outs.append(zb)
                    pbar.update(1)
            finally:
                pbar.close()
            return np.concatenate(outs, axis=0) if outs else np.empty((0, cfg.ae_latent_dim), dtype=np.float32)

        try:
            group_df_z, x_z, _ = aggregate_group_features(
                chunk_pairs=chunk_pairs,
                provider=provider,
                cfg=cfg,
                transform_batch=encode_batch,
                collect_train_news=False,
                progress_desc="Aggregating encoded news chunks",
            )
        except RuntimeError as exc:
            if "No grouped samples created" not in str(exc):
                raise
            print(
                "[ae_linear][warn] Encoded second-pass aggregation returned no samples. "
                "Falling back to encoding aggregated raw vectors."
            )
            group_df_z = group_df.copy()
            x_z = encode_matrix(x_raw, batch_size=cfg.ae_batch_size)

        pred_df_ae, metrics["ae_linear"], ae_linear_payload = train_and_score_linear_head(
            head_name="ae_linear",
            group_df=group_df_z,
            feat_sum=x_z,
            chunk_pairs=chunk_pairs,
            provider=provider,
            cfg=cfg,
            daily_target_universe_by_split=daily_target_universe_by_split,
            transform_batch=encode_batch,
        )
        metrics["ae_linear"]["ae_train_rows"] = int(len(ae_train_news))
        predictions_all.append(pred_df_ae)

        torch.save(
            {
                "ae_state_dict": ae_model.state_dict(),
                "encoder_state_dict": ae_model.encoder.state_dict(),
                "input_dim": int(ae_train_news.shape[1]),
                "hidden_dim": int(cfg.ae_hidden_dim),
                "latent_dim": int(cfg.ae_latent_dim),
            },
            out_dir / "ae_encoder.pt",
        )
        torch.save(ae_linear_payload, out_dir / "ae_linear_head.pt")

    if not predictions_all:
        raise RuntimeError("No heads were trained. Check --heads.")

    pred_all_df = pd.concat(predictions_all, ignore_index=True).sort_values(["model", "trade_date", "symbol"]).reset_index(
        drop=True
    )
    pred_all_df.to_parquet(out_dir / "predictions.parquet", index=False)

    payload = {
        "config": vars(cfg),
        "timezone_assumption": {
            "news_date_field": "ISO8601 from EODHD News API, parsed as UTC",
            "signal_cutoff": "09:00 America/New_York",
            "window_definition": (
                "(previous_trade_day_close_NY, current_trade_day_09:00_NY]"
                if bool(cfg.overnight_news)
                else "(previous_trade_day_09:00_NY, current_trade_day_09:00_NY]"
            ),
            "target_return": "open_to_open",
            "midnight_timestamp_shift_domains": list(MIDNIGHT_SHIFT_DOMAINS),
        },
        "metrics": metrics,
        "artifacts": {
            "predictions": str(out_dir / "predictions.parquet"),
            "linear_head": str(out_dir / "linear_head.pt") if "linear" in cfg.heads else None,
            "ae_encoder": str(out_dir / "ae_encoder.pt") if "ae_linear" in cfg.heads else None,
            "ae_linear_head": str(out_dir / "ae_linear_head.pt") if "ae_linear" in cfg.heads else None,
        },
    }
    save_json(out_dir / "metrics.json", payload)
    print(json.dumps(payload["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
