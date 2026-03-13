from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_plot_frame(daily_bt: pd.DataFrame) -> pd.DataFrame:
    if "trade_date" not in daily_bt.columns or "net_ret" not in daily_bt.columns:
        raise ValueError("daily_backtest data must contain trade_date and net_ret columns.")
    df = daily_bt.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["net_ret"] = pd.to_numeric(df["net_ret"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows available to plot PnL.")
    df["equity"] = (1.0 + df["net_ret"]).cumprod()
    return df


def build_baseline_frame(baseline_ohlcv_path: Path, trade_dates: pd.Series) -> pd.DataFrame:
    cols = ["date", "adjusted_close", "close"]
    df = pd.read_parquet(Path(baseline_ohlcv_path).expanduser().resolve(), columns=cols)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows available to plot baseline.")

    px = df["adjusted_close"].where(df["adjusted_close"] > 0, df["close"])
    px = pd.to_numeric(px, errors="coerce")
    df["baseline_ret"] = px.pct_change().fillna(0.0)
    df["baseline_equity"] = (1.0 + df["baseline_ret"]).cumprod()
    baseline_df = df[["date", "baseline_equity"]].copy()

    plot_dates = pd.DataFrame({"trade_date": pd.to_datetime(trade_dates, errors="coerce")}).dropna()
    baseline_df = plot_dates.merge(baseline_df, left_on="trade_date", right_on="date", how="left").drop(columns=["date"])
    baseline_df["baseline_equity"] = baseline_df["baseline_equity"].ffill()
    first_valid = baseline_df["baseline_equity"].first_valid_index()
    if first_valid is None:
        raise ValueError("Baseline has no overlap with backtest dates.")
    baseline_df = baseline_df.loc[first_valid:].copy()
    baseline_df["baseline_equity"] = baseline_df["baseline_equity"] / float(baseline_df["baseline_equity"].iloc[0])
    baseline_df["trade_date"] = pd.to_datetime(baseline_df["trade_date"], errors="coerce")
    return baseline_df


def build_baseline_frame_from_daily(baseline_daily_path: Path, trade_dates: pd.Series) -> pd.DataFrame:
    df = pd.read_parquet(Path(baseline_daily_path).expanduser().resolve())
    if "trade_date" not in df.columns:
        raise ValueError("Baseline daily parquet must contain trade_date.")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows available to plot baseline daily data.")

    if "equity" in df.columns:
        df["baseline_equity"] = pd.to_numeric(df["equity"], errors="coerce")
    elif "net_ret" in df.columns:
        ret = pd.to_numeric(df["net_ret"], errors="coerce").fillna(0.0)
        df["baseline_equity"] = (1.0 + ret).cumprod()
    else:
        raise ValueError("Baseline daily parquet must contain either equity or net_ret.")

    baseline_df = df[["trade_date", "baseline_equity"]].copy()
    plot_dates = pd.DataFrame({"trade_date": pd.to_datetime(trade_dates, errors="coerce")}).dropna()
    baseline_df = plot_dates.merge(baseline_df, on="trade_date", how="left")
    baseline_df["baseline_equity"] = baseline_df["baseline_equity"].ffill()
    first_valid = baseline_df["baseline_equity"].first_valid_index()
    if first_valid is None:
        raise ValueError("Baseline daily data has no overlap with backtest dates.")
    baseline_df = baseline_df.loc[first_valid:].copy()
    baseline_df["baseline_equity"] = baseline_df["baseline_equity"] / float(baseline_df["baseline_equity"].iloc[0])
    return baseline_df


def save_pnl_plot(
    daily_bt: pd.DataFrame,
    output_path: Path,
    title: str = "Backtest PnL",
    baseline_ohlcv_path: str | Path | None = None,
    baseline_daily_path: str | Path | None = None,
    baseline_label: str = "Baseline",
    secondary_baseline_ohlcv_path: str | Path | None = None,
    secondary_baseline_daily_path: str | Path | None = None,
    secondary_baseline_label: str = "Secondary Baseline",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = build_plot_frame(daily_bt)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["trade_date"], df["equity"], label="Strategy", linewidth=2.0, color="#1f77b4")
    baseline_df = None
    if baseline_daily_path:
        baseline_df = build_baseline_frame_from_daily(Path(baseline_daily_path), df["trade_date"])
    elif baseline_ohlcv_path:
        baseline_df = build_baseline_frame(Path(baseline_ohlcv_path), df["trade_date"])
    if baseline_df is not None:
        ax.plot(
            baseline_df["trade_date"],
            baseline_df["baseline_equity"],
            label=baseline_label,
            linewidth=1.8,
            color="#ff7f0e",
            linestyle="--",
        )
    secondary_baseline_df = None
    if secondary_baseline_daily_path:
        secondary_baseline_df = build_baseline_frame_from_daily(Path(secondary_baseline_daily_path), df["trade_date"])
    elif secondary_baseline_ohlcv_path:
        secondary_baseline_df = build_baseline_frame(Path(secondary_baseline_ohlcv_path), df["trade_date"])
    if secondary_baseline_df is not None:
        ax.plot(
            secondary_baseline_df["trade_date"],
            secondary_baseline_df["baseline_equity"],
            label=secondary_baseline_label,
            linewidth=1.6,
            color="#2ca02c",
            linestyle=":",
        )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PnL from daily_backtest.parquet.")
    parser.add_argument("--daily-backtest", required=True, help="Path to daily_backtest.parquet.")
    parser.add_argument("--output-path", required=True, help="Path to output PNG file.")
    parser.add_argument("--title", default="Backtest PnL", help="Plot title.")
    parser.add_argument("--baseline-ohlcv-path", default=None, help="Optional path to baseline OHLCV parquet.")
    parser.add_argument("--baseline-daily-path", default=None, help="Optional path to baseline daily parquet.")
    parser.add_argument("--baseline-label", default="Baseline", help="Legend label for the optional baseline.")
    parser.add_argument("--secondary-baseline-ohlcv-path", default=None, help="Optional path to a second baseline OHLCV parquet.")
    parser.add_argument("--secondary-baseline-daily-path", default=None, help="Optional path to a second baseline daily parquet.")
    parser.add_argument("--secondary-baseline-label", default="Secondary Baseline", help="Legend label for the second optional baseline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily_bt = pd.read_parquet(Path(args.daily_backtest).expanduser().resolve())
    out = save_pnl_plot(
        daily_bt,
        Path(args.output_path),
        title=str(args.title),
        baseline_ohlcv_path=args.baseline_ohlcv_path,
        baseline_daily_path=args.baseline_daily_path,
        baseline_label=str(args.baseline_label),
        secondary_baseline_ohlcv_path=args.secondary_baseline_ohlcv_path,
        secondary_baseline_daily_path=args.secondary_baseline_daily_path,
        secondary_baseline_label=str(args.secondary_baseline_label),
    )
    print(str(out))


if __name__ == "__main__":
    main()
