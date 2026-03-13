from __future__ import annotations

import numpy as np
import pandas as pd


def run_weight_execution_engine(
    ret_d: pd.DataFrame,
    cal: pd.DataFrame,
    symbols: pd.Index,
    rebalance_mask: pd.Series,
    target_by_date: dict[pd.Timestamp, pd.Series],
    meta_by_date: dict[pd.Timestamp, dict],
    elig_by_date: dict[pd.Timestamp, pd.Series],
    *,
    cost_bps: float,
    vol_target_ann: float = 0.0,
    vol_lookback_m: int = 12,
    max_leverage: float = 1.0,
    period_col: str = "period_end",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_dates = pd.DatetimeIndex(ret_d.index)
    w_prev_close = pd.Series(0.0, index=symbols, dtype="float64")
    w_target_last = pd.Series(0.0, index=symbols, dtype="float64")
    cost_rate = float(cost_bps) / 10000.0
    equity = 1.0
    daily_rows: list[dict] = []
    rb_rows: list[dict] = []
    w_rows: list[dict] = []
    ret_hist: list[float] = []

    for dt in trade_dates:
        dt = pd.Timestamp(dt)
        period_end = pd.Timestamp(cal.at[dt, period_col])
        do_rb = bool(rebalance_mask.loc[dt])
        signal_period_end = pd.NaT
        n_pb_pool = 0
        n_force_cash = 0
        n_held_before_rebalance = 0
        n_eligible = 0
        vol_est_ann = np.nan
        vol_scale = 1.0

        if do_rb:
            held = w_prev_close.abs() > 0
            n_held_before_rebalance = int(held.sum())
            meta = meta_by_date.get(dt, {})
            signal_period_end = meta.get("signal_period_end", pd.NaT)
            n_eligible = int(meta.get("n_eligible", 0))
            n_pb_pool = int(meta.get("n_pb_pool", 0))

            elig = elig_by_date.get(dt)
            if elig is not None:
                force_cash = held & (~elig.astype(bool))
                n_force_cash = int(force_cash.sum())
                if n_force_cash > 0:
                    w_prev_close.loc[force_cash] = 0.0

            w_target_last = target_by_date.get(dt, pd.Series(0.0, index=symbols, dtype="float64")).astype("float64")
            if float(vol_target_ann) > 0 and float(w_target_last.sum()) > 0:
                lb_days = max(20, int(vol_lookback_m) * 21)
                hist = np.asarray(ret_hist[-lb_days:], dtype="float64")
                if hist.size >= 20:
                    vol_d = float(np.nanstd(hist, ddof=1))
                    vol_est_ann = float(vol_d * np.sqrt(252.0)) if np.isfinite(vol_d) else np.nan
                    if np.isfinite(vol_est_ann) and vol_est_ann > 0:
                        vol_scale = float(vol_target_ann) / vol_est_ann
                if not np.isfinite(vol_scale) or vol_scale <= 0:
                    vol_scale = 1.0
                vol_scale = float(min(vol_scale, float(max_leverage)))
                w_target_last = w_target_last * vol_scale

        if do_rb:
            cash_before = 1.0 - float(w_prev_close.sum())
            cash_target = 1.0 - float(w_target_last.sum())
            turnover = 0.5 * (float((w_target_last - w_prev_close).abs().sum()) + abs(cash_target - cash_before))
            cost = turnover * cost_rate
            w_exec = w_target_last.copy()
            rb_rows.append(
                {
                    "trade_date": pd.Timestamp(dt),
                    "period_end": period_end,
                    "signal_period_end": signal_period_end,
                    "turnover": float(turnover),
                    "cost": float(cost),
                    "n_holdings": int((w_target_last > 0).sum()),
                    "n_held_before_rebalance": int(n_held_before_rebalance),
                    "n_eligible": int(n_eligible),
                    "n_pb_pool": int(n_pb_pool),
                    "n_force_cash": int(n_force_cash),
                    "force_cash_ratio_in_held": (
                        float(n_force_cash / n_held_before_rebalance) if n_held_before_rebalance > 0 else np.nan
                    ),
                    "leverage": float(w_target_last.sum()),
                    "vol_est_ann": float(vol_est_ann),
                    "vol_scale": float(vol_scale),
                }
            )
        else:
            turnover = 0.0
            cost = 0.0
            w_exec = w_prev_close.copy()

        r = ret_d.loc[dt].fillna(0.0).astype("float64")
        gross = float((w_exec * r).sum())
        net = float(gross - cost)
        equity = float(equity * (1.0 + net))
        ret_hist.append(float(net))

        value_after = 1.0 + gross
        if value_after > 0 and np.isfinite(value_after):
            w_prev_close = (w_exec * (1.0 + r)) / value_after
        else:
            w_prev_close = pd.Series(0.0, index=symbols, dtype="float64")

        daily_rows.append(
            {
                "trade_date": pd.Timestamp(dt),
                "period_end": period_end,
                "signal_period_end": signal_period_end,
                "rebalance": bool(do_rb),
                "gross_ret": float(gross),
                "net_ret": float(net),
                "turnover": float(turnover),
                "cost": float(cost),
                "equity": float(equity),
                "leverage": float(w_exec.sum()),
            }
        )
        w_rows.append({"trade_date": pd.Timestamp(dt), **{s: float(v) for s, v in w_exec.items()}})

    rebalance_info = pd.DataFrame(rb_rows).sort_values("trade_date").reset_index(drop=True)
    daily_bt = pd.DataFrame(daily_rows).sort_values("trade_date").reset_index(drop=True)
    w_daily = pd.DataFrame(w_rows).set_index("trade_date").sort_index() if w_rows else pd.DataFrame()
    return rebalance_info, daily_bt, w_daily


def performance_stats(ret: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    r = pd.to_numeric(ret, errors="coerce").fillna(0.0).astype("float64")
    n = len(r)
    if n == 0:
        return {"annualized_return": np.nan, "annualized_vol": np.nan, "sharpe": np.nan, "total_return": np.nan}
    eq = (1.0 + r).cumprod()
    years = n / float(periods_per_year)
    ann_ret = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else np.nan
    sharpe = float((r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year)) if n > 1 and r.std(ddof=1) != 0 else np.nan
    dd = (eq / eq.cummax() - 1.0).astype("float64")
    max_dd = float(dd.min()) if len(dd) else np.nan
    return {
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "total_return": float(eq.iloc[-1] - 1.0),
        "max_drawdown": max_dd,
    }
