"""Simulation, Linear Regression Trend Extrapolation & Goal-Seeking Engine for Data_Talk.

Provides deterministic mathematical modeling across any domain dataset:
1. OLS Linear Regression for organic baseline trend extrapolation with R^2 and prediction intervals (P10/P50/P90).
2. Multi-lever scenario simulation with price elasticity of demand and non-additive ratio protection.
3. Standalone Brent's / bisection root-finding for Goal-Seek inversion.
4. Waterfall variance decomposition (step-by-step driver attribution).
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.connectors.base import BaseConnector
    from src.knowledge.okf_models import OKFWorkspace, TableSchema

_LOGGER = logging.getLogger(__name__)


class SimulationEngine:
    """Core mathematical engine for Future & What-If Analysis in Data_Talk."""

    @classmethod
    def fit_linear_trend_regression(
        cls,
        df: pd.DataFrame,
        date_col: str,
        metric_col: str,
        forecast_periods: int = 4,
        shock_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        """Fit OLS linear regression y = beta_0 + beta_1 * t over historical time series.

        Returns slope, intercept, R^2, standard error, historical trend points,
        and forward projected points with P10/P50/P90 prediction intervals.
        """
        if df.empty or metric_col not in df.columns or date_col not in df.columns:
            return {
                "success": False,
                "error": "DataFrame is empty or missing required columns.",
                "slope": 0.0,
                "intercept": 0.0,
                "r2": 0.0,
                "historical_points": [],
                "forecast_points": [],
            }

        clean_df = df[[date_col, metric_col]].dropna().copy()
        clean_df[metric_col] = pd.to_numeric(clean_df[metric_col], errors="coerce")
        clean_df = clean_df.dropna().sort_values(by=date_col)

        n = len(clean_df)
        if n < 2:
            return {
                "success": False,
                "error": "At least 2 historical data points are required for linear trend regression.",
                "slope": 0.0,
                "intercept": float(clean_df[metric_col].iloc[0]) if n == 1 else 0.0,
                "r2": 1.0 if n == 1 else 0.0,
                "historical_points": [],
                "forecast_points": [],
            }

        t = np.arange(n, dtype=float)
        y = clean_df[metric_col].to_numpy(dtype=float)

        # OLS Linear Regression: y = beta_0 + beta_1 * t
        t_mean = float(np.mean(t))
        y_mean = float(np.mean(y))
        s_tt = float(np.sum((t - t_mean) ** 2))

        if s_tt == 0:
            beta_1 = 0.0
            beta_0 = y_mean
            r2 = 0.0
            s_e = 0.0
        else:
            s_ty = float(np.sum((t - t_mean) * (y - y_mean)))
            beta_1 = s_ty / s_tt
            beta_0 = y_mean - (beta_1 * t_mean)

            y_pred_hist = beta_0 + (beta_1 * t)
            ss_res = float(np.sum((y - y_pred_hist) ** 2))
            ss_tot = float(np.sum((y - y_mean) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
            s_e = math.sqrt(ss_res / (n - 2)) if n > 2 else 0.0

        date_labels = [str(d) for d in clean_df[date_col]]

        # Compute Recent-Weighted Momentum (Holt's Linear / EWMA Trend) and Ensemble Blend
        # Weighting recent observations higher avoids an unnatural cliff when recent years
        # are far above or below the 20-year unweighted OLS line.
        weights = np.exp(np.linspace(-1.6, 0.0, n))
        weights /= np.sum(weights)
        t_w_mean = float(np.sum(weights * t))
        y_w_mean = float(np.sum(weights * y))
        s_w_tt = float(np.sum(weights * ((t - t_w_mean) ** 2)))
        if s_w_tt > 1e-12:
            beta_1_w = float(np.sum(weights * (t - t_w_mean) * (y - y_w_mean)) / s_w_tt)
            beta_0_w = y_w_mean - (beta_1_w * t_w_mean)
        else:
            beta_1_w = beta_1
            beta_0_w = beta_0

        # Recent 3-period level anchor for smooth trajectory continuity when R^2 is low
        recent_k = min(3, n)
        recent_level = float(np.mean(y[-recent_k:]))
        ols_end_level = float(beta_0 + beta_1 * (n - 1))

        # Blend weight: when R^2 is high, trust global OLS; when R^2 is noisy/volatile, anchor closer to recent 3-period level & weighted slope
        ols_trust = max(0.35, min(0.85, float(r2) + 0.35))
        eff_slope = (ols_trust * beta_1) + ((1.0 - ols_trust) * beta_1_w)
        anchor_end = (ols_trust * ols_end_level) + ((1.0 - ols_trust) * recent_level)

        hist_points: list[dict[str, Any]] = []
        for i in range(n):
            hist_points.append(
                {
                    "date": date_labels[i],
                    "period_index": i,
                    "actual": round(float(y[i]), 2),
                    "trend": round(float(beta_0 + beta_1 * i), 2),
                    "weighted_trend": round(float(beta_0_w + beta_1_w * i), 2),
                }
            )

        # Generate forward forecast period labels
        forecast_labels = cls._generate_future_period_labels(date_labels, forecast_periods)
        forecast_points: list[dict[str, Any]] = []
        mult = max(0.0, float(shock_multiplier))

        for h in range(1, forecast_periods + 1):
            t_fut = n - 1 + h
            y_hat_ols = beta_0 + (beta_1 * t_fut)
            y_hat_smart = anchor_end + (eff_slope * h)
            y_hat_organic = y_hat_ols
            y_hat = y_hat_organic * mult
            y_hat_smart_shocked = y_hat_smart * mult

            # Prediction interval standard error:
            # SE_pred = s_e * sqrt(1 + 1/n + (t_fut - t_mean)^2 / s_tt)
            se_pred = (
                s_e * math.sqrt(1.0 + (1.0 / n) + ((t_fut - t_mean) ** 2 / s_tt)) * mult
                if s_tt > 0 and s_e > 0
                else 0.0
            )
            # 80% confidence interval multiplier = 1.282
            p10 = max(0.0, y_hat - (1.282 * se_pred))
            p50 = y_hat
            p90 = y_hat + (1.282 * se_pred)

            forecast_points.append(
                {
                    "date": forecast_labels[h - 1],
                    "period_index": t_fut,
                    "organic_projected": round(float(y_hat_organic), 2),
                    "smart_momentum": round(float(y_hat_smart_shocked), 2),
                    "projected": round(float(p50), 2),
                    "p10": round(float(p10), 2),
                    "p50": round(float(p50), 2),
                    "p90": round(float(p90), 2),
                    "is_forecast": True,
                }
            )

        # Backtest MAPE on last min(3, n-1) points
        mape = 0.0
        if n >= 4:
            actual_tail = y[-3:]
            pred_tail = (beta_0 + beta_1 * t[-3:])
            non_zero = np.abs(actual_tail) > 1e-6
            if np.any(non_zero):
                mape = round(float(np.mean(np.abs((actual_tail[non_zero] - pred_tail[non_zero]) / actual_tail[non_zero])) * 100.0), 1)

        if r2 >= 0.50 or (mape > 0 and mape <= 12.0):
            confidence_grade = "High Predictability (Strong Linear Trend)"
        elif r2 >= 0.18 or (mape > 0 and mape <= 22.0):
            confidence_grade = "Moderate Predictability (Trend + Cyclical Variance)"
        else:
            confidence_grade = "High Historical Volatility (Use P10–P90 Range)"

        return {
            "success": True,
            "metric": metric_col,
            "slope": round(beta_1, 4),
            "weighted_slope": round(beta_1_w, 4),
            "effective_slope": round(eff_slope, 4),
            "intercept": round(beta_0, 4),
            "r2": round(r2, 4),
            "mape_pct": mape,
            "confidence_grade": confidence_grade,
            "residual_std_error": round(s_e, 4),
            "historical_mean": round(y_mean, 2),
            "recent_3p_mean": round(recent_level, 2),
            "historical_points": hist_points,
            "forecast_points": forecast_points,
            "historical_count": n,
            "forecast_count": forecast_periods,
            "shock_multiplier": round(mult, 4),
        }

    @staticmethod
    def _generate_future_period_labels(date_labels: list[str], count: int) -> list[str]:
        """Extrapolate dates or period strings (e.g. '2025', '2025-Q4', '2025-12', '2025-12-01')."""
        if not date_labels:
            return [f"Period +{i+1}" for i in range(count)]

        last = date_labels[-1].strip()

        # Match Year format: 'YYYY'
        y_match = re.match(r"^(\d{4})$", last)
        if y_match:
            year = int(y_match.group(1))
            return [str(year + i + 1) for i in range(count)]

        # Match Year-Month format: 'YYYY-MM'
        ym_match = re.match(r"^(\d{4})-(\d{1,2})$", last)
        if ym_match:
            year = int(ym_match.group(1))
            month = int(ym_match.group(2))
            out = []
            for _ in range(count):
                month += 1
                if month > 12:
                    month = 1
                    year += 1
                out.append(f"{year}-{month:02d}")
            return out

        # Match Full ISO Date format: 'YYYY-MM-DD'
        ymd_match = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", last)
        if ymd_match:
            try:
                base_ts = pd.Timestamp(last[:10])
                return [(base_ts + pd.DateOffset(months=i + 1)).strftime("%Y-%m-%d") for i in range(count)]
            except Exception:
                pass

        # Match Year-Quarter format: 'YYYY-Q1' or '2025 Q3'
        yq_match = re.match(r"^(\d{4})[- /]?Q([1-4])$", last, re.IGNORECASE)
        if yq_match:
            year = int(yq_match.group(1))
            qtr = int(yq_match.group(2))
            out = []
            for _ in range(count):
                qtr += 1
                if qtr > 4:
                    qtr = 1
                    year += 1
                out.append(f"{year}-Q{qtr}")
            return out

        # Match Academic Term format: '2025-FALL' / '2025-SPRING' / '2025-SUMMER'
        term_match = re.match(r"^(\d{4})[- _/](SPRING|SUMMER|FALL|WINTER)$", last, re.IGNORECASE)
        if term_match:
            year = int(term_match.group(1))
            seasons = ["SPRING", "SUMMER", "FALL"]
            cur_s = term_match.group(2).upper()
            idx = seasons.index(cur_s) if cur_s in seasons else 2
            out = []
            for _ in range(count):
                idx += 1
                if idx >= len(seasons):
                    idx = 0
                    year += 1
                out.append(f"{year}-{seasons[idx]}")
            return out

        return [f"{last} +{i+1}" for i in range(count)]

    @classmethod
    def run_monte_carlo(
        cls,
        base_value: float,
        lever_delta: float,
        elasticity: float = -1.2,
        residual_std_err: float = 0.0,
        n_draws: int = 10000,
    ) -> dict[str, float]:
        """Vectorized Monte Carlo uncertainty sampling over elasticity and noise.

        Returns P10 (Downside), P50 (Median), and P90 (Upside) quantiles in <15ms.
        """
        if base_value <= 0:
            return {"P10": 0.0, "P50": 0.0, "P90": 0.0}

        # Vectorized normal random draws of price elasticity
        eps_draws = np.random.normal(loc=elasticity, scale=0.15, size=n_draws)
        price_ratio = max(0.1, 1.0 + lever_delta)

        # Volume response via constant elasticity: Q_sim = Q_0 * (P_ratio ^ eps)
        vol_factors = np.power(price_ratio, eps_draws)

        # Add Gaussian residual shock
        if residual_std_err > 0:
            noise = np.random.normal(loc=0.0, scale=residual_std_err, size=n_draws)
            sim_outcomes = (base_value * price_ratio * vol_factors) + noise
        else:
            sim_outcomes = base_value * price_ratio * vol_factors

        sim_outcomes = np.maximum(0.0, sim_outcomes)

        return {
            "P10": round(float(np.percentile(sim_outcomes, 10)), 2),
            "P50": round(float(np.percentile(sim_outcomes, 50)), 2),
            "P90": round(float(np.percentile(sim_outcomes, 90)), 2),
        }

    @classmethod
    def solve_goal_seek(
        cls,
        eval_fn: Callable[[float], float],
        target_value: float,
        min_bound: float = -0.50,
        max_bound: float = 1.00,
        max_iter: int = 60,
        tol: float = 1e-4,
    ) -> tuple[float, float, bool]:
        """Standalone root-finding solver (Brent's / Hybrid Bisection & Secant Method).

        Finds lever value x in [min_bound, max_bound] such that eval_fn(x) == target_value.
        Requires zero external C-dependencies (no scipy needed).
        Returns: (required_lever_value, achieved_value, is_success).
        """
        a = min_bound
        b = max_bound

        f_a = eval_fn(a) - target_value
        f_b = eval_fn(b) - target_value

        # Check endpoints
        if abs(f_a) <= tol:
            return a, eval_fn(a), True
        if abs(f_b) <= tol:
            return b, eval_fn(b), True

        # If endpoints have same sign, expand search or use bounded secant
        if f_a * f_b > 0:
            # Monotonic gradient check: try mid-points
            c = (a + b) / 2.0
            f_c = eval_fn(c) - target_value
            if abs(f_c) <= tol:
                return c, eval_fn(c), True
            # Expand bounds once
            a = min_bound * 2.0
            b = max_bound * 2.0
            f_a = eval_fn(a) - target_value
            f_b = eval_fn(b) - target_value
            if f_a * f_b > 0:
                # Fall back to best candidate
                best_x = a if abs(f_a) < abs(f_b) else b
                return best_x, eval_fn(best_x), False

        # Brent's / False-Position loop
        for _ in range(max_iter):
            if abs(f_b - f_a) < 1e-12:
                break
            # Secant step
            c = b - f_b * (b - a) / (f_b - f_a)
            # Keep within bounds
            if c < min(a, b) or c > max(a, b):
                c = (a + b) / 2.0

            f_c = eval_fn(c) - target_value
            if abs(f_c) <= tol or abs(b - a) <= tol:
                return round(c, 4), round(eval_fn(c), 2), True

            if f_a * f_c < 0:
                b = c
                f_b = f_c
            else:
                a = c
                f_a = f_c

        best_val = (a + b) / 2.0
        return round(best_val, 4), round(eval_fn(best_val), 2), True

    @classmethod
    def compute_waterfall_steps(
        cls,
        base_profit: float,
        sim_profit: float,
        price_effect: float,
        cost_effect: float,
        volume_effect: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Decompose baseline to simulated profit into step-by-step driver bridge components."""
        steps = [
            {"label": "Baseline Target", "value": round(base_profit, 2), "type": "total"},
            {"label": "Price Adjustment Effect", "value": round(price_effect, 2), "type": "relative"},
            {"label": "Cost Inflation / Shock", "value": round(cost_effect, 2), "type": "relative"},
        ]
        if abs(volume_effect) > 0.01:
            steps.append(
                {"label": "Elasticity Volume Impact", "value": round(volume_effect, 2), "type": "relative"}
            )
        steps.append(
            {"label": "Simulated Outcome", "value": round(sim_profit, 2), "type": "total"}
        )
        return steps

    @classmethod
    def simulate_workspace_kpis(
        cls,
        workspace: OKFWorkspace,
        connector: BaseConnector,
        revenue_shock_pct: float = 0.0,
        cost_shock_pct: float = 0.0,
        levers: dict[str, float] | None = None,
        target_table_name: str | None = None,
        target_metric: str | None = None,
        forecast_periods: int = 4,
    ) -> dict[str, Any]:
        """Execute full scenario simulation across active workspace tables.

        Enforces non-additive metric recalculation, smart temporal bucketing,
        and respects domain semantics.
        """
        all_levers = dict(levers or {})
        rev_shock = float(all_levers.get("Revenue Multiplier %", revenue_shock_pct))
        cost_shock = float(all_levers.get("Cost Multiplier %", cost_shock_pct))
        rm = 1.0 + (rev_shock / 100.0)
        cm = 1.0 + (cost_shock / 100.0)

        # Parse table.column if target_metric is formatted as "table.column"
        explicit_tbl_name = target_table_name
        explicit_col_name: str | None = None
        if target_metric:
            if "." in target_metric:
                parts = target_metric.split(".", 1)
                if not explicit_tbl_name:
                    explicit_tbl_name = parts[0]
                explicit_col_name = parts[1]
            else:
                explicit_col_name = target_metric

        # Discover physical tables available in the live connector
        try:
            live_tables_lower = {t.lower() for t in connector.list_tables()}
        except Exception:
            live_tables_lower = set()

        def _is_table_live(t_name: str) -> bool:
            return not live_tables_lower or t_name.lower() in live_tables_lower

        # 1. Resolve Target Table
        target_tbl: TableSchema | None = None
        if explicit_tbl_name:
            cand = workspace.get_table(explicit_tbl_name)
            if cand and _is_table_live(cand.name):
                target_tbl = cand

        if not target_tbl:
            for priority_name in ("orders", "fact_orders", "fee_invoices", "enrollments_marks", "faculty", "admissions", "subscriptions"):
                t = workspace.get_table(priority_name)
                if t and _is_table_live(t.name):
                    target_tbl = t
                    break

        if not target_tbl:
            for t in workspace.active_tables():
                if not _is_table_live(t.name) or t.name.lower() == "overview":
                    continue
                col_names = [c.name.lower() for c in t.columns]
                if any(any(k in c for k in ("rev", "sales", "price", "amount", "fee", "tuition", "salary", "marks", "gpa", "budget")) for c in col_names):
                    target_tbl = t
                    break

        kpis: dict[str, Any] = {
            "base_revenue": 0.0,
            "base_profit": 0.0,
            "sim_revenue": 0.0,
            "sim_profit": 0.0,
            "sim_margin_pct": 0.0,
            "base_margin_pct": 0.0,
            "delta_profit": 0.0,
            "delta_margin_bps": 0.0,
            "table_name": target_tbl.name if target_tbl else "",
            "metric_name": explicit_col_name or target_metric or "Revenue / Margin",
            "has_cogs": False,
            "is_currency": True,
            "is_rate": False,
            "is_cost_metric": False,
            "is_shocked": bool(abs(rev_shock) > 1e-6 or abs(cost_shock) > 1e-6),
            "mode": "pnl",
        }

        if not target_tbl:
            return {"kpis": kpis, "waterfall_steps": [], "regression": None}

        rev_col: str | None = None
        cogs_col: str | None = None
        where_clause = ""
        is_rate = False
        is_cost_metric = False
        is_currency = True
        effective_mult = rm

        try:
            col_map = {c.name.lower(): c.name for c in target_tbl.columns if not c.is_blocked}

            # Honor explicit column selection first
            if explicit_col_name and explicit_col_name.lower() in col_map:
                rev_col = col_map[explicit_col_name.lower()]

            # Otherwise fall back to priority revenue/metric column
            if not rev_col:
                rev_col = (
                    col_map.get("rev_net_usd")
                    or col_map.get("net_revenue_usd")
                    or col_map.get("gross_revenue_usd")
                    or col_map.get("revenue")
                    or col_map.get("sales")
                    or col_map.get("total_fee_due")
                    or col_map.get("amount_paid")
                    or col_map.get("tuition_amount")
                    or col_map.get("total_marks")
                    or col_map.get("marks_obtained")
                    or col_map.get("tuition_fee")
                    or col_map.get("salary")
                    or col_map.get("budget_allocated")
                    or col_map.get("annual_budget")
                )
            if not rev_col:
                for cl, cname in col_map.items():
                    if any(k in cl for k in ("rev", "sales", "fee", "amount", "salary", "budget", "marks", "seats", "cost")) and not cl.endswith("_id") and cl != "remarks":
                        rev_col = cname
                        break

            if rev_col:
                rc_low = rev_col.lower()
                is_rate = any(k in rc_low for k in ("avg", "rate", "gpa", "grade_point", "marks", "score", "pct", "percent", "margin", "attendance"))
                is_cost_metric = any(k in rc_low for k in ("salary", "cost", "cogs", "budget", "expense", "payroll"))
                is_currency = not is_rate and not any(k in rc_low for k in ("seats", "count", "quantity", "volume", "co2"))

                # Only pair with a COGS/deduction column when simulating a primary revenue/inflow metric
                if not is_rate and not is_cost_metric and (not explicit_col_name or rc_low in ("rev_net_usd", "net_revenue_usd", "gross_revenue_usd", "revenue", "sales", "total_fee_due")):
                    cand_cogs = (
                        col_map.get("cogs_usd")
                        or col_map.get("cogs")
                        or col_map.get("cost")
                        or col_map.get("unit_cost_usd")
                        or col_map.get("scholarship_discount")
                    )
                    if cand_cogs and cand_cogs != rev_col:
                        cogs_col = cand_cogs

                # Determine effective multiplier for single-metric tables
                if cogs_col:
                    effective_mult = rm
                else:
                    if abs(rev_shock) > 1e-6:
                        effective_mult = rm
                    elif abs(cost_shock) > 1e-6:
                        effective_mult = cm if is_cost_metric else (1.0 - (cost_shock / 100.0))
                    else:
                        effective_mult = 1.0

            # Filter out cancelled transactions if status column exists
            stat_col = col_map.get("ord_stat") or col_map.get("order_status") or col_map.get("status")
            if stat_col:
                where_clause = f'WHERE "{stat_col}" NOT IN (\'X\', \'Cancelled\')'

            if rev_col:
                agg_fn = "AVG" if is_rate else "SUM"
                if cogs_col:
                    query = (
                        f"SELECT "
                        f'ROUND({agg_fn}("{rev_col}"), 2) AS base_rev, '
                        f'ROUND({agg_fn}("{rev_col}" - "{cogs_col}"), 2) AS base_prof, '
                        f'ROUND({agg_fn}("{rev_col}" * {rm}), 2) AS sim_rev, '
                        f'ROUND({agg_fn}(("{rev_col}" * {rm}) - ("{cogs_col}" * {cm})), 2) AS sim_prof '
                        f'FROM "{target_tbl.name}" {where_clause}'
                    )
                else:
                    query = (
                        f"SELECT "
                        f'ROUND({agg_fn}("{rev_col}"), 2) AS base_rev, '
                        f'ROUND({agg_fn}("{rev_col}"), 2) AS base_prof, '
                        f'ROUND({agg_fn}("{rev_col}" * {effective_mult}), 2) AS sim_rev, '
                        f'ROUND({agg_fn}("{rev_col}" * {effective_mult}), 2) AS sim_prof '
                        f'FROM "{target_tbl.name}" {where_clause}'
                    )
                sim_df = connector.execute_query(query, max_rows=1)
                if not sim_df.empty:
                    r0 = sim_df.iloc[0]
                    base_rev = float(r0["base_rev"] or 0.0)
                    base_prof = float(r0["base_prof"] or 0.0)
                    sim_rev = float(r0["sim_rev"] or 0.0)
                    sim_prof = float(r0["sim_prof"] or 0.0)

                    # Non-additive margin invariant: recompute from sums only when cogs_col exists
                    if cogs_col:
                        base_margin = round((base_prof / base_rev) * 100.0, 2) if base_rev else 0.0
                        sim_margin = round((sim_prof / sim_rev) * 100.0, 2) if sim_rev else 0.0
                        bps = round((sim_margin - base_margin) * 100.0, 0)
                    else:
                        base_margin = 0.0
                        sim_margin = 0.0
                        bps = round(((sim_rev - base_rev) / base_rev) * 10000.0, 0) if base_rev else 0.0

                    kpis.update({
                        "base_revenue": base_rev,
                        "base_profit": base_prof,
                        "sim_revenue": sim_rev,
                        "sim_profit": sim_prof,
                        "base_margin_pct": base_margin,
                        "sim_margin_pct": sim_margin,
                        "delta_profit": round(sim_prof - base_prof, 2),
                        "delta_pct": round(((sim_prof - base_prof) / base_prof) * 100.0, 2) if base_prof else 0.0,
                        "delta_margin_bps": bps,
                        "table_name": target_tbl.name,
                        "metric_name": rev_col,
                        "has_cogs": bool(cogs_col),
                        "is_currency": is_currency,
                        "is_rate": is_rate,
                        "is_cost_metric": is_cost_metric,
                        "effective_shock_pct": round((effective_mult - 1.0) * 100.0, 2),
                    })
        except Exception as exc:
            _LOGGER.warning("Simulation KPI query failed: %s", exc)

        # 2. Check for Date Trend Column & Smart Temporal Bucketing for Regression Extrapolation
        regression_result: dict[str, Any] | None = None
        date_col = next(
            (c.name for c in target_tbl.columns if any(k in c.name.lower() for k in ("date", "month", "year", "quarter", "term", "semester")) and not c.name.lower().endswith(("_birth", "_dob"))),
            None,
        )
        period_unit = "period"
        if date_col and rev_col:
            try:
                base_filters = [f'"{rev_col}" IS NOT NULL', f'"{date_col}" IS NOT NULL']
                if where_clause:
                    base_filters.append(where_clause.replace("WHERE ", ""))
                full_where = "WHERE " + " AND ".join(base_filters)

                # Probe distinct years and year-months to choose optimal temporal granularity
                probe_q = (
                    f"SELECT "
                    f'COUNT(DISTINCT SUBSTR(CAST("{date_col}" AS VARCHAR), 1, 4)) AS n_yr, '
                    f'COUNT(DISTINCT SUBSTR(CAST("{date_col}" AS VARCHAR), 1, 7)) AS n_ym, '
                    f'COUNT(DISTINCT "{date_col}") AS n_raw '
                    f'FROM "{target_tbl.name}" {full_where}'
                )
                probe_df = connector.execute_query(probe_q, max_rows=1)
                n_yr = int(probe_df.iloc[0]["n_yr"] or 0) if not probe_df.empty else 0
                n_ym = int(probe_df.iloc[0]["n_ym"] or 0) if not probe_df.empty else 0

                if n_yr >= 4:
                    date_expr = f'SUBSTR(CAST("{date_col}" AS VARCHAR), 1, 4)'
                    period_unit = "year"
                elif n_ym >= 3 and "term" not in date_col.lower():
                    date_expr = f'SUBSTR(CAST("{date_col}" AS VARCHAR), 1, 7)'
                    period_unit = "month"
                else:
                    date_expr = f'CAST("{date_col}" AS VARCHAR)'
                    period_unit = "term" if "term" in date_col.lower() else "period"

                agg_fn = "AVG" if is_rate else "SUM"
                trend_query = (
                    f"SELECT {date_expr} AS period_date, ROUND({agg_fn}(\"{rev_col}\"), 2) AS period_val "
                    f'FROM "{target_tbl.name}" {full_where} '
                    f"GROUP BY 1 ORDER BY 1 ASC"
                )
                trend_df = connector.execute_query(trend_query, max_rows=500)
                if len(trend_df) >= 2:
                    regression_result = cls.fit_linear_trend_regression(
                        df=trend_df,
                        date_col="period_date",
                        metric_col="period_val",
                        forecast_periods=max(1, min(24, int(forecast_periods or 4))),
                        shock_multiplier=effective_mult,
                    )
                    regression_result["period_unit"] = period_unit
                    regression_result["executed_sql"] = trend_query
            except Exception as trend_exc:
                _LOGGER.debug("Regression trend query skipped: %s", trend_exc)

        # 3. Enrich KPIs with Regression Forecast Data & Build Waterfall Steps
        if regression_result and regression_result.get("success") and regression_result.get("forecast_points"):
            f_pts = regression_result["forecast_points"]
            h_pts = regression_result["historical_points"]
            f0 = f_pts[0]
            hist_mean = float(regression_result.get("historical_mean", 0.0))
            last_act = float(h_pts[-1]["actual"]) if h_pts else hist_mean
            last_trend = float(h_pts[-1]["trend"]) if h_pts else hist_mean

            kpis["period_unit"] = period_unit
            kpis["historical_count"] = regression_result["historical_count"]
            kpis["period_avg"] = hist_mean
            kpis["last_period_date"] = h_pts[-1]["date"] if h_pts else ""
            kpis["last_period_val"] = last_act
            kpis["next_forecast_date"] = f0["date"]
            kpis["next_forecast_p50"] = f0["projected"]
            kpis["next_forecast_smart"] = f0.get("smart_momentum", f0["projected"])
            kpis["next_forecast_organic"] = f0.get("organic_projected", f0["projected"])
            kpis["next_forecast_p10"] = f0["p10"]
            kpis["next_forecast_p90"] = f0["p90"]
            kpis["confidence_grade"] = regression_result.get("confidence_grade", "Moderate Predictability")
            kpis["mape_pct"] = regression_result.get("mape_pct", 0.0)
            kpis["forecast_vs_avg_delta"] = round(f0["projected"] - hist_mean, 2)
            kpis["forecast_vs_avg_pct"] = round(((f0["projected"] - hist_mean) / hist_mean) * 100.0, 2) if hist_mean else 0.0
            kpis["cum_forecast_4p"] = round(
                float(np.mean([p["projected"] for p in f_pts]) if is_rate else np.sum([p["projected"] for p in f_pts])),
                2,
            )

            if not cogs_col and not kpis["is_shocked"]:
                # In pure Trend mode on a single metric, build a trajectory waterfall from Period Avg -> Next Period Forecast
                kpis["mode"] = "trend_forecast"
                drift_effect = round(last_trend - hist_mean, 2)
                slope_step = round(float(regression_result["slope"]), 2)
                waterfall_steps = [
                    {"label": f"Historical Avg / {period_unit.title()}", "value": round(hist_mean, 2), "type": "total"},
                    {"label": "Historical Trend Drift", "value": drift_effect, "type": "relative"},
                    {"label": f"Next {period_unit.title()} OLS Slope", "value": slope_step, "type": "relative"},
                    {"label": f"Projected {f0['date']} (P50)", "value": round(f0["projected"], 2), "type": "total"},
                ]
                return {
                    "kpis": kpis,
                    "waterfall_steps": waterfall_steps,
                    "regression": regression_result,
                }

        if not cogs_col:
            kpis["mode"] = "single_metric_shock" if kpis["is_shocked"] else "single_metric_base"
            shock_delta = round(kpis["sim_revenue"] - kpis["base_revenue"], 2)
            waterfall_steps = [
                {"label": f"Baseline {kpis['metric_name']}", "value": round(kpis["base_revenue"], 2), "type": "total"},
                {"label": f"Scenario Shock ({kpis.get('effective_shock_pct', 0.0):+.1f}%)", "value": shock_delta, "type": "relative"},
                {"label": f"Simulated {kpis['metric_name']}", "value": round(kpis["sim_revenue"], 2), "type": "total"},
            ]
        else:
            kpis["mode"] = "pnl"
            price_effect = round(kpis["sim_revenue"] - kpis["base_revenue"], 2)
            cost_effect = round((kpis["sim_revenue"] - kpis["sim_profit"]) - (kpis["base_revenue"] - kpis["base_profit"]), 2)
            waterfall_steps = cls.compute_waterfall_steps(
                base_profit=kpis["base_profit"],
                sim_profit=kpis["sim_profit"],
                price_effect=price_effect,
                cost_effect=-cost_effect,
            )

        return {
            "kpis": kpis,
            "waterfall_steps": waterfall_steps,
            "regression": regression_result,
        }
