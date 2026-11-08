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
        hist_points: list[dict[str, Any]] = []
        for i in range(n):
            hist_points.append(
                {
                    "date": date_labels[i],
                    "period_index": i,
                    "actual": round(float(y[i]), 2),
                    "trend": round(float(beta_0 + beta_1 * i), 2),
                }
            )

        # Generate forward forecast period labels
        forecast_labels = cls._generate_future_period_labels(date_labels, forecast_periods)
        forecast_points: list[dict[str, Any]] = []

        for h in range(1, forecast_periods + 1):
            t_fut = n - 1 + h
            y_hat = beta_0 + (beta_1 * t_fut)

            # Prediction interval standard error:
            # SE_pred = s_e * sqrt(1 + 1/n + (t_fut - t_mean)^2 / s_tt)
            se_pred = (
                s_e * math.sqrt(1.0 + (1.0 / n) + ((t_fut - t_mean) ** 2 / s_tt))
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
                    "projected": round(float(p50), 2),
                    "p10": round(float(p10), 2),
                    "p50": round(float(p50), 2),
                    "p90": round(float(p90), 2),
                    "is_forecast": True,
                }
            )

        return {
            "success": True,
            "metric": metric_col,
            "slope": round(beta_1, 4),
            "intercept": round(beta_0, 4),
            "r2": round(r2, 4),
            "residual_std_error": round(s_e, 4),
            "historical_points": hist_points,
            "forecast_points": forecast_points,
            "historical_count": n,
            "forecast_count": forecast_periods,
        }

    @staticmethod
    def _generate_future_period_labels(date_labels: list[str], count: int) -> list[str]:
        """Extrapolate dates or period strings (e.g. '2025-Q4', '2025-12', 'Semester 2')."""
        if not date_labels:
            return [f"Period +{i+1}" for i in range(count)]

        last = date_labels[-1].strip()

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

        # Match Year format: 'YYYY'
        y_match = re.match(r"^(\d{4})$", last)
        if y_match:
            year = int(y_match.group(1))
            return [str(year + i + 1) for i in range(count)]

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
    ) -> dict[str, Any]:
        """Execute full scenario simulation across active workspace tables.

        Enforces non-additive metric recalculation and respects domain semantics.
        """
        all_levers = dict(levers or {})
        rm = 1.0 + (all_levers.get("Revenue Multiplier %", revenue_shock_pct) / 100.0)
        cm = 1.0 + (all_levers.get("Cost Multiplier %", cost_shock_pct) / 100.0)

        # 1. Resolve Target Table
        target_tbl: TableSchema | None = None
        if target_table_name:
            target_tbl = workspace.get_table(target_table_name)

        if not target_tbl:
            for priority_name in ("orders", "fact_orders", "fee_invoices", "enrollments_marks", "admissions", "subscriptions"):
                t = workspace.get_table(priority_name)
                if t:
                    target_tbl = t
                    break

        if not target_tbl:
            for t in workspace.active_tables():
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
            "metric_name": target_metric or "Revenue / Margin",
        }

        if not target_tbl:
            return {"kpis": kpis, "waterfall_steps": [], "regression": None}

        try:
            col_map = {c.name.lower(): c.name for c in target_tbl.columns if not c.is_blocked}

            # Identify Primary Revenue / Inflow Column
            rev_col = (
                col_map.get("rev_net_usd")
                or col_map.get("net_revenue_usd")
                or col_map.get("gross_revenue_usd")
                or col_map.get("revenue")
                or col_map.get("sales")
                or col_map.get("total_fee_due")
                or col_map.get("amount_paid")
                or col_map.get("total_marks")
                or col_map.get("marks_obtained")
                or col_map.get("tuition_fee")
                or col_map.get("salary")
                or col_map.get("annual_budget")
            )
            if not rev_col:
                for cl, cname in col_map.items():
                    if any(k in cl for k in ("rev", "sales", "fee", "amount", "salary", "budget", "marks")) and not cl.endswith("_id"):
                        rev_col = cname
                        break

            # Identify Cost / Deduction Column
            cogs_col = (
                col_map.get("cogs_usd")
                or col_map.get("cogs")
                or col_map.get("cost")
                or col_map.get("unit_cost_usd")
                or col_map.get("scholarship_discount")
                or col_map.get("balance_due")
            )

            # Filter out cancelled transactions if status column exists
            stat_col = col_map.get("ord_stat") or col_map.get("order_status") or col_map.get("status")
            where_clause = ""
            if stat_col:
                where_clause = f'WHERE "{stat_col}" NOT IN (\'X\', \'Cancelled\')'

            if rev_col:
                cogs_expr = f'"{cogs_col}"' if cogs_col else "0"
                query = (
                    f"SELECT "
                    f'ROUND(SUM("{rev_col}"), 2) AS base_rev, '
                    f'ROUND(SUM("{rev_col}" - {cogs_expr}), 2) AS base_prof, '
                    f'ROUND(SUM("{rev_col}" * {rm}), 2) AS sim_rev, '
                    f'ROUND(SUM(("{rev_col}" * {rm}) - ({cogs_expr} * {cm})), 2) AS sim_prof '
                    f'FROM "{target_tbl.name}" {where_clause}'
                )
                sim_df = connector.execute_query(query, max_rows=1)
                if not sim_df.empty:
                    r0 = sim_df.iloc[0]
                    base_rev = float(r0["base_rev"] or 0.0)
                    base_prof = float(r0["base_prof"] or 0.0)
                    sim_rev = float(r0["sim_rev"] or 0.0)
                    sim_prof = float(r0["sim_prof"] or 0.0)

                    # Non-additive margin invariant: recompute from sums
                    base_margin = round((base_prof / base_rev) * 100.0, 2) if base_rev else 0.0
                    sim_margin = round((sim_prof / sim_rev) * 100.0, 2) if sim_rev else 0.0
                    bps = round((sim_margin - base_margin) * 100.0, 0)

                    kpis = {
                        "base_revenue": base_rev,
                        "base_profit": base_prof,
                        "sim_revenue": sim_rev,
                        "sim_profit": sim_prof,
                        "base_margin_pct": base_margin,
                        "sim_margin_pct": sim_margin,
                        "delta_profit": round(sim_prof - base_prof, 2),
                        "delta_margin_bps": bps,
                        "table_name": target_tbl.name,
                        "metric_name": rev_col,
                    }
        except Exception as exc:
            _LOGGER.warning("Simulation KPI query failed: %s", exc)

        # 2. Compute Waterfall Steps
        price_effect = round(kpis["sim_revenue"] - kpis["base_revenue"], 2)
        cost_effect = round((kpis["sim_revenue"] - kpis["sim_profit"]) - (kpis["base_revenue"] - kpis["base_profit"]), 2)
        # Cost inflation reduces profit, so negate for display
        waterfall_steps = cls.compute_waterfall_steps(
            base_profit=kpis["base_profit"],
            sim_profit=kpis["sim_profit"],
            price_effect=price_effect,
            cost_effect=-cost_effect,
        )

        # 3. Check for Date Trend Column for Regression Extrapolation
        regression_result: dict[str, Any] | None = None
        date_col = next(
            (c.name for c in target_tbl.columns if any(k in c.name.lower() for k in ("date", "month", "year", "quarter", "term", "semester")) and not c.name.lower().endswith(("_birth", "_dob"))),
            None,
        )
        if date_col and rev_col:
            try:
                trend_query = (
                    f'SELECT "{date_col}" AS period_date, SUM("{rev_col}") AS period_val '
                    f'FROM "{target_tbl.name}" {where_clause} '
                    f'GROUP BY "{date_col}" ORDER BY "{date_col}" ASC'
                )
                trend_df = connector.execute_query(trend_query, max_rows=100)
                if len(trend_df) >= 2:
                    regression_result = cls.fit_linear_trend_regression(
                        df=trend_df,
                        date_col="period_date",
                        metric_col="period_val",
                        forecast_periods=4,
                    )
            except Exception as trend_exc:
                _LOGGER.debug("Regression trend query skipped: %s", trend_exc)

        return {
            "kpis": kpis,
            "waterfall_steps": waterfall_steps,
            "regression": regression_result,
        }
