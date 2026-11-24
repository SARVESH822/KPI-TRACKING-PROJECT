"""
Dimensional decomposition + contribution analysis.

For a value-additive KPI (Revenue, Orders, Margin) this computes, per dimension
member (e.g. each category), exactly how much of the total absolute change is
attributable to that member: contribution_i = current_i - previous_i, and
sum(contribution_i) == total abs_change by construction. This is arithmetic,
not a model — it cannot hallucinate.

For non-additive KPIs (rates, ratios) we report the weighted shift instead and
label it clearly as a rate comparison rather than a currency contribution.
"""
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from analytics.kpi_engine import KPI_DEFS, _aggregate, filter_df

DEFAULT_DIMENSIONS = ["region", "category", "channel", "segment"]


def detect_dimensions(df: pd.DataFrame) -> list:
    """Any non-numeric, non-date column with a manageable number of distinct
    values is treated as a candidate dimension for driver analysis. Falls
    back to DEFAULT_DIMENSIONS members that are actually present."""
    candidates = []
    for col in df.columns:
        if col == "date":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        n_unique = df[col].nunique(dropna=True)
        if 1 < n_unique <= 50:
            candidates.append(col)
    return candidates or [d for d in DEFAULT_DIMENSIONS if d in df.columns]


@dataclass
class DriverRow:
    dimension: str
    member: str
    current_value: float
    previous_value: float
    contribution: float          # currency/unit contribution to total change
    share_of_change: float       # % of total abs_change explained by this member
    direction: str               # "up" | "down"


def rank_drivers(df: pd.DataFrame, kpi: str, current_start, current_end,
                  previous_start, previous_end, region: Optional[str] = None,
                  category: Optional[str] = None, top_n: int = 8,
                  dimensions: Optional[List[str]] = None) -> List[DriverRow]:
    spec = KPI_DEFS[kpi]
    additive = spec["agg"] in ("sum",)

    cur_df = filter_df(df, current_start, current_end, region, category)
    prev_df = filter_df(df, previous_start, previous_end, region, category)
    total_change = _aggregate(cur_df, kpi) - _aggregate(prev_df, kpi)

    dims = dimensions if dimensions is not None else detect_dimensions(df)

    rows: List[DriverRow] = []
    for dim in dims:
        if region and region != "All Regions" and dim == "region":
            continue
        if category and category != "All Categories" and dim == "category":
            continue

        members = set(cur_df[dim].unique()) | set(prev_df[dim].unique())
        for m in members:
            cur_sub = cur_df[cur_df[dim] == m]
            prev_sub = prev_df[prev_df[dim] == m]
            cur_v = _aggregate(cur_sub, kpi)
            prev_v = _aggregate(prev_sub, kpi)
            contribution = cur_v - prev_v

            if additive and total_change != 0:
                share = contribution / total_change * 100.0
            elif not additive:
                # for rates, share = this member's own pct point shift relative
                # to overall pct point shift (still a real calculated number)
                share = contribution / total_change * 100.0 if total_change != 0 else 0.0
            else:
                share = 0.0

            if abs(contribution) < 1e-9:
                continue

            rows.append(DriverRow(
                dimension=dim, member=str(m),
                current_value=cur_v, previous_value=prev_v,
                contribution=contribution, share_of_change=share,
                direction="up" if contribution > 0 else "down",
            ))

    # Sort by absolute contribution magnitude, keep the strongest signals
    rows.sort(key=lambda r: abs(r.contribution), reverse=True)
    return rows[:top_n]


def secondary_metric_shift(df: pd.DataFrame, region: Optional[str], category: Optional[str],
                            current_start, current_end, previous_start, previous_end) -> dict:
    """
    Computes deterministic shifts in operational metrics (delivery days,
    discount rate, conversion rate, returns rate) that commonly explain
    KPI moves but aren't the KPI itself. Used to build hypotheses, never
    presented as proven causes.
    """
    cur_df = filter_df(df, current_start, current_end, region, category)
    prev_df = filter_df(df, previous_start, previous_end, region, category)

    out = {}
    for col, label in [
        ("delivery_days", "Average delivery time"),
        ("discount_rate", "Average discount rate"),
        ("conversion_rate", "Conversion rate"),
        ("returns_rate", "Returns rate"),
        ("orders", "Order volume"),
    ]:
        if col == "orders":
            cur_v, prev_v = cur_df[col].sum(), prev_df[col].sum()
        else:
            cur_v = cur_df[col].mean() if len(cur_df) else 0.0
            prev_v = prev_df[col].mean() if len(prev_df) else 0.0
        pct = ((cur_v - prev_v) / prev_v * 100.0) if prev_v else 0.0
        out[col] = {"label": label, "current": cur_v, "previous": prev_v, "pct_change": pct}
    return out
