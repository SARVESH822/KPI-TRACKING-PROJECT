"""
Deterministic KPI computation. No LLM involved anywhere in this module.
All numeric claims the product ever makes originate here or in decomposition.py.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

KPI_DEFS = {
    "Revenue": {"col": "revenue", "agg": "sum", "fmt": "currency"},
    "Orders": {"col": "orders", "agg": "sum", "fmt": "int"},
    "Margin": {"col": "margin", "agg": "sum", "fmt": "currency"},
    "Average Order Value": {"col": None, "agg": "aov", "fmt": "currency"},
    "Conversion Rate": {"col": "conversion_rate", "agg": "mean", "fmt": "percent"},
    "Gross Margin %": {"col": None, "agg": "margin_pct", "fmt": "percent"},
    "Average Delivery Days": {"col": "delivery_days", "agg": "mean", "fmt": "days"},
}


@dataclass
class KpiResult:
    kpi: str
    current_value: float
    previous_value: float
    abs_change: float
    pct_change: float
    fmt: str


def _aggregate(df: pd.DataFrame, kpi: str) -> float:
    spec = KPI_DEFS[kpi]
    if spec["agg"] == "sum":
        return float(df[spec["col"]].sum())
    if spec["agg"] == "mean":
        return float(df[spec["col"]].mean()) if len(df) else 0.0
    if spec["agg"] == "aov":
        orders = df["orders"].sum()
        return float(df["revenue"].sum() / orders) if orders else 0.0
    if spec["agg"] == "margin_pct":
        rev = df["revenue"].sum()
        return float(df["margin"].sum() / rev) if rev else 0.0
    raise ValueError(f"Unknown agg for KPI {kpi}")


def filter_df(df: pd.DataFrame, start, end, region: Optional[str] = None,
              category: Optional[str] = None) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    if region and region != "All Regions":
        mask &= df["region"] == region
    if category and category != "All Categories":
        mask &= df["category"] == category
    return df[mask]


def compute_kpi(df: pd.DataFrame, kpi: str, current_start, current_end,
                 previous_start, previous_end, region: Optional[str] = None,
                 category: Optional[str] = None) -> KpiResult:
    cur_df = filter_df(df, current_start, current_end, region, category)
    prev_df = filter_df(df, previous_start, previous_end, region, category)

    cur_val = _aggregate(cur_df, kpi)
    prev_val = _aggregate(prev_df, kpi)

    abs_change = cur_val - prev_val
    pct_change = (abs_change / prev_val * 100.0) if prev_val not in (0, 0.0) else 0.0

    return KpiResult(
        kpi=kpi,
        current_value=cur_val,
        previous_value=prev_val,
        abs_change=abs_change,
        pct_change=pct_change,
        fmt=KPI_DEFS[kpi]["fmt"],
    )


def load_structured_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df
