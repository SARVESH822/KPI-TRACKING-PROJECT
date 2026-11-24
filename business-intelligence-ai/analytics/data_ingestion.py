"""
Handles user-uploaded structured and unstructured CSVs: validation, column
mapping, and normalization into the internal schema the analytics engine
expects. Never crashes on bad input — returns (df_or_None, list_of_errors).
"""
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# Internal canonical schema. Only "date" and one numeric measure are strictly
# required; everything else degrades gracefully if absent.
REQUIRED_STRUCTURED = ["date"]
NUMERIC_STRUCTURED = ["revenue", "orders", "units", "price", "discount_rate",
                       "margin", "conversion_rate", "delivery_days", "returns_rate"]
DIMENSION_CANDIDATES_HINT = ["region", "category", "channel", "segment"]

REQUIRED_UNSTRUCTURED = ["date", "text"]
OPTIONAL_UNSTRUCTURED = ["document_id", "region", "category", "source", "title"]


@dataclass
class IngestResult:
    df: Optional[pd.DataFrame]
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    detected_dimensions: list = field(default_factory=list)
    detected_measures: list = field(default_factory=list)


def load_uploaded_csv(file) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(file)
    except Exception:
        return None


def validate_and_map_structured(raw_df: pd.DataFrame, column_map: dict) -> IngestResult:
    """
    column_map: {"date": "<user's column name>", "revenue": "<...>", ...}
    Only "date" is mandatory; any measure/dimension left unmapped (empty
    string) is simply skipped.
    """
    errors, warnings = [], []

    date_col = column_map.get("date")
    if not date_col or date_col not in raw_df.columns:
        errors.append("A date column is required and must be mapped.")
        return IngestResult(df=None, errors=errors)

    out = pd.DataFrame()
    try:
        out["date"] = pd.to_datetime(raw_df[date_col], errors="coerce")
    except Exception:
        errors.append(f"Could not parse '{date_col}' as dates.")
        return IngestResult(df=None, errors=errors)

    n_bad_dates = out["date"].isna().sum()
    if n_bad_dates > 0:
        warnings.append(f"{n_bad_dates} rows had unparseable dates and will be dropped.")
    out = out[out["date"].notna()]

    detected_measures = []
    for measure in NUMERIC_STRUCTURED:
        col = column_map.get(measure)
        if col and col in raw_df.columns:
            numeric = pd.to_numeric(raw_df.loc[out.index, col], errors="coerce")
            n_bad = numeric.isna().sum()
            if n_bad > 0:
                warnings.append(f"{n_bad} non-numeric values in '{col}' mapped to {measure} were treated as 0.")
            out[measure] = numeric.fillna(0.0)
            detected_measures.append(measure)

    if not detected_measures:
        errors.append("At least one numeric measure (revenue, orders, etc.) must be mapped.")
        return IngestResult(df=None, errors=errors)

    detected_dimensions = []
    for dim in DIMENSION_CANDIDATES_HINT:
        col = column_map.get(dim)
        if col and col in raw_df.columns:
            out[dim] = raw_df.loc[out.index, col].astype(str)
            detected_dimensions.append(dim)

    # Also allow arbitrary extra dimension columns the user explicitly mapped
    for key, col in column_map.items():
        if key.startswith("extra_dim:") and col and col in raw_df.columns:
            dim_name = key.split("extra_dim:", 1)[1]
            out[dim_name] = raw_df.loc[out.index, col].astype(str)
            detected_dimensions.append(dim_name)

    if not detected_dimensions:
        warnings.append("No dimension columns mapped — driver/contribution analysis will be unavailable.")

    if len(out) == 0:
        errors.append("No valid rows remained after validation.")
        return IngestResult(df=None, errors=errors)

    return IngestResult(df=out, errors=errors, warnings=warnings,
                         detected_dimensions=detected_dimensions,
                         detected_measures=detected_measures)


def validate_and_map_unstructured(raw_df: pd.DataFrame, column_map: dict) -> IngestResult:
    errors, warnings = [], []

    date_col = column_map.get("date")
    text_col = column_map.get("text")
    if not date_col or date_col not in raw_df.columns:
        errors.append("A date column is required and must be mapped.")
        return IngestResult(df=None, errors=errors)
    if not text_col or text_col not in raw_df.columns:
        errors.append("A text column is required and must be mapped.")
        return IngestResult(df=None, errors=errors)

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw_df[date_col], errors="coerce")
    n_bad_dates = out["date"].isna().sum()
    if n_bad_dates > 0:
        warnings.append(f"{n_bad_dates} rows had unparseable dates and will be dropped.")
    out = out[out["date"].notna()]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    out["text"] = raw_df.loc[out.index, text_col].astype(str)
    out = out[out["text"].str.strip() != ""]

    for optional_field in OPTIONAL_UNSTRUCTURED:
        col = column_map.get(optional_field)
        if col and col in raw_df.columns:
            out[optional_field] = raw_df.loc[out.index, col].astype(str)

    if "document_id" not in out.columns:
        out["document_id"] = [f"DOC-{i:04d}" for i in range(1, len(out) + 1)]
    if "title" not in out.columns:
        out["title"] = out["text"].str.slice(0, 40)
    if "source" not in out.columns:
        out["source"] = "Uploaded Document"
    if "region" not in out.columns:
        out["region"] = None
    if "category" not in out.columns:
        out["category"] = None

    if len(out) == 0:
        errors.append("No valid rows remained after validation.")
        return IngestResult(df=None, errors=errors)

    return IngestResult(df=out, errors=errors, warnings=warnings)
