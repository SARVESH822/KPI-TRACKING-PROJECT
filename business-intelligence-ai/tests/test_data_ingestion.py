import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from analytics.data_ingestion import validate_and_map_structured, validate_and_map_unstructured
from analytics.decomposition import detect_dimensions


def test_valid_structured_upload_maps_correctly():
    raw = pd.DataFrame({
        "Order Date": ["2026-08-01", "2026-08-02", "2026-08-03"],
        "Rev": [100.0, 200.0, 300.0],
        "Area": ["North", "South", "North"],
    })
    result = validate_and_map_structured(raw, {"date": "Order Date", "revenue": "Rev",
                                                 "extra_dim:region": "Area"})
    assert result.errors == []
    assert result.df is not None
    assert "revenue" in result.df.columns
    assert "region" in result.df.columns
    assert len(result.df) == 3


def test_structured_upload_missing_date_column_errors():
    raw = pd.DataFrame({"Rev": [1, 2, 3]})
    result = validate_and_map_structured(raw, {"date": "", "revenue": "Rev"})
    assert result.df is None
    assert any("date" in e.lower() for e in result.errors)


def test_structured_upload_no_measures_errors():
    raw = pd.DataFrame({"Order Date": ["2026-08-01"]})
    result = validate_and_map_structured(raw, {"date": "Order Date"})
    assert result.df is None
    assert any("measure" in e.lower() for e in result.errors)


def test_structured_upload_bad_dates_dropped_with_warning():
    raw = pd.DataFrame({
        "Order Date": ["2026-08-01", "not-a-date", "2026-08-03"],
        "Rev": [100.0, 200.0, 300.0],
    })
    result = validate_and_map_structured(raw, {"date": "Order Date", "revenue": "Rev"})
    assert len(result.df) == 2
    assert any("unparseable" in w.lower() for w in result.warnings)


def test_valid_unstructured_upload_maps_correctly():
    raw = pd.DataFrame({
        "Ticket Date": ["2026-08-01", "2026-08-02"],
        "Body": ["Delivery was late", "Great service"],
    })
    result = validate_and_map_unstructured(raw, {"date": "Ticket Date", "text": "Body"})
    assert result.errors == []
    assert len(result.df) == 2
    assert "document_id" in result.df.columns
    assert "source" in result.df.columns


def test_unstructured_upload_missing_text_column_errors():
    raw = pd.DataFrame({"Ticket Date": ["2026-08-01"]})
    result = validate_and_map_unstructured(raw, {"date": "Ticket Date", "text": ""})
    assert result.df is None
    assert any("text" in e.lower() for e in result.errors)


def test_detect_dimensions_finds_categorical_columns():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-01", "2026-08-02"]),
        "revenue": [100.0, 200.0],
        "region": ["North", "South"],
    })
    dims = detect_dimensions(df)
    assert "region" in dims
    assert "revenue" not in dims
    assert "date" not in dims
