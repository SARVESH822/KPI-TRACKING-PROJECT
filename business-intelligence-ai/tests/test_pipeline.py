import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from analytics.kpi_engine import compute_kpi, load_structured_data
from analytics.decomposition import rank_drivers, secondary_metric_shift
from analytics.pipeline import run_analysis, previous_period
from retrieval.retriever import DocumentRetriever, load_documents

STRUCT_PATH = ROOT / "data" / "structured" / "sales.csv"
UNSTRUCT_PATH = ROOT / "data" / "unstructured" / "documents.csv"


@pytest.fixture(scope="module")
def df():
    return load_structured_data(str(STRUCT_PATH))


@pytest.fixture(scope="module")
def docs_df():
    return load_documents(str(UNSTRUCT_PATH))


@pytest.fixture(scope="module")
def retriever(docs_df):
    return DocumentRetriever(docs_df)


# ---------- KPI engine ----------

def test_kpi_pct_change_basic(df):
    result = compute_kpi(df, "Revenue", date(2026, 8, 1), date(2026, 8, 31),
                          date(2026, 7, 1), date(2026, 7, 31))
    assert result.previous_value > 0
    expected_pct = (result.current_value - result.previous_value) / result.previous_value * 100
    assert abs(result.pct_change - expected_pct) < 1e-6


def test_kpi_zero_previous_period_no_crash(df):
    # A window with no data at all should not divide by zero / crash
    result = compute_kpi(df, "Revenue", date(2026, 8, 1), date(2026, 8, 31),
                          date(2020, 1, 1), date(2020, 1, 31))
    assert result.previous_value == 0
    assert result.pct_change == 0.0


def test_previous_period_helper():
    start, end = previous_period(date(2026, 8, 1), date(2026, 8, 31))
    assert (end - start).days + 1 == 31
    assert end == date(2026, 7, 31)


# ---------- Decomposition / contribution analysis ----------

def test_driver_contributions_sum_to_total_change(df):
    from analytics.kpi_engine import compute_kpi
    result = compute_kpi(df, "Revenue", date(2026, 8, 1), date(2026, 8, 31),
                          date(2026, 7, 1), date(2026, 7, 31))
    drivers = rank_drivers(df, "Revenue", date(2026, 8, 1), date(2026, 8, 31),
                            date(2026, 7, 1), date(2026, 7, 31), top_n=1000)
    region_only = [d for d in drivers if d.dimension == "region"]
    contribution_sum = sum(d.contribution for d in region_only)
    assert abs(contribution_sum - result.abs_change) < 1.0  # rounding tolerance


def test_south_india_electronics_declines_in_august(df):
    """Hero scenario check: the injected event must be discoverable."""
    result = compute_kpi(df, "Revenue", date(2026, 8, 1), date(2026, 8, 31),
                          date(2026, 7, 1), date(2026, 7, 31),
                          region="South India", category="Electronics")
    assert result.pct_change < -10  # meaningfully negative


def test_west_india_revenue_rises_in_august(df):
    result = compute_kpi(df, "Revenue", date(2026, 8, 1), date(2026, 8, 31),
                          date(2026, 7, 1), date(2026, 7, 31), region="West India")
    assert result.pct_change > 10


def test_west_india_margin_pct_falls_despite_revenue_rise(df):
    rev = compute_kpi(df, "Revenue", date(2026, 8, 1), date(2026, 8, 31),
                       date(2026, 7, 1), date(2026, 7, 31), region="West India")
    margin_pct = compute_kpi(df, "Gross Margin %", date(2026, 8, 1), date(2026, 8, 31),
                              date(2026, 7, 1), date(2026, 7, 31), region="West India")
    assert rev.pct_change > 0
    assert margin_pct.abs_change < 0


def test_operational_shift_delivery_days_up_for_south_electronics(df):
    shifts = secondary_metric_shift(df, "South India", "Electronics",
                                     date(2026, 8, 1), date(2026, 8, 31),
                                     date(2026, 7, 1), date(2026, 7, 31))
    assert shifts["delivery_days"]["pct_change"] > 5


# ---------- Retrieval ----------

def test_retrieval_finds_delivery_docs_for_south_electronics(retriever):
    docs = retriever.search("delivery delay shipment", region="South India",
                             category="Electronics", start=date(2026, 8, 1), end=date(2026, 8, 31))
    assert len(docs) > 0
    assert any("delay" in d.text.lower() or "delay" in d.title.lower() for d in docs)


def test_retrieval_empty_for_irrelevant_query(retriever):
    docs = retriever.search("spacecraft launch telemetry anomaly",
                             region="South India", category="Electronics",
                             start=date(2026, 8, 1), end=date(2026, 8, 31))
    assert len(docs) == 0


def test_retrieval_more_docs_in_august_than_july(retriever):
    aug = retriever.count_by_topic("delivery delay shipment", "South India", "Electronics",
                                    date(2026, 8, 1), date(2026, 8, 31))
    jul = retriever.count_by_topic("delivery delay shipment", "South India", "Electronics",
                                    date(2026, 7, 1), date(2026, 7, 31))
    assert aug > jul


# ---------- End-to-end hero scenario ----------

def test_end_to_end_hero_scenario_runs_and_is_grounded(df, retriever):
    result = run_analysis(df, retriever, "Revenue", date(2026, 8, 1), date(2026, 8, 31),
                           "South India", "Electronics", use_llm=False)
    story = result["story"]
    assert story["generated_by"] == "deterministic"
    assert "recommendations" in story
    assert len(story["recommendations"]) >= 1
    # top driver / hypothesis should reference delivery delays given the injected event
    top_conf = [h.confidence for h in result["hypotheses"]]
    assert "High" in top_conf or "Medium" in top_conf


def test_end_to_end_no_llm_key_falls_back_cleanly(df, retriever, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = run_analysis(df, retriever, "Revenue", date(2026, 8, 1), date(2026, 8, 31),
                           None, None, use_llm=True)
    assert result["story"]["generated_by"] == "deterministic"


def test_all_kpis_computable_without_error(df):
    from analytics.kpi_engine import KPI_DEFS
    for kpi in KPI_DEFS:
        result = compute_kpi(df, kpi, date(2026, 8, 1), date(2026, 8, 31),
                              date(2026, 7, 1), date(2026, 7, 31))
        assert result.current_value is not None
