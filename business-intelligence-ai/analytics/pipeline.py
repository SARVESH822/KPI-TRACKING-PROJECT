"""
Top-level pipeline: KPI compute -> decomposition -> evidence/hypotheses ->
narrative (LLM if available, deterministic otherwise). This is the single
entry point the UI calls.
"""
from datetime import date, timedelta
from typing import Optional

from analytics.decomposition import rank_drivers, secondary_metric_shift
from analytics.evidence import build_hypotheses
from analytics.kpi_engine import KPI_DEFS, compute_kpi
from llm.deterministic_narrator import build_story as build_story_deterministic
from llm.llm_narrator import generate_story_with_llm
from retrieval.retriever import DocumentRetriever


def previous_period(start: date, end: date):
    span = (end - start).days + 1
    prev_end = start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=span - 1)
    return prev_start, prev_end


def run_analysis(df, retriever: DocumentRetriever, kpi: str, current_start: date,
                  current_end: date, region: Optional[str], category: Optional[str],
                  use_llm: bool = True, dimensions: Optional[list] = None) -> dict:
    previous_start, previous_end = previous_period(current_start, current_end)

    kpi_result = compute_kpi(df, kpi, current_start, current_end, previous_start,
                              previous_end, region, category)

    additive = KPI_DEFS[kpi]["agg"] == "sum"
    drivers = rank_drivers(df, kpi, current_start, current_end, previous_start,
                            previous_end, region, category, dimensions=dimensions)

    op_shifts = secondary_metric_shift(df, region, category, current_start, current_end,
                                        previous_start, previous_end)

    hypotheses = build_hypotheses(op_shifts, retriever, region, category, current_start,
                                   current_end, previous_start, previous_end)

    period_label = f"{current_start.strftime('%b %Y')}" if current_start.month == current_end.month else f"{current_start} to {current_end}"
    compare_label = f"{previous_start.strftime('%b %Y')}" if previous_start.month == previous_end.month else f"{previous_start} to {previous_end}"

    evidence_object = {
        "kpi": kpi,
        "region": region,
        "category": category,
        "current_period": period_label,
        "previous_period": compare_label,
        "current_value": kpi_result.current_value,
        "previous_value": kpi_result.previous_value,
        "pct_change": kpi_result.pct_change,
        "abs_change": kpi_result.abs_change,
        "drivers": [
            {"dimension": d.dimension, "member": d.member, "contribution": d.contribution,
             "share_of_change_pct": d.share_of_change, "direction": d.direction}
            for d in drivers
        ],
        "operational_shifts": op_shifts,
        "hypotheses": [
            {"statement": h.statement, "confidence": h.confidence, "rationale": h.rationale,
             "doc_count_current": h.doc_count_current, "doc_count_previous": h.doc_count_previous,
             "sample_documents": [
                 {"title": doc.title, "source": doc.source, "date": doc.date}
                 for doc in h.supporting_docs[:5]
             ]}
            for h in hypotheses
        ],
    }

    story = None
    if use_llm:
        story = generate_story_with_llm(evidence_object)
    if story is None:
        story = build_story_deterministic(kpi_result, drivers, hypotheses, region, category,
                                           period_label, compare_label)

    return {
        "kpi_result": kpi_result,
        "drivers": drivers,
        "hypotheses": hypotheses,
        "evidence_object": evidence_object,
        "story": story,
        "additive": additive,
        "period_label": period_label,
        "compare_label": compare_label,
    }
