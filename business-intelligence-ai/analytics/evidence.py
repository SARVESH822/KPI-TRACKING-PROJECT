"""
Ties deterministic drivers to retrieved unstructured evidence, and assigns
a confidence label. This module NEVER calls an LLM — it produces the
structured "evidence object" that the LLM (or the rule-based fallback
narrator) later turns into prose. This is what keeps numeric claims grounded.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from analytics.decomposition import DriverRow
from retrieval.retriever import DocumentRetriever, RetrievedDoc

# Operational-metric -> search topic mapping used to find corroborating documents.
OPERATIONAL_TOPICS = {
    "delivery_days": "delivery delay shipment late dispatch",
    "discount_rate": "promotion discount campaign",
    "conversion_rate": "conversion funnel checkout",
    "returns_rate": "returns refund complaint",
}


@dataclass
class Hypothesis:
    statement: str
    metric: str
    metric_pct_change: float
    supporting_docs: List[RetrievedDoc] = field(default_factory=list)
    doc_count_current: int = 0
    doc_count_previous: int = 0
    confidence: str = "Low"       # High | Medium | Low
    rationale: str = ""


def _confidence_label(metric_pct_change: float, doc_count_current: int, doc_count_previous: int) -> str:
    """
    Deterministic confidence rule (transparent, not learned):
      - High: strong quantitative shift (>=15% relative move) AND corroborating
        unstructured evidence at least doubled vs. the comparison period.
      - Medium: quantitative shift present AND some corroborating evidence exists,
        but doesn't meet the High bar.
      - Low: quantitative shift alone, with weak/no unstructured corroboration.
    """
    strong_quant = abs(metric_pct_change) >= 15.0
    evidence_ratio = (doc_count_current / doc_count_previous) if doc_count_previous > 0 else (
        2.0 if doc_count_current > 0 else 0.0
    )
    if strong_quant and evidence_ratio >= 1.8 and doc_count_current >= 3:
        return "High"
    if (abs(metric_pct_change) >= 5.0 or doc_count_current >= 2) and doc_count_current >= 1:
        return "Medium"
    return "Low"


def build_hypotheses(op_shifts: dict, retriever: DocumentRetriever, region: Optional[str],
                      category: Optional[str], current_start, current_end,
                      previous_start, previous_end) -> List[Hypothesis]:
    hypotheses = []
    for metric, topic_query in OPERATIONAL_TOPICS.items():
        shift = op_shifts.get(metric)
        if shift is None:
            continue
        pct = shift["pct_change"]
        if abs(pct) < 3.0:  # not a meaningful operational shift, skip
            continue

        cur_docs = retriever.search(topic_query, region, category, current_start, current_end, top_k=15)
        prev_docs = retriever.search(topic_query, region, category, previous_start, previous_end, top_k=15)

        conf = _confidence_label(pct, len(cur_docs), len(prev_docs))
        direction = "increased" if pct > 0 else "decreased"

        hypotheses.append(Hypothesis(
            statement=f"{shift['label']} {direction} {abs(pct):.1f}% and may be a contributing factor.",
            metric=metric,
            metric_pct_change=pct,
            supporting_docs=cur_docs,
            doc_count_current=len(cur_docs),
            doc_count_previous=len(prev_docs),
            confidence=conf,
            rationale=(
                f"{len(cur_docs)} related documents found in the current period vs "
                f"{len(prev_docs)} in the previous period."
            ),
        ))

    hypotheses.sort(key=lambda h: {"High": 0, "Medium": 1, "Low": 2}[h.confidence])
    return hypotheses


def top_driver_summary(drivers: List[DriverRow], additive: bool) -> str:
    if not drivers:
        return "No dimension explained a material share of the change."
    lines = []
    for d in drivers[:3]:
        if additive:
            lines.append(f"{d.member} ({d.dimension}) contributed {d.share_of_change:.0f}% of the change")
        else:
            lines.append(f"{d.member} ({d.dimension}) shifted {d.direction} by {abs(d.contribution):.2f}")
    return "; ".join(lines) + "."
