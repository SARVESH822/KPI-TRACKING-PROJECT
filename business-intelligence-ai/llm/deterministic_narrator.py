"""
Rule-based narrative generator. Produces the same structured "KPI story"
shape as the LLM-enriched path, using only string templates over already-
computed numbers. This is the guaranteed fallback: the product must never
be fully dependent on an external API to demonstrate its core value.
"""
from typing import Optional

from analytics.decomposition import DriverRow
from analytics.evidence import Hypothesis
from analytics.kpi_engine import KpiResult

FMT = {
    "currency": lambda v: f"₹{v:,.0f}",
    "int": lambda v: f"{v:,.0f}",
    "percent": lambda v: f"{v*100:.2f}%" if abs(v) <= 1 else f"{v:.2f}%",
    "days": lambda v: f"{v:.2f} days",
}


def format_value(v: float, fmt: str) -> str:
    return FMT.get(fmt, lambda x: f"{x:,.2f}")(v)


def build_story(kpi_result: KpiResult, drivers: list[DriverRow], hypotheses: list[Hypothesis],
                 region: Optional[str], category: Optional[str], period_label: str,
                 compare_label: str) -> dict:
    scope = " / ".join([x for x in [region, category] if x and "All" not in x]) or "All regions/categories"
    direction = "increased" if kpi_result.abs_change > 0 else "decreased"
    val_str = format_value(kpi_result.current_value, kpi_result.fmt)

    executive_summary = (
        f"{kpi_result.kpi} in {scope} {direction} {abs(kpi_result.pct_change):.1f}% "
        f"in {period_label} compared with {compare_label}, reaching {val_str}."
    )

    what_changed = (
        f"{kpi_result.kpi} moved from {format_value(kpi_result.previous_value, kpi_result.fmt)} "
        f"({compare_label}) to {format_value(kpi_result.current_value, kpi_result.fmt)} "
        f"({period_label}), a change of {format_value(kpi_result.abs_change, kpi_result.fmt)} "
        f"({kpi_result.pct_change:+.1f}%)."
    )

    why_lines = []
    for d in drivers[:5]:
        why_lines.append(
            f"{d.member} ({d.dimension}): {format_value(d.contribution, kpi_result.fmt)} "
            f"({d.share_of_change:+.0f}% of total change)"
        )

    evidence_lines = []
    for h in hypotheses:
        evidence_lines.append(
            f"[{h.confidence} confidence] {h.statement} {h.rationale}"
        )

    recommendations = []
    for h in hypotheses:
        if h.confidence in ("High", "Medium"):
            action = {
                "delivery_days": "Investigate fulfillment and dispatch capacity in the affected region.",
                "discount_rate": "Review promotional spend allocation for the affected category/region.",
                "conversion_rate": "Audit the conversion funnel for the affected scope, especially on mobile/app traffic.",
                "returns_rate": "Investigate product quality or listing-accuracy issues driving returns.",
            }.get(h.metric, "Investigate this operational metric further.")
            recommendations.append({
                "action": action,
                "reason": h.statement,
                "evidence": h.rationale,
                "priority": "High" if h.confidence == "High" else "Medium",
                "confidence": h.confidence,
            })

    if not recommendations:
        recommendations.append({
            "action": "Monitor the metric for another period before taking action.",
            "reason": "No strong supporting evidence was found for a specific operational cause.",
            "evidence": "No operational or unstructured signal cleared the confidence threshold.",
            "priority": "Low",
            "confidence": "Low",
        })

    return {
        "executive_summary": executive_summary,
        "what_changed": what_changed,
        "why_it_changed": why_lines,
        "supporting_evidence": evidence_lines,
        "recommendations": recommendations,
        "generated_by": "deterministic",
    }
