import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from analytics.data_ingestion import (NUMERIC_STRUCTURED, OPTIONAL_UNSTRUCTURED,
                                       load_uploaded_csv, validate_and_map_structured,
                                       validate_and_map_unstructured)
from analytics.decomposition import detect_dimensions
from analytics.kpi_engine import KPI_DEFS, load_structured_data
from analytics.pipeline import run_analysis
from llm.deterministic_narrator import format_value
from retrieval.retriever import DocumentRetriever, load_documents

st.set_page_config(page_title="BusinessIntelligence.ai — KPI Intelligence",
                    layout="wide", initial_sidebar_state="expanded")

# ---------------- Minimal, restrained styling (no gradients/neon/chat UI) ----------------
st.markdown("""
<style>
:root { --accent: #5B2A86; }
.block-container { padding-top: 1.6rem; max-width: 1200px; }
h1, h2, h3 { font-weight: 600; }
.kpi-card { border: 1px solid #E4E4E7; border-radius: 8px; padding: 1.1rem 1.3rem; background: #FFFFFF; }
.kpi-value { font-size: 2.0rem; font-weight: 700; }
.kpi-delta-down { color: #B91C1C; font-weight: 600; }
.kpi-delta-up { color: #15803D; font-weight: 600; }
.section-label { font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase;
                  color: #6B6B76; font-weight: 600; margin-top: 1.4rem; margin-bottom: 0.4rem; }
.conf-High { color: #15803D; font-weight: 600; }
.conf-Medium { color: #B45309; font-weight: 600; }
.conf-Low { color: #6B6B76; font-weight: 600; }
.pill { display:inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem;
        font-weight: 600; margin-right: 6px; }
.pill-high { background:#DCFCE7; color:#15803D; }
.pill-medium { background:#FEF3C7; color:#B45309; }
.pill-low { background:#F1F1F3; color:#6B6B76; }
.evidence-doc { border-left: 3px solid var(--accent); padding: 0.4rem 0.8rem; margin-bottom: 0.5rem;
                 background: #FAFAFB; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def _load_sample_data():
    df = load_structured_data(str(ROOT / "data" / "structured" / "sales.csv"))
    docs_df = load_documents(str(ROOT / "data" / "unstructured" / "documents.csv"))
    return df, docs_df


@st.cache_resource
def _get_retriever(docs_df):
    return DocumentRetriever(docs_df)


st.title("BusinessIntelligence.ai")
st.caption("KPI Intelligence — explains what changed, why, and what to do next")

# ---------------- Data source selection ----------------
with st.sidebar:
    st.markdown("### Data Source")
    data_source = st.radio("Choose data source", ["Use sample data (synthetic demo)", "Upload my own data"],
                            index=0, label_visibility="collapsed")

    custom_dims = None
    if data_source == "Use sample data (synthetic demo)":
        df, docs_df = _load_sample_data()
        st.success(f"Sample data loaded: {len(df):,} structured rows, {len(docs_df):,} documents.")
    else:
        st.markdown("**1. Structured data (required)**")
        struct_file = st.file_uploader("Upload structured CSV (sales/transactions)", type=["csv"], key="struct")
        df = None
        if struct_file is not None:
            raw_struct = load_uploaded_csv(struct_file)
            if raw_struct is None:
                st.error("Could not read that CSV. Please check the file format.")
            else:
                st.caption("Map your columns to the fields below. Leave unmapped fields blank.")
                cols = ["-- none --"] + list(raw_struct.columns)
                date_col = st.selectbox("Date column *", cols, key="s_date")
                measure_map = {}
                st.caption("Numeric measures (map at least one):")
                mc1, mc2 = st.columns(2)
                for i, m in enumerate(NUMERIC_STRUCTURED):
                    with (mc1 if i % 2 == 0 else mc2):
                        sel = st.selectbox(m, cols, key=f"s_measure_{m}")
                        if sel != "-- none --":
                            measure_map[m] = sel
                st.caption("Dimension columns (optional, e.g. region/category/channel — any column names work):")
                dim_cols_selected = st.multiselect("Dimension columns", list(raw_struct.columns), key="s_dims")

                if st.button("Validate & Load Structured Data"):
                    column_map = dict(measure_map)
                    if date_col != "-- none --":
                        column_map["date"] = date_col
                    for dc in dim_cols_selected:
                        column_map[f"extra_dim:{dc}"] = dc
                    result = validate_and_map_structured(raw_struct, column_map)
                    if result.errors:
                        for e in result.errors:
                            st.error(e)
                    else:
                        for w in result.warnings:
                            st.warning(w)
                        st.session_state["uploaded_struct_df"] = result.df
                        custom_dims = result.detected_dimensions
                        st.session_state["uploaded_dims"] = custom_dims
                        st.success(f"Loaded {len(result.df):,} rows with dimensions: {', '.join(custom_dims) or 'none'}")

        df = st.session_state.get("uploaded_struct_df")
        custom_dims = st.session_state.get("uploaded_dims")

        st.markdown("**2. Unstructured data (optional)**")
        unstruct_file = st.file_uploader("Upload unstructured CSV (tickets/notes)", type=["csv"], key="unstruct")
        docs_df = None
        if unstruct_file is not None:
            raw_unstruct = load_uploaded_csv(unstruct_file)
            if raw_unstruct is None:
                st.error("Could not read that CSV.")
            else:
                cols_u = ["-- none --"] + list(raw_unstruct.columns)
                u_date = st.selectbox("Date column *", cols_u, key="u_date")
                u_text = st.selectbox("Text column *", cols_u, key="u_text")
                u_optional_map = {}
                for f in OPTIONAL_UNSTRUCTURED:
                    sel = st.selectbox(f"{f} (optional)", cols_u, key=f"u_{f}")
                    if sel != "-- none --":
                        u_optional_map[f] = sel
                if st.button("Validate & Load Unstructured Data"):
                    column_map = dict(u_optional_map)
                    if u_date != "-- none --":
                        column_map["date"] = u_date
                    if u_text != "-- none --":
                        column_map["text"] = u_text
                    result = validate_and_map_unstructured(raw_unstruct, column_map)
                    if result.errors:
                        for e in result.errors:
                            st.error(e)
                    else:
                        for w in result.warnings:
                            st.warning(w)
                        st.session_state["uploaded_docs_df"] = result.df
                        st.success(f"Loaded {len(result.df):,} documents.")
        docs_df = st.session_state.get("uploaded_docs_df")

        if df is None:
            st.info("Upload and validate structured data to continue.")
            st.stop()
        if docs_df is None:
            docs_df = pd.DataFrame(columns=["document_id", "date", "region", "category",
                                             "source", "title", "text"])
            st.caption("No unstructured data loaded — evidence retrieval will return no results.")

retriever = _get_retriever(docs_df)
active_dimensions = custom_dims if (data_source != "Use sample data (synthetic demo)" and custom_dims) else None

def _kpi_available(kpi: str, df: pd.DataFrame) -> bool:
    spec = KPI_DEFS[kpi]
    if spec["agg"] == "sum" or spec["agg"] == "mean":
        return spec["col"] in df.columns
    if spec["agg"] == "aov":
        return "revenue" in df.columns and "orders" in df.columns
    if spec["agg"] == "margin_pct":
        return "revenue" in df.columns and "margin" in df.columns
    return False

available_kpis = [k for k in KPI_DEFS if _kpi_available(k, df)]
data_min, data_max = df["date"].min().date(), df["date"].max().date()

# ---------------- Executive overview (sample data only — fixed hero period) ----------------
if data_source == "Use sample data (synthetic demo)":
    st.markdown('<div class="section-label">Executive Overview — August 2026 vs July 2026</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    overview_kpis = [k for k in ["Revenue", "Orders", "Conversion Rate", "Gross Margin %"] if k in available_kpis]
    from analytics.kpi_engine import compute_kpi
    for col, kpi_name in zip(cols, overview_kpis):
        r = compute_kpi(df, kpi_name, date(2026, 8, 1), date(2026, 8, 31), date(2026, 7, 1), date(2026, 7, 31))
        delta_class = "kpi-delta-down" if r.pct_change < 0 else "kpi-delta-up"
        arrow = "↓" if r.pct_change < 0 else "↑"
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div style="color:#6B6B76;font-size:0.85rem;">{kpi_name}</div>
                <div class="kpi-value">{format_value(r.current_value, r.fmt)}</div>
                <div class="{delta_class}">{arrow} {abs(r.pct_change):.1f}% vs Jul</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")

# ---------------- KPI story controls ----------------
st.markdown('<div class="section-label">Generate a KPI Story</div>', unsafe_allow_html=True)

dims_for_ui = active_dimensions if active_dimensions is not None else detect_dimensions(df)

c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
with c1:
    kpi = st.selectbox("KPI", available_kpis, index=0)
with c2:
    region = (st.selectbox("Region", ["All Regions"] + sorted(df["region"].dropna().unique().tolist()))
              if "region" in df.columns else None)
with c3:
    category = (st.selectbox("Category", ["All Categories"] + sorted(df["category"].dropna().unique().tolist()))
                if "category" in df.columns else None)
with c4:
    if data_source == "Use sample data (synthetic demo)":
        period = st.selectbox("Period", ["August 2026 vs July 2026", "July 2026 vs June 2026"])
        current_start, current_end = ((date(2026, 8, 1), date(2026, 8, 31)) if period == "August 2026 vs July 2026"
                                       else (date(2026, 7, 1), date(2026, 7, 31)))
    else:
        current_start, current_end = st.date_input(
            "Current period", value=(data_max - pd.Timedelta(days=29), data_max),
            min_value=data_min, max_value=data_max)

use_llm = st.checkbox("Enable AI narrative enrichment (falls back automatically if unavailable)", value=True)

generate = st.button("Generate Story", type="primary")

if generate:
    with st.spinner("Running analysis..."):
        result = run_analysis(df, retriever, kpi, current_start, current_end,
                               region, category, use_llm=use_llm, dimensions=dims_for_ui)
    st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]
    story = result["story"]
    kpi_result = result["kpi_result"]
    drivers = result["drivers"]
    hypotheses = result["hypotheses"]

    badge = "🤖 AI-enriched narrative" if story["generated_by"] == "llm" else "⚙️ Deterministic narrative (no LLM used)"
    st.caption(badge)

    tab_overview, tab_drivers, tab_evidence, tab_actions = st.tabs(
        ["Why did it change?", "Top Drivers", "Supporting Evidence", "Recommended Actions"])

    with tab_overview:
        st.subheader("Executive Summary")
        st.write(story["executive_summary"])
        st.subheader("What Changed")
        st.write(story["what_changed"])
        st.subheader("Why It Changed")
        for line in story["why_it_changed"]:
            st.markdown(f"- {line}")

    with tab_drivers:
        if drivers:
            ddf = pd.DataFrame([{
                "Dimension": d.dimension, "Member": d.member,
                "Current": d.current_value, "Previous": d.previous_value,
                "Contribution": d.contribution, "Share of Change (%)": round(d.share_of_change, 1),
            } for d in drivers])
            st.dataframe(ddf, use_container_width=True, hide_index=True)
            st.bar_chart(ddf.set_index("Member")["Contribution"])
        else:
            st.info("No dimension explained a material share of the change for this scope.")

    with tab_evidence:
        if not hypotheses:
            st.info("No operational shift met the threshold to search for supporting unstructured evidence.")
        for h in hypotheses:
            pill_class = {"High": "pill-high", "Medium": "pill-medium", "Low": "pill-low"}[h.confidence]
            st.markdown(f'<span class="pill {pill_class}">{h.confidence} confidence</span> {h.statement}',
                        unsafe_allow_html=True)
            st.caption(h.rationale)
            for doc in h.supporting_docs[:4]:
                st.markdown(f"""
                <div class="evidence-doc"><b>{doc.title}</b> — {doc.source}, {doc.date}<br>{doc.text}</div>
                """, unsafe_allow_html=True)
            st.markdown("")

    with tab_actions:
        for rec in story["recommendations"]:
            pill_class = {"High": "pill-high", "Medium": "pill-medium", "Low": "pill-low"}[rec["priority"]]
            st.markdown(f'<span class="pill {pill_class}">{rec["priority"]} priority</span> **{rec["action"]}**',
                        unsafe_allow_html=True)
            st.caption(f"Reason: {rec['reason']}")
            st.caption(f"Evidence: {rec['evidence']} | Confidence: {rec['confidence']}")
            st.markdown("")
else:
    st.info("Select a KPI, scope, and period above, then click **Generate Story**.")

st.markdown("---")
st.caption("Decision support, not autonomous decision-making. All numeric claims are computed deterministically; "
           "the AI layer only narrates and never invents figures.")
