# Architecture

```
Structured Data (data/structured/sales.csv)
        │
        ▼
KPI Engine (analytics/kpi_engine.py)          <- deterministic aggregation
        │
        ▼
Decomposition & Contribution Analysis          <- deterministic arithmetic,
(analytics/decomposition.py)                     contributions sum to total change
        │
        ▼
Operational Metric Shift Detection             <- delivery days, discount rate,
(analytics/decomposition.py)                     conversion, returns — deterministic
        │
        ▼                                      Unstructured Data
Hypothesis Generation                          (data/unstructured/documents.csv)
(analytics/evidence.py)  ◄───────────────────────────────┐
        │                                                 │
        ▼                                                 │
Evidence Retrieval (retrieval/retriever.py, TF-IDF) ───────┘
        │
        ▼
Confidence Scoring (deterministic rule, analytics/evidence.py)
        │
        ▼
Structured "Evidence Object" (JSON)
        │
        ├──► LLM Narrator (llm/llm_narrator.py) — optional, schema-validated
        │        │ (on failure/timeout/missing key)
        │        ▼
        └──► Deterministic Narrator (llm/deterministic_narrator.py) — guaranteed fallback
                 │
                 ▼
        Streamlit UI (app/main.py)
```

## Why these choices

- **No vector database.** The unstructured corpus for this prototype is ~70-100
  documents. TF-IDF + cosine similarity (`scikit-learn`) gives reliable, fully
  offline retrieval with zero infrastructure and zero external calls. A vector
  DB (Chroma/pgvector) is the natural upgrade once corpus size and query
  variety justify the added complexity — noted in Limitations.
- **LLM is narration-only.** The LLM never receives raw transaction rows and
  never computes a number. It receives a structured evidence object and
  rewrites it into prose, with a system prompt that forbids inventing figures
  and forbids causal language beyond "associated with" / "may contribute to".
  Its JSON output is schema-validated before being shown; on any failure the
  app falls back to `llm/deterministic_narrator.py`, which produces the same
  structure from string templates with no dependency on any external service.
- **Confidence is a transparent rule, not a model.** See
  `analytics/evidence._confidence_label` — it's an explicit function of
  (quantitative shift size, doc-count ratio between periods), so it can be
  explained to a judge in one sentence and audited.
- **Streamlit single-process app.** A separate FastAPI backend was considered
  but adds a second process to keep in sync for marginal benefit at this
  scale; the analytics/retrieval/llm modules are pure Python with no Streamlit
  dependency, so a FastAPI layer can be added later without rewriting logic.

## What is deterministic vs. AI-generated

| Component | Deterministic | AI-generated |
|---|---|---|
| KPI values, % change | ✅ | |
| Driver contributions | ✅ | |
| Operational metric shifts | ✅ | |
| Confidence label | ✅ | |
| Document retrieval | ✅ (TF-IDF, no LLM) | |
| Narrative prose | fallback available | ✅ when API key present |
| Recommendation wording | fallback available | ✅ when API key present |
| Recommendation *trigger* (which hypotheses become recommendations) | ✅ | |

## Known limitations

- Retrieval is TF-IDF, not semantic embeddings — exact wording match matters
  more than it would with an embedding model.
- Contribution analysis is first-order (single-dimension decomposition); it
  does not model interaction effects between dimensions.
- Confidence labels are a transparent heuristic, not a statistically fitted
  causal-inference model — by design, to avoid false certainty in a prototype.
- The synthetic dataset is generated with fixed random seeds for demo
  reproducibility; it does not reflect a real client's data distribution.
