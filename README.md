# Cynbit InsightIQ — AI-Powered Business Decision Intelligence Engine

*From KPI anomalies to evidence-backed business decisions.*

An AI/ML proof of concept exploring how Cynbit can extend its existing products and enterprise software services through intelligent business analytics.

## Problem Statement

A dashboard shows that revenue dropped by 8% in a particular region. However, it does not explain why the decline occurred or what the business should do next.

Answering these questions often requires analysts to manually investigate sales records, customer complaints, operational reports and other information scattered across multiple systems.

The idea behind InsightIQ is to reduce this gap between identifying a business problem and understanding it.

## Proposed Solution

InsightIQ is a decision intelligence engine that combines structured business data with unstructured information to explain KPI changes, identify likely contributing factors and recommend possible actions.

Rather than relying entirely on an LLM, the system performs numerical analysis using deterministic methods and uses AI to communicate the findings.

Every explanation is supported by available evidence, with confidence levels attached to individual hypotheses. The system also distinguishes correlation from causation to avoid misleading conclusions.

**Detect → Diagnose → Explain → Act → Learn**

The current prototype focuses on detection, diagnosis, explanation and actionable recommendations. Feedback-driven learning is planned as a future extension.

## Why Cynbit?

While exploring Cynbit's website and product portfolio, I noticed that its products serve different industries but share a common need: understanding performance and identifying opportunities for improvement.

Revio deals with customer feedback, TenantEase handles property-related operations, and Cynbit Studio focuses on learning and career development.

Instead of building separate analytical engines for each product, Cynbit could develop a common decision intelligence system and adapt it to different business requirements.

### Potential Applications

| Product / Service | Potential Application |
|---|---|
| Revio | Investigate declining customer ratings by connecting reviews with operational metrics and location-specific performance. |
| TenantEase | Understand changes in rental income, occupancy rates and payment collection. |
| Cynbit Studio | Identify learning bottlenecks and investigate declining engagement or course completion. |
| BitElevate | Analyse recurring technical difficulties and delays in task completion. |
| Digital Marketing | Investigate changes in campaign ROI, conversion rates and customer acquisition costs. |
| Enterprise Software Services | Provide intelligent KPI analysis as an additional feature within customized business applications. |

### Potential Integration with Revio

Revio already provides customer feedback management, sentiment analysis and location-wise reputation monitoring.

InsightIQ could extend these capabilities by connecting customer feedback with operational and financial data.

For example, if a restaurant experiences declining revenue alongside an increase in negative reviews, the system could investigate whether the decline is associated with longer waiting times, delivery delays or product availability.

Instead of simply reporting negative customer sentiment, it could help the business understand the operational factors associated with it.

The current prototype demonstrates this general approach using synthetic retail data. It is not directly integrated with Revio or any other Cynbit product.

### Business Value

- Reduce the time spent manually investigating KPI fluctuations.
- Improve the analytical capabilities of Cynbit's existing products.
- Reuse the core analytics architecture across different industries.
- Make business performance analysis accessible to non-technical users.
- Explore additional revenue opportunities through premium analytics features and enterprise integrations.

These are potential benefits that would need to be validated through real-world implementation and customer feedback.

## Key Features

### 1. Custom Dataset Upload

Users can upload their own structured and unstructured CSV datasets through a column-mapping interface.

The application automatically identifies categorical columns for dimensional analysis. It is not restricted to the schema used in the demonstration dataset.

A bundled synthetic dataset is available for users who want to explore the application without uploading their own data.

### 2. KPI Computation and Change Detection

The system supports multiple business KPIs:

- Revenue
- Orders
- Margin
- Average Order Value (AOV)
- Conversion Rate
- Gross Margin Percentage
- Average Delivery Days

Only KPIs supported by the available dataset columns are displayed.

### 3. Driver and Contribution Analysis

The system breaks down KPI changes across dimensions such as region, category, channel and customer segment.

For additive KPIs, individual contributions sum to the overall change. This ensures that reported numerical contributions are mathematically consistent rather than generated by an LLM.

### 4. Evidence Retrieval

Unstructured information, including support tickets, sales notes, operational reports and management commentary, is retrieved using TF-IDF.

The retrieval process runs locally and does not require an external embedding API or vector database.

### 5. Evidence-Grounded AI Narration

The LLM receives a structured evidence object containing pre-computed KPI values, driver contributions, operational changes and retrieved documents.

Its responsibility is to explain the findings and formulate recommendations, not to perform numerical calculations.

### 6. Explainable Confidence Scoring

Possible explanations are assigned High, Medium or Low confidence levels using transparent rules.

These labels indicate the strength of the available evidence rather than statistically calibrated probabilities or proven causal relationships.

### 7. Deterministic Fallback

The application works even without an LLM API key.

If an API key is unavailable or an external request fails, a template-based narrator generates the KPI explanation using the same analytical results.

This allows the core functionality to operate independently of external AI services.

## System Architecture

```text
Structured Data + Unstructured Data
                 |
                 v
            KPI Engine
                 |
                 v
     Driver Contribution Analysis
                 |
                 v
        Hypothesis Generation
                 |
                 v
         Evidence Retrieval
                 |
                 v
         Confidence Scoring
                 |
                 v
     Structured Evidence Object
                 |
                 v
       Optional LLM Narrator
                 |
          Fallback if needed
                 |
                 v
       Deterministic Narrator
                 
       Narration output from
          either approach
                 |
                 v
           Streamlit UI
```

For a detailed explanation of the pipeline and the separation between deterministic and AI-generated components, refer to [Architecture Documentation](docs/architecture.md).

## AI and Analytics Approach

One of the main design decisions in this project was to separate mathematical analysis from language generation.

LLMs are useful for explaining complex information, but asking them to directly calculate financial metrics or identify numerical contributions introduces unnecessary reliability risks.

The system therefore divides its responsibilities as follows:

| Component | Approach |
|---|---|
| KPI computation | Deterministic |
| Percentage change calculation | Deterministic |
| Driver contribution analysis | Deterministic |
| Operational metric analysis | Deterministic |
| Evidence retrieval | TF-IDF |
| Confidence scoring | Rule-based |
| Business explanation | LLM or deterministic fallback |
| Recommendation wording | LLM-assisted or template-based |

The LLM receives a structured JSON evidence object generated by `analytics/pipeline.run_analysis`.

It does not receive the complete raw dataset or independently calculate KPI values. Its response is validated against a JSON schema before being displayed.

This approach keeps the analytical results auditable while using AI where it adds the most value.

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming | Python | Core application development |
| Analytics | Pandas, NumPy | KPI calculation and dimensional analysis |
| Information Retrieval | Scikit-learn, TF-IDF | Local document retrieval |
| Generative AI | Gemini, Anthropic | Optional explanation and recommendation generation |
| User Interface | Streamlit | Interactive dashboard and dataset upload |
| Testing | Pytest | Unit and end-to-end testing |
| Configuration | Environment variables | API key management |

TF-IDF was selected for the initial implementation because the document corpus is relatively small. It avoids additional infrastructure and allows the retrieval pipeline to operate entirely offline.

Semantic embeddings and vector databases can be introduced as the volume and complexity of documents increase.

## Repository Structure

```text
Cynbit-InsightIQ/
├── app/
│   └── main.py
├── analytics/
│   ├── KPI engine
│   ├── contribution analysis
│   ├── evidence processing
│   └── analysis pipeline
├── retrieval/
│   └── TF-IDF document retrieval
├── llm/
│   ├── deterministic narrator
│   └── optional LLM narrator
├── data/
│   ├── structured/
│   │   └── sales.csv
│   └── unstructured/
│       └── documents.csv
├── scripts/
│   └── generate_data.py
├── tests/
├── docs/
│   ├── architecture.md
│   └── demo-script.md
├── .env.example
├── requirements.txt
├── cynbit.pdf
└── README.md
```

## Installation

Clone the repository and navigate to the project directory.

```bash
git clone <repository-url>
cd Cynbit-InsightIQ
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate the environment.

Windows:

```bash
venv\Scripts\activate
```

Linux or macOS:

```bash
source venv/bin/activate
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

### Optional LLM Configuration

Create a `.env` file using `.env.example` as a reference.

Add either of the following API keys if you want to enable LLM-based narration:

```env
GOOGLE_API_KEY=your_api_key
ANTHROPIC_API_KEY=your_api_key
```

The application can run without either key using its deterministic narration mechanism.

## Running the Application

```bash
streamlit run app/main.py
```

Open the application in your browser:

http://localhost:8501

Users can explore the bundled demonstration dataset or upload their own business data.

## Dataset

The proof of concept uses synthetic datasets with deliberately introduced business events to evaluate whether the system can identify relevant contributing factors.

### Structured Dataset

`data/structured/sales.csv`

- Approximately 15,800 rows.
- Four geographical regions.
- Five product categories.
- Three sales channels.
- Three customer segments.
- Data covering June to August 2026.
- Three intentionally introduced business events.

### Unstructured Dataset

`data/unstructured/documents.csv`

Contains 74 synthetic documents, including:

- Customer support tickets.
- Sales commentary.
- Operational reports.
- Management and finance notes.
- Unrelated documents to evaluate retrieval relevance.

The datasets are generated using a fixed random seed to make the demonstration reproducible.

To regenerate the datasets:

```bash
python scripts/generate_data.py
```

## Demonstration Scenario

The demonstration focuses on a decline in electronics revenue in South India during August.

The dataset contains an intentionally introduced revenue decline of more than 8%.

The system investigates possible contributing factors, including:

- Reduced order volume.
- Increased delivery delays.
- Reduced promotional spending.

Relevant operational records and customer support tickets are retrieved to support the analysis. In this scenario, a distribution centre capacity issue provides additional context for the delivery delays.

The application presents the KPI change, contributing dimensions, supporting documents, confidence levels and recommended actions through an interactive interface.

For the complete walkthrough, refer to [Demo Script](docs/demo-script.md).

## Testing

The repository includes 21 tests covering:

- KPI calculation and edge cases.
- Driver contribution correctness.
- Detection of intentionally introduced business events.
- Document retrieval relevance.
- Complete analytical pipeline execution.
- Narration with and without an LLM API key.
- Dataset upload and column mapping.
- Missing required fields and invalid dates.
- Automatic dimension detection.

Run the test suite using:

```bash
pytest tests/ -v
```

## Proposed Implementation Roadmap for Cynbit

### Phase 1 — Proof of Concept

- Validate the analytical pipeline using synthetic datasets.
- Evaluate KPI calculations and contribution analysis.
- Test document retrieval and explanation quality.
- Demonstrate the workflow through Streamlit.

### Phase 2 — Product-Specific Pilot

- Select an initial product, preferably Revio.
- Identify relevant KPIs and available data sources.
- Adapt the analytical pipeline to product-specific requirements.
- Evaluate generated explanations with domain experts.

### Phase 3 — Product Integration

- Expose the analytics engine through FastAPI endpoints.
- Integrate with existing Cynbit applications and databases.
- Introduce authentication and tenant-level data isolation.
- Collect feedback from business users.

### Phase 4 — Continuous Improvement

- Introduce semantic embeddings and vector-based retrieval.
- Improve hypothesis ranking using validated historical incidents.
- Add real-time KPI monitoring and automated alerts.
- Explore causal inference and predictive analytics.

## Evaluation Strategy

For a future pilot within Cynbit, the system could be evaluated using the following metrics:

| Metric | Evaluation Objective |
|---|---|
| KPI calculation accuracy | Verify results against independently calculated values. |
| Retrieval precision | Measure the relevance of retrieved supporting documents. |
| Hypothesis ranking accuracy | Check whether known contributing factors appear among the top-ranked explanations. |
| Narrative faithfulness | Verify that generated explanations remain consistent with the evidence. |
| Investigation time | Compare AI-assisted investigation with manual analysis. |
| User feedback | Evaluate the usefulness and clarity of generated recommendations. |

These are proposed evaluation metrics for future validation, not measured production results.

## Limitations

- The current demonstration uses synthetic data rather than actual Cynbit customer data.
- TF-IDF relies on textual relevance and does not provide the same semantic understanding as embedding-based retrieval.
- Confidence scoring uses heuristic rules and is not a trained causal inference model.
- Statistical associations and supporting documents do not establish causation.
- The system currently operates as a Streamlit prototype rather than a production-ready SaaS integration.
- The proposed business opportunities are based on publicly available information about Cynbit and would require further validation using internal product requirements.

## Future Improvements

- Semantic search using embedding models and vector databases.
- Causal inference methods such as difference-in-differences.
- FastAPI-based backend integration.
- Real-time anomaly detection and automated notifications.
- What-if analysis to estimate the potential impact of business decisions.
- Human feedback to improve hypothesis ranking.
- Multi-tenant deployment across Cynbit's product ecosystem.

## Project Background

The underlying concept originated from a collaborative business intelligence hackathon project.

For this assessment, the concept has been adapted to explore how decision intelligence could support Cynbit's existing products and enterprise software services.

The accompanying presentation describes the business opportunity, proposed architecture and anticipated impact.

## Project Documentation

- [AI Opportunity Presentation](cynbit.pdf)
- [Technical Architecture](docs/architecture.md)
- [Demonstration Walkthrough](docs/demo-script.md)

## References

- [Cynbit Technologies](https://www.cynbit.in/)
- [Cynbit Product Portfolio](https://www.cynbit.in/products)
- [Cynbit Services](https://www.cynbit.in/services)
- [BitElevate](https://www.cynbit.in/bitelevate)

---

*Decision support, not autonomous decision-making.*

*We are not replacing the dashboard. We are building the intelligence layer that turns dashboards into decisions.*
