# Real-Time Analytics Dashboard + LLM Integration (Power BI + Copilot)

## 1. Problem Statement
Augment real-time BI dashboards with natural language querying, anomaly explanations, and proactive narrative summaries without degrading dashboard refresh SLAs.

## 2. Use Cases
| Use Case | Description |
|----------|-------------|
| NLQ (Natural Language Query) | "Show Q3 revenue by region vs last year" → generate DAX/SQL, run, visualize |
| Anomaly Explanation | Alert triggers → LLM explains potential causes using context slices |
| Narrative Summaries | Hourly auto-generated summary cards appended to dashboard |
| Proactive Insights | LLM suggests drill-downs based on trending KPIs |
| Data Governance Q&A | Ask about metric definitions / lineage |

## 3. Architecture Overview
```
User (Dashboard) → NLQ Service → Query Planner → Semantic Model (Power BI / Tabular) → Execution Engine → Result Cache
      ↘ Vector Store (Metric Definitions, Glossary, Past Queries)
      ↘ Anomaly Detector (stream) → Explanation Pipeline → LLM → Cards API
      ↘ Summary Scheduler → LLM Narrative → Dashboard Tiles
```

## 4. NLQ Flow
1. User NL question → Intent classify (aggregation? comparison? time-series?).
2. Schema grounding: retrieve metric + dimension definitions via vector search over semantic layer metadata.
3. Generate candidate DAX/SQL queries (template + LLM refinement) with guardrails.
4. Validate: static analyzer ensures only allowed tables/columns, WHERE filters safe.
5. Execute; apply row-level security (RLS) filters.
6. Return data + minimal tokens prompt for explanation (optional) → LLM crafts natural language answer.

## 5. Anomaly Explanation Pipeline
- Stream ingestion of KPI metrics (Kafka) → anomaly detection (ESD/Prophet or Twitter AD)
- Anomaly event triggers retrieval: related dimensions (region/product), last N periods, recent notes/incidents.
- LLM prompt: provide concise explanation hypotheses + confidence + recommended drill-downs.
- Store explanation with lineage (anomaly_id) for audit.

## 6. Narrative Summaries
- Scheduler (cron) collects latest KPIs + deltas + anomalies (top 3) + significant segments (Pareto 80/20) → structured JSON.
- LLM template produces summary (limit tokens) → cache; diff vs previous to highlight changes.

## 7. Caching & Performance
- Result cache: keyed by (normalized_query_signature, RLS_segment).
- Semantic layer metadata cache in-memory.
- Async pre-warm of popular morning queries.

## 8. Governance & Security
- Enforce least-privilege RLS inside semantic model; LLM receives only schema subset.
- Metric definition retrieval returns authoritative text; LLM never invents metrics.
- Queries logged with hash; sensitive filters masked.

## 9. Observability
- Metrics: NLQ latency breakdown (parse, retrieval, execution, LLM), accuracy acceptance rate (human validation), anomaly detection precision/recall, summary generation cost, cache hit rate.
- Drift: monitor increase in LLM hallucination corrections.

## 10. Cost Control
- Cap max result rows for explanation (sample + aggregate).
- Reuse embeddings for unchanged metric definitions.
- Temperature=0 for deterministic query gen; smaller model for standard NLQ; larger for complex diagnostics.

## 11. Tech Stack (Example)
| Layer | Choice |
|-------|--------|
| Semantic Model | Power BI Tabular / Azure Analysis Services |
| Storage | Synapse / Snowflake |
| Vector Store | Azure Cognitive Search (hybrid) |
| LLM | Azure OpenAI (GPT) + small local model fallback |
| Anomaly Detection | Prophet / custom z-score streaming |
| Scheduler | Azure Functions / Logic Apps |
| Cache | Redis |

## 12. Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| Incorrect SQL generation | Static validation + dry-run LIMIT 1 |
| Data leakage via prompt | Schema sanitization; no raw PII fields exposed |
| High cost from verbose summaries | Token budgeting; enforce word count |
| Anomaly false positives | Ensemble detectors + feedback loop |
| Latency spikes | Pre-warm cache + parallel retrieval/execution |

## 13. Future Enhancements
- User feedback weighting into query planner ranking.
- Semantic layer lineage graph visualization.
- Domain-specific fine-tuned summarizer.

## 14. Interview Summary
"Layered NLQ service retrieves semantic definitions, safely generates & validates queries, and augments dashboards with anomaly-aware narratives while enforcing RLS and cost controls."