# Financial Data GenAI: Hallucination Mitigation Architecture

## 1. Problem Statement
Deliver accurate, regulation-compliant financial Q&A / summarization (earnings calls, filings, market data) with minimized hallucinations, explicit citations, and clear abstentions when data is absent.

## 2. Risk Landscape
| Risk | Impact |
|------|--------|
| Fabricated metrics (EPS, revenue) | Regulatory exposure, loss of trust |
| Outdated data usage | Misleading decisions |
| Mis-citation or broken links | Compliance issues |
| Prompt injection in sourced docs | Policy override / leakage |
| Data leakage (MNPI) | Legal liability |

## 3. High-Level Architecture
```
User Query → Policy + Intent Classifier → Retrieval Orchestrator
  → Source Catalog (Filings DB, Earnings Transcripts, Market Data API, Internal Models)
  → Hybrid Retrieval + Freshness Filter + Access Control
  → Re-rank (Cross-Encoder) → Context Compression (numeric preservation)
  → LLM Answer Draft (with required citation schema)
  → Groundedness & Numeric Validation Layer → Repair / Abstain
  → Safety (Compliance Rules) → Final Answer + Citations + Confidence
```

## 4. Data Ingestion & Provenance
- Source fingerprinting: hash(doc_text), store filing type, CIK, timestamp, data vendor ID.
- Structured numeric extraction (XBRL parsing) stored in structured datastore for authoritative numbers.
- Embed both raw text chunks and structured numeric tuples (symbol, metric, period, value).
- Version fields: embedding_model_version, ingestion_version, audit_batch_id.

## 5. Retrieval Strategy
- Hybrid lexical + dense; boost by recency (time decay on filings > 1 year old).
- Numeric-aware reranker (features: presence of requested metric symbol, period alignment, source reliability score).
- Diversity constraint to avoid over-reliance on one document.

## 6. Prompt Pattern (Simplified)
```
SYSTEM: You are a financial assistant. Use ONLY provided context. Cite each fact as [source_id]. If data absent, reply "Data not available".
USER QUESTION: <normalized>
CONTEXT: <chunk_id=..., text=...>
STRUCT_NUMERICS: <symbol,metric,period,value,source_id>
RESPONSE FORMAT (JSON): {"answer": string, "citations": [ ... ], "unsupported_claims": [], "abstained": bool}
```

## 7. Draft → Validate → Repair Loop
1. LLM Draft (temp=0.2) produces JSON.
2. Claim extractor splits answer into atomic claims.
3. For each claim: verify citation presence + NLI entailment vs cited chunk.
4. Numeric validator compares mentioned figures to authoritative structured store (tolerance thresholds, e.g., ±0.1%).
5. Unsupported or mismatched claims removed; if critical missing → ask clarifying question or abstain.
6. Repaired answer re-scored for coherence.

## 8. Guardrails
- Disallow forward-looking speculative statements unless explicitly requested with disclaimer.
- Block MNPI-labeled sources from retrieval for general users.
- Policy engine checks user role (analyst vs public) for access to internal models.

## 9. Observability & Metrics
| Metric | Purpose |
|--------|---------|
| Groundedness@claim | % claims with valid supporting citation |
| Numeric accuracy | % numeric claims within tolerance |
| Abstention rate | Healthy refusal vs over/under abstain balance |
| Recency coverage | % answers using latest filing period |
| Injection block rate | Defense efficacy |

## 10. Evaluation Harness
- Golden set: curated Q&A with verified numeric answers & sources.
- Regression tests: disallow increase in unsupported_claims > threshold.
- Stress tests: ambiguous queries, outdated figure trap (ask for last year's vs current), injection attempts inside context.

## 11. Scaling & Performance
- Pre-normalize numeric tables into vector-friendly text form for retrieval fallback.
- Cache embedding vectors for stable filings; expedite new filing ingestion pipeline (target < 5 min from publish).
- Use smaller guardrail model for claim extraction to reduce latency.

## 12. Tech Stack (Example)
| Layer | Choice |
|-------|--------|
| Embeddings | Domain-tuned E5 variant |
| Vector DB | Qdrant (metadata filters: symbol, period) |
| Structured Store | Postgres or ClickHouse (time-series metrics) |
| LLM | Claude / GPT or tuned open model |
| NLI / Entailment | DeBERTa Large or small fine-tuned model |
| Claim Extractor | Lightweight prompting or distilled classifier |

## 13. Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| Over-abstention | Calibrate threshold; human review of false abstains |
| Latency from validation layers | Parallelize numeric & entailment checks |
| Model drift causing metric hallucination | Nightly eval + rollback to prior model |
| Citation tampering | Store chunk signature; verify before serving |
| Partial context truncation | Context compression with citation preservation |

## 14. Future Enhancements
- Table-aware retrieval (structure preservation).
- Fine-tuned citation generation model.
- Real-time market data streaming integration with refresh triggers.

## 15. Interview Summary
"Mitigate hallucinations by enforcing citation-grounded JSON output, numeric validation against authoritative structured data, entailment-based claim filtering, and abstention when coverage gaps exist—wrapped in a retrieval pipeline tuned for recency and regulatory compliance."
