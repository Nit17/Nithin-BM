# Multi-Cloud RAG Pipeline (AWS Bedrock + Azure AI Search)

## 1. Problem Statement
Design a resilient, compliant Retrieval-Augmented Generation system spanning AWS and Azure to leverage Bedrock model endpoints while using Azure AI Search (and optional Cosmos DB) for enterprise document retrieval, supporting regional failover and data residency.

## 2. Goals & Constraints
| Goals | Constraints |
|-------|------------|
| High recall, low latency cross-cloud retrieval | Data residency (EU vs US segregation) |
| Regional failover (RTO < 15m) | Minimize cross-cloud egress cost |
| Consistent grounding & citation enforcement | Zero PII crossing region boundaries |
| Cost visibility & routing optimization | Multi-cloud IAM complexity |

## 3. High-Level Architecture
```
Client → Global DNS (Geo) → Cloud Router
  ├─ AWS Region: API GW → Orchestrator → Bedrock Model (LLM) → Observability
  │     ↘ Async Embedding (Bedrock Titan) / S3 Staging
  └─ Azure Region: Front Door → APIM → Orchestrator → Azure AI Search + OpenAI (fallback) → Monitor

Shared: Event Bus (Schema Manifests), Index Sync Service, Policy Engine, Key Management (separate per cloud), Audit Store.
```

## 4. Data Ingestion & Indexing
1. Documents arrive (SharePoint / S3 upload) → normalized to canonical JSON (doc_id, tenant_id, region, text, metadata).
2. PII classifier & redaction per region before leaving boundary.
3. Dual embedding generation: Primary in-region (Azure OpenAI Embedding) + Secondary (AWS Titan) for shadow eval.
4. Azure AI Search index updated (HNSW vector + filters). Optionally replicate subset to AWS (if legal) via hashed content only.
5. Index sync manifest stores embedding model version + chunk params + checksum.

## 5. Retrieval Flow
1. Query hits geo router → region selection (latency + compliance) → orchestrator.
2. Orchestrator: identify tenant → region policy → query rewriting (spelling, anonymization) → hybrid retrieval (Azure AI Search: vector + BM25) → top-k.
3. Re-rank subset with cross-encoder (Azure OpenAI / local small model) → contextual compression (drop low relevance sentences) → token budget enforcement.
4. Compose prompt with chunk citations → send to chosen model (Bedrock Claude/Sonnet or Azure GPT) based on routing cost & complexity.
5. Enforce citation formatting; post-generation groundedness validator (LLM judge or NLI).

## 6. Cross-Cloud Routing Strategy
- Complexity classifier: if requires advanced reasoning or multi-doc synthesis → Bedrock (Claude) else Azure GPT for simpler queries.
- Latency + Cost Score = w1*pred_latency + w2*token_cost + w3*model_quality_delta.
- Maintain rolling quality score (groundedness, refusal accuracy) per model.

## 7. Security & Compliance
- Separate KMS: AWS KMS & Azure Key Vault; no raw keys shared.
- Regional isolation: EU data never leaves EU region; hashed statistical signals allowed (aggregate metrics only).
- Tenant-level encryption context for stored embeddings.
- Audit trail replicated (event-sourcing) with redacted fields.

## 8. Observability
- Metrics per cloud: retrieval latency breakdown, model latency, groundedness score distribution, citation missing rate, cost per 1k tokens.
- Synthetic canary queries every 5 min comparing answers across clouds; drift alert if semantic similarity < threshold.

## 9. Failover & DR
- Azure outage: route to AWS; if retrieval index unreachable, use last replicated snapshot (read-only, disclaim freshness) or fallback to S3 pre-summarized docs.
- AWS model outage: redirect generation to Azure GPT subset; widen safety thresholds if style differences detected.
- RPO (embedding index): under 1 hour via incremental manifests.

## 10. Cost Optimization
- Cache frequent Q&A prompts (deterministic) in-region.
- Pre-compress large documents into hierarchical summary tree → fewer tokens per query.
- Use smaller embedding dimension (e.g., 768) if recall unaffected (<2% delta) after A/B.
- Adaptive k: start k=8; expand to 12–16 only if confidence classifier < threshold.

## 11. Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| Cross-cloud egress spikes | Keep generation in same region as retrieval; compress payloads |
| Embedding model drift | Shadow dual embedding eval; nightly recall regression |
| Inconsistent safety policies | Central policy version; propagate via manifest; verify hash pre-prompt |
| Latency variance | Adaptive routing with live p95 feed; circuit breakers |
| Citation hallucination | Post-generation groundedness validator + reject/repair loop |

## 12. Future Enhancements
- Introduce on-prem edge cache for high-frequency tenants.
- Multi-index strategic routing (finance index vs general).
- Distill dedicated reranker fine-tuned on domain relevance.

## 13. Interview Summary
"Geo-aware orchestrator performs hybrid retrieval in-region (Azure AI Search) then dynamically selects Bedrock or Azure LLM based on latency, cost, and complexity, enforcing grounding via citations and cross-cloud policy/DR controls."
