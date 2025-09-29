# 8. Caching Strategies (CACHE-STACK)

## Layers
1. Prompt hash (full completion) – deterministic requests (temp=0) with normalized inputs.
2. Prefix / partial (KV state) – system prompt / policy preamble reuse.
3. Session KV cache – fast continuation; sliding window + summarization beyond threshold.
4. Retrieval layer – query→top-k doc IDs (short TTL), embedding content hash.
5. Embedding ingestion – dedupe via content hash before embedding.
6. Rerank scores – (query_id, doc_id) pairs.
7. Tool/function outputs – deterministic expensive calls.
8. Post-processing – formatting, citation restructuring.
9. Semantic (approximate) – embedding similarity reuse with guard validation.

## Deterministic vs Semantic
Deterministic: same key → identical output (temp=0, fixed params). Semantic: high similarity (≥0.92 cosine) + LLM validation for reuse; risk-managed with thresholds & audits.

## Invalidation & Versioning
Triggers: model/ tokenizer / policy / safety pipeline upgrade, doc re-index, tool schema change. Keys include model_version, policy_version, schema_version.

## Metrics
Hit-rate (exact vs semantic), latency saved Σ(baseline - cache), token savings, cost reduction %, semantic false reuse rate (<1%).

## Governance
Sample audits, A/B bypass slice, schema versioning, security (PII redaction, encryption), capacity planning (hot vs warm vs cold tiers), eviction telemetry.

## Mnemonic: CACHE-STACK
Canonicalize, Assign tiers, Control invalidation, Hit-rate monitor, Embedding/semantic layer – Stale-while-revalidate, Token savings metrics, Audit samples, Capacity planning, KV optimization.

---
End of Caching Strategies.

---
## Quick Reference
- Deterministic Keys: model_id + normalized_prompt + decoding_params + context_ids.
- Semantic Threshold: start high (≥0.92 cosine) then tune.
- Invalidation Triggers: model/policy/version shifts, index rebuild, tool schema change.
- Core Metrics: Hit-rate, Token Savings %, False Reuse (semantic), Latency Saved.

## Common Pitfalls
| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| Hash drift (whitespace/order) | Hit-rate collapse | Canonical normalization pipeline |
| Caching non-deterministic outputs | Inconsistent UX | Restrict to temp=0 / fixed sampling |
| Stale unsafe answers post policy update | Safety regression | Versioned keys + flush policy |
| KV cache OOM | Instability | LRU + TTL + size watermark eviction |
| Semantic false positives | Wrong answers reused | LLM validation + high threshold |

## Interview Checklist
1. Explain semantic cache safety controls.
2. Distinguish prefix cache vs session KV.
3. Compute token savings & throughput uplift formula.
4. Design cache observability dashboard.
5. Implement stale-while-revalidate logic.

## Cross-Links
- Retrieval cache synergy: see [RAG & Agents](03-rag-agents.md#chunking-strategies).
- Quantization memory effects: see [Quantization](06-quantization.md#deployment-patterns).

## Further Reading
- Semantic Caching blog posts (OpenAI/LangChain community)
- HTTP cache validation concepts adapted to LLMs

## Exercise
- Outline steps to add semantic caching with guard LLM and rollback switch.

---
