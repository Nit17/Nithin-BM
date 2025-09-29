# 5. Deployment & MLOps

## Scalable Inference Architecture (BATON-CRAFT)
Objectives: minimize tail latency, maximize tokens/sec/GPU, reduce $/1k tokens, ensure safety & reliability.
Pipeline: Ingest → Auth/Quota → Normalize & Hash → Prompt Cache → Retrieval (dense+sparse) → Compression (MMR/summarize) → Batch Scheduler → Inference (speculative or standard) → Stream & Safety Scan → Post-process (citations/format) → Emit → Metrics/Logs/Feedback.
Tactics: dynamic micro-batching, speculative decoding, multi/grouped-query attention, FlashAttention, prefix/KV sharing, quantization, routing.
Observability: latency (p50/p95/p99), queue wait, compute time, tokens/sec, cache hit %, retrieval latency breakdown, safety block rate.
Reliability: circuit breakers, warm standbys, graceful degradation ladder, SLA tenant partitions.
Cost levers: right-size hardware mix, pruning, lazy load, nightly efficiency reports.
Mnemonic: BATON-CRAFT → Batching, Autoscaling, Token efficiency, Orchestration, Networking locality — Caching, Retrieval optimization, Alignment, Feedback loop, Token economics.

## Hosting Strategy (Managed vs Self-Hosted)
Managed (OpenAI/Bedrock/Azure): rapid time-to-value, frontier quality, compliance baked-in, autoscaling; trade cost, deep customization, granular observability, potential lock-in.
Self-Hosted (LLaMA/Mistral/etc.): cost efficiency at scale, customization (fine-tuning, safety stack, kernels), privacy; trade operational burden, talent needs, reliability engineering.
Hybrid: routing layer (small self-hosted for simple, frontier managed for complex), fallback failover, distillation & evaluation harness bridging both.
Decision mnemonic: FACTORS (Fine-tuning depth, Annual token volume, Compliance isolation, Talent, Observability needs, Routing complexity, Sensitivity).
Migration phases: log & label → PEFT fine-tunes → staging benchmarks → canary → gradual shift → periodic re-benchmark frontier models.

---
End of Deployment & MLOps.

---
## Quick Reference
- Pipeline Core: Gateway → Router/Batcher → Retrieval/Context → Inference (speculative) → Safety → Post-process → Metrics.
- Performance Levers: Batching, Speculative Decoding, Quantization, KV Sharing, FlashAttention.
- Reliability Ladder: Fallback model → Reduce max tokens → Disable speculative → Queue shedding.
- Cost Levers: Model routing, quant tiers, cache hit %, utilization tuning.
- Decision (FACTORS): Fine-tune depth, Annual volume, Compliance, Talent, Observability, Routing complexity, Sensitivity.

## Common Pitfalls
| Pitfall | Effect | Mitigation |
|---------|--------|------------|
| Ignoring queue wait | Underestimates latency | Separate queue vs compute metrics |
| Static batch size | Inefficient GPU use | Dynamic micro-batching window |
| No model routing | Overpay for simple queries | Complexity classifier + tiered models |
| Stale caches after deploy | Wrong / unsafe answers | Cache versioning & invalidation hooks |
| Lack of golden set | Silent regressions | Block deploy on threshold failures |

## Interview Checklist
1. Explain speculative decoding mechanics & acceptance criteria.
2. Design autoscaling signals (which metrics?).
3. Strategy for multi-model routing & fallback.
4. Observability suite essentials for LLM API.
5. Migration from managed to hybrid hosting—phases.

## Cross-Links
- Quantization tiers: see [Quantization](06-quantization.md#deployment-patterns).
- Caching layers: see [Caching](08-caching.md#layers).

## Further Reading
- vLLM architecture notes
- Speculative Decoding (LLM.int8() / Medusa / Recurrent Draft)
- FlashAttention papers
- Service Level Objective (SRE) best practices

## Scenario Exercise
- Given 2× latency spike p95 with unchanged compute time, list 5 likely root causes & diagnostics.

---
