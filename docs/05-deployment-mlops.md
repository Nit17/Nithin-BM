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
