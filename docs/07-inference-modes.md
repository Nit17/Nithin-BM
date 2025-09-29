# 7. Inference Modes (BOLT)

## Online Inference
Low-latency interactive requests (<1–3s). Always-on servers, dynamic micro-batching, streaming, prefix/KV caches, routing by complexity, aggressive observability (p95/p99, queue depth). Hardware: high-perf GPUs (H100/A100). Risk: idle cost, burst handling.

## Batch Inference
High-throughput offline jobs (minutes-hours). Large static batches, sequence packing, spot instances, aggressive quantization, pipeline/data parallelism, checkpointing. Optimize tokens/hour/$ rather than per-request latency.

## Hybrid Patterns
Tiered serving (interactive vs fast-batch vs offline), overflow routing from online to batch queue, precompute embeddings/summaries offline for online reuse, distillation (large batch model → smaller online), feature engineering pipelines feeding real-time.

## Decision Guide
Choose Online: user-facing, unpredictable traffic, conversation state. Choose Batch: large datasets, latency tolerant, cost sensitive. Hybrid: mixed workloads, overflow, preprocessing synergy.

## Mnemonic: BOLT
Batch for Bulk, Online for user Operations, Latency vs throughput trade-offs, Tiered hybrid approaches.

---
End of Inference Modes.

---
## Quick Reference
- Online: latency SLO, dynamic batching, cache heavy, smaller deterministic batches.
- Batch: throughput/cost focus, large static batches, spot instances, sequence packing.
- Hybrid: overflow routing, precompute heavy features, distill large → small.

## Common Pitfalls
| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| Using online infra for batch | Wasted cost | Separate batch queue/cluster |
| No overflow strategy | User latency spikes | Async queue & notify fallback |
| Over-aggressive batch window online | Latency violations | Cap batching delay (e.g. 10–20 ms) |
| Ignoring packing inefficiency | Token waste | Length-bucket & pack sequences |
| Missing cost attribution | Blind optimization | Per-endpoint token & $ metrics |

## Interview Checklist
1. Design metrics to decide routing online vs batch.
2. Sequence packing vs dynamic batching—differences.
3. Overflow handling when GPU saturation occurs.
4. Precompute opportunities for hybrid system.
5. Cost reduction levers for batch cluster.

## Cross-Links
- Speculative decoding: see [Deployment & MLOps](05-deployment-mlops.md#scalable-inference-architecture-baton-craft).
- Token savings via caching: see [Caching](08-caching.md#metrics).

## Further Reading
- vLLM / TensorRT-LLM batching docs
- Ray Serve & Kubernetes batch job patterns

## Exercise
- Create a decision flow to classify a request into online vs fast-batch vs offline.

---
