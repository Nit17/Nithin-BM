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
