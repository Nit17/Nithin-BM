# 6. Quantization (SCALE-Q)

## Concepts
Lower precision (INT8 / INT4 / FP8) for weights (and optionally activations/KV cache) to cut memory & bandwidth, raising effective batch/context → lower latency & cost.

## Methods
- Post-training INT8 (dynamic/static), Weight-only INT8/4-bit (GPTQ, AWQ, NF4), FP8 mixed precision, QAT, SmoothQuant (activation redistribution), KV cache quantization.

## Accuracy Considerations
- Layer sensitivity: embeddings, final LM head, layer norms more fragile.
- Typical impact: INT8 near-lossless; 4-bit small perplexity delta (+0.2–0.8) — validate on reasoning & domain tasks.
- Watch cumulative degradation combining long context + low-rank adapters + aggressive quant.

## Evaluation Checklist
1. Perplexity delta (≤ +0.5 target).
2. Task accuracy (≥95–98% baseline).
3. Latency & tokens/sec (cold/warm).
4. Memory footprint & max batch/context.
5. Safety refusal accuracy & toxicity unchanged.
6. Determinism/regression tests (fixed seed) if required.

## Deployment Patterns
- Start INT8 weight-only → validate → move hot endpoints to 4-bit if SLO holds.
- QLoRA: base NF4 + LoRA adapters (FP16); optional merge.
- Mixed tier: FP16/BF16 for premium, INT4/8 for bulk.
- Nightly diff vs reference outputs (semantic + safety).

## Trade-offs & Risks
- Quality regression on reasoning/code; calibration mismatch; limited kernel support; safety drift; maintenance overhead (multiple variants).

## Interactions
- Speculative decoding acceptance can drop if quant noise high.
- LoRA + quant: freeze quant base; train LoRA high precision.
- KV cache quantization for long sessions; evaluate coherence.

## Mnemonic: SCALE-Q
Speed, Cost, Afford larger context, Layer sensitivity, Evaluate thoroughly, Quality trade-off.

---
End of Quantization.

---
## Quick Reference
- Goals: Memory ↓, Bandwidth ↓, Throughput ↑, Context ↑, Cost ↓.
- Typical Order: INT8 weight-only → 4-bit (GPTQ/AWQ) → KV cache quant (if memory bound).
- Evaluate: Perplexity, Task Accuracy, Latency, Memory, Safety.
- Interactions: Speculative acceptance, LoRA adapters, KV reuse.

## Common Pitfalls
| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| Poor calibration set | Large accuracy drop | Diverse representative sample |
| Over-aggressive 4-bit on reasoning tasks | Logic errors | Keep sensitive layers higher precision |
| Ignoring safety metrics | Hidden refusal drift | Re-run safety suite post-quant |
| Multiple quant variants unmanaged | Artifact sprawl | Registry with metadata (model hash, method, date) |
| KV quant without eval | Coherence loss long chats | A/B conversation transcripts |

## Interview Checklist
1. Why scaling factor important in quant groups?
2. Compare GPTQ vs AWQ core idea.
3. Trade-offs INT8 vs FP8 vs 4-bit NF4.
4. Steps to add QLoRA to existing pipeline.
5. Detect subtle reasoning regression post-quant.

## Cross-Links
- Deployment memory planning: see [Deployment & MLOps](05-deployment-mlops.md#scalable-inference-architecture-baton-craft).
- Cache memory pressure interplay: see [Caching](08-caching.md#layers).

## Further Reading
- GPTQ paper, AWQ paper
- SmoothQuant, QLoRA
- NVIDIA FP8 whitepapers

## Exercise
- Propose evaluation matrix for approving 4-bit model into production.

---
