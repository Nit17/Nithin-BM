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
