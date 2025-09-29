# 2. LLM Mechanics

## Positional Encoding

Why needed: self-attention is order-agnostic; positions inject sequence order.
- Sinusoidal (absolute): PE[pos,2i] = sin(pos/10000^(2i/d)), PE[pos,2i+1] = cos(...); no params; extrapolates.
- Learned absolute: trainable position vectors; simple, limited extrapolation unless interpolated.
- Relative variants:
	- Shaw et al. biases a(i−j) added to logits.
	- RoPE: rotary transformation of Q,K enabling relative positioning; strong long-context performance.
	- ALiBi: linear head-specific bias m_h·(i−j); cheap & extrapolates.
Practical: apply once (decoder-only) or per layer; use padding/causal masks; for long context use RoPE scaling or NTK-aware interpolation.

## Causal (Decoder-only) vs Seq2Seq

Architecture:
- Decoder-only: single stack, causal mask; examples GPT, LLaMA, Mistral.
- Seq2seq: encoder (bidirectional) + decoder (causal + cross-attention); examples T5, BART, mT5.
Training:
- Decoder-only: next-token prediction on concatenated text.
- Seq2seq: supervised mapping or denoising objective (span corruption).
Strengths:
- Decoder-only: open-ended generation, long prompts, simple serving.
- Seq2seq: strong conditioning on input, efficient prompts, structured transforms.
Trade-offs:
- Decoder-only: long prompts cost; all conditioning in-context.
- Seq2seq: extra encode pass latency + params.
Use cases:
- Chat/RAG/agents → decoder-only.
- Translation, summarization, structured extraction → seq2seq.

## Perplexity

Definition: PPL = exp( −(1/N) Σ log p(x_t | x_<t) ). Lower better.
Intuition: effective branching factor.
Uses: monitor training progress, domain shift detection, baseline quality.
Limitations: not aligned with instruction following/helpfulness; not comparable across tokenizers.
Report: dataset, tokenizer, normalization, whether per token or per word.

## LoRA (Low-Rank Adaptation)

Idea: Freeze W, learn ΔW = s·B A with rank r ≪ min(d_in, d_out). Forward: W'x = Wx + s·B(Ax).
Where: attention projections (Q,K,V,O) and sometimes MLPs.
Benefits: <1% params trainable, 10–100× lower optimizer memory; stackable adapters; compatible with 4-bit bases (QLoRA).
Hyperparams: rank r (4–64), scaling α, lora_dropout; target impactful layers first.
Trade-offs: slightly lower ceiling on large domain shifts; selection of layers/rank critical; multiple adapters may require calibration.

## PEFT vs Full Fine-Tuning

Full FT:
- Update all weights; max capacity; high compute & memory (optimizer states ~2–3× params); risk of catastrophic forgetting.

PEFT Methods:
- LoRA/QLoRA, Adapters, Prefix/Prompt Tuning, BitFit (bias-only).
Benefits: drastically reduced memory, multiple tasks share base, easier multi-tenant deployment.
Trade-offs: small performance gap on very large shifts; careful layer targeting & rank tuning needed.
Deployment: load base once; hot-swap adapters; optional merge for inference.
When: most downstream tasks; resource-constrained or multi-tenant; prefer full FT only for deep domain changes.

---
End of LLM Mechanics.

---
## Quick Reference
- Positional Families: Sinusoidal (no params), Learned (flexible), RoPE (rotary relative), ALiBi (bias), Shaw Relative (additive).
- Decoder-only vs Seq2Seq: single causal stack vs encoder+decoder w/ cross-attention.
- Perplexity: exp(mean token NLL); only comparable with same tokenizer+dataset.
- LoRA Formula: W' = W + s·B A (rank r ≪ dims). Train only A,B.
- PEFT Menu: LoRA, Adapters, Prefix/Prompt Tuning, BitFit.

## Common Pitfalls
| Topic | Pitfall | Mitigation |
|-------|---------|------------|
| Positional Scaling | RoPE extrapolation failure at long context | NTK-aware scaling / interpolation |
| Perplexity Use | Over-indexing on PPL for instruction tasks | Add task & human eval metrics |
| LoRA Tuning | Too high rank causing overfit | Sweep r & monitor validation loss | 
| PEFT Deployment | Adapter merge without version tagging | Tag adapter + base hash; checksum post-merge |
| Mixed Precision | Silent overflow when forcing FP16 | Prefer BF16; monitor loss spikes |

## Interview Checklist
1. Derive why softmax(QK^T/√d) needs scaling factor.
2. Compare relative position approaches (RoPE vs ALiBi trade-offs).
3. When does full fine-tune beat LoRA significantly?
4. How to detect overfitting in SFT beyond PPL?
5. Steps to safely merge multiple LoRA adapters.

## Cross-Links
- Quantization interplay w/ LoRA: see [Quantization](06-quantization.md#interactions).
- Memory efficiency & batching: see [Deployment & MLOps](05-deployment-mlops.md#scalable-inference-architecture-baton-craft).

## Further Reading
- RoFormer (RoPE introduction)
- ALiBi paper & blog posts
- LoRA original paper (Hu et al.)
- DPO (Direct Preference Optimization)

## Mini Exercises
- Explain difference between low perplexity and good instruction adherence.
- Given GPU VRAM constraints, outline LoRA + QLoRA setup.

---
