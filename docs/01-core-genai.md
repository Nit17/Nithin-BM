# 1. Core GenAI Concepts

## 1.1 Foundational GenAI

### Differentiate between Generative AI and Discriminative AI models with examples.

- **Generative** (models p(x) or p(x, y)): learn how data is distributed and can sample new data. Objective: maximize likelihood (e.g., cross-entropy), diffusion loss, adversarial loss.
	- Examples: GPT/LLaMA (text generation), Stable Diffusion/DALL·E (images), VAEs, GANs.
	- Use cases: text/image/audio/code generation, data augmentation, simulation.
	- Pros: leverages unlabeled data, can generate and score sequences; Cons: slower inference, harder evaluation.
- **Discriminative** (models p(y|x) or direct decision boundary f(x)→y): predict labels/scores given inputs. Objective: minimize classification/regression loss.
	- Examples: Logistic Regression, SVM, XGBoost/Random Forest, ResNet classifier, BERT/DeBERTa classifiers, ColBERT/BERT re-rankers.
	- Use cases: classification, NER, QA span extraction, ranking/retrieval, toxicity/safety filters.
	- Pros: efficient and often higher task accuracy with labeled data; Cons: needs labels, cannot generate samples.
- In RAG: the generator is the LLM; discriminators include retrievers, re-rankers, and safety/toxicity classifiers.
- Hybrids: generative LLMs used as discriminators via log-likelihood scoring; pretrain generative, then add a discriminative head for classification.

### What are transformers? How do self-attention and multi-head attention work?

- Transformers: attention-centric sequence models that replace recurrence with parallel attention. Layers = [Self-Attention → Feed-Forward] with residual connections + LayerNorm; add positional info (sinusoidal, learned, or RoPE). Architectures: encoder–decoder (seq2seq), encoder-only (BERT), decoder-only (GPT).
- Self-attention (per layer) for tokens X ∈ R^{n×d_model}:
	- Compute projections: Q = X W_Q, K = X W_K, V = X W_V
	- Weights: A = softmax((Q K^T) / sqrt(d_k) + mask); Output: A V
	- Use padding masks; decoder uses a causal mask (no peeking at future tokens).
- Multi-head attention: run h attention heads in parallel
	- For each head i: head_i = softmax((Q_i K_i^T)/sqrt(d_k)) V_i with its own W_Q^i, W_K^i, W_V^i
	- Concatenate and project: Concat(head_1..head_h) W_O
	- Benefit: different heads capture different relations (positions, syntax, long-range deps).
- Complexity: O(n^2) time/memory in sequence length n; long-context variants use sparse/linear attention.

### Explain tokenization in LLMs. How do Byte Pair Encoding (BPE) and SentencePiece differ?

- Tokenization: converts text to tokens (subwords) that models process. Goals: handle any input (no OOV), keep vocab small, keep sequence lengths reasonable.
- **BPE (Byte Pair Encoding)**:
	- Greedy merges of most frequent symbol pairs until vocab size reached.
	- Often uses whitespace pre-tokenization; GPT-2/RoBERTa use byte-level BPE (operates on bytes 0–255) to avoid OOV entirely.
	- Deterministic segmentation; fast; can over-fragment rare words/morphemes.
- **SentencePiece**:
	- Trains directly on raw text without external pre-tokenization; encodes spaces with a meta symbol (e.g., ▁).
	- Algorithms: Unigram LM (probabilistic) & BPE; Unigram selects optimal subword set via likelihood; supports subword regularization (stochastic sampling) for robustness.
	- Language-agnostic; strong for non-whitespace languages; stable with mixed scripts.
- Differences:
	- Pre-tokenization: BPE often relies on whitespace; SentencePiece handles spaces internally.
	- Algorithm: BPE = greedy merges; Unigram = probabilistic model.
	- Coverage: Byte-level BPE guarantees coverage of any byte; SentencePiece covers Unicode with fallback.
	- Trade-offs: BPE simple/deterministic; Unigram yields fewer tokens and better multilingual generalization.

### What are pre-training, fine-tuning, and instruction-tuning in LLMs?

- **Pre-training (self-supervised)**: next-token prediction (causal) or masked LM over massive unlabeled corpora → broad knowledge.
- **Continual / domain-adaptive pre-training**: keep pretraining on in-domain text to shift distribution before task tuning.
- **Supervised fine-tuning (SFT)**: task/domain-specific labeled (or synthetic) data; objective: cross-entropy; full FT or PEFT.
- **Instruction-tuning**: SFT on instruction–response pairs so model follows natural-language instructions (often multi-task). Followed by preference tuning (RLHF/DPO/ORPO) for tone, helpfulness, safety.
- When to use:
	- Pre-train: only with huge compute/data.
	- Continual pre-train: large domain shift + abundant unlabeled text.
	- Fine-tune: specific tasks/domains with labeled or curated synthetic data.
	- Instruction + preference: behavior, formatting, alignment changes.

### What are embeddings, and how do they help in semantic search and RAG?

- Embeddings: dense vectors encoding semantics of text/code/images so similar meaning clusters.
- Models: sentence-transformers, E5/GTE, OpenAI text-embedding-3, Cohere, multilingual variants (dims 384–3072).
- Similarity: cosine (via normalized dot) or inner product; pick consistent normalization.

**Semantic search workflow:**
1. Ingest: split docs into chunks; compute embeddings; store vectors + metadata (ANN index: HNSW/IVF) in vector DB (FAISS, Qdrant, Pinecone, Weaviate).
2. Query: embed user query; ANN search returns top-k.
3. (Optional) Hybrid: combine BM25/sparse + dense; re-rank with cross-encoder.

**In RAG:**
- Retrieve relevant chunks; optional re-rank/filter/time decay.
- Generate answer grounded in retrieved context → fewer hallucinations + citations.

**Practical tips:**
- Choose domain/language-aligned model; tune chunk size & overlap.
- Store metadata (source, section, timestamp); dedupe; normalize text.
- Re-embed after model upgrade or drift.
- Evaluate retrieval (Recall@k, MRR/NDCG); RAG answer-level (groundedness/faithfulness).

---
End of Core GenAI Concepts.

---
## Quick Reference Cheat Sheet
- Generative vs Discriminative: p(x)/p(x,y) vs p(y|x); sample vs classify.
- Transformer Block: (Multi-Head Self-Attention + FFN) with residual + LayerNorm.
- Tokenization Choice: BPE (deterministic) vs SentencePiece Unigram (probabilistic, better multilingual).
- Training Stack: Pre-train → (Optional continual) → SFT → Instruction Tuning → Preference (RLHF/DPO) → PEFT adapter specialization.
- Embedding Retrieval Loop: Chunk → Embed → Index → Query Embed → ANN Search → (Hybrid fuse) → (Re-rank) → Prompt.

## Common Pitfalls
| Area | Pitfall | Mitigation |
|------|---------|------------|
| Tokenization | Excessive sequence length blow-up | Inspect avg tokens/char; switch model or adjust pre-tokenization |
| Chunking | Too large: recall loss; too small: fragmented context | Grid search size & overlap; hierarchical strategy |
| Fine-Tuning | Catastrophic forgetting | Mix a small slice of general data; regularization |
| Embeddings | Mixed model versions in one index | Version field + dual-write migration |
| Evaluation | Comparing perplexity across tokenizers | Always fix tokenizer/dataset; report both |

## Interview Checklist
1. Explain transformer attention math succinctly.
2. Distinguish BPE vs SentencePiece and when to prefer each.
3. Outline full adaptation pipeline (pre-train → instruction → preference → PEFT).
4. Describe how embeddings power RAG and recall metrics used.
5. Give pros/cons of hierarchical chunking vs flat.

## Cross-Links
- LoRA & PEFT details: see [LLM Mechanics](02-llm-mechanics.md#lora-low-rank-adaptation).
- Retrieval diversification & re-ranking: see [RAG & Agents](03-rag-agents.md#re-ranking-models).
- Groundedness metrics: see [Evaluation & Safety](04-eval-safety.md#hallucination--groundedness).

## Further Reading
- Attention Is All You Need (Vaswani et al.)
- SentencePiece: A simple and language independent subword tokenizer
- Chinchilla Scaling Laws (Hoffmann et al.)

## Practice Prompts
- “Why does Unigram tokenization sometimes produce fewer tokens than BPE?”
- “Design a chunking strategy for legal contracts (multi-level sections).”

---
