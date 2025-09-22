

## 1. Core GenAI Concepts

### Foundational GenAI
- Differentiate between Generative AI and Discriminative AI models with examples.

- Generative (models p(x) or p(x, y)): learn how data is distributed and can sample new data. Objective: maximize likelihood (e.g., cross-entropy), diffusion loss, adversarial loss.
    - Examples: GPT/LLaMA (text generation), Stable Diffusion/DALL·E (images), VAEs, GANs.
    - Use cases: text/image/audio/code generation, data augmentation, simulation.
    - Pros: leverages unlabeled data, can generate and score sequences; Cons: slower inference, harder evaluation.
  - Discriminative (models p(y|x) or direct decision boundary f(x)→y): predict labels/scores given inputs. Objective: minimize classification/regression loss.
    - Examples: Logistic Regression, SVM, XGBoost/Random Forest, ResNet classifier, BERT/DeBERTa classifiers, ColBERT/BERT re-rankers.
    - Use cases: classification, NER, QA span extraction, ranking/retrieval, toxicity/safety filters.
    - Pros: efficient and often higher task accuracy with labeled data; Cons: needs labels, cannot generate samples.
  - In RAG: the generator is the LLM; discriminators include retrievers, re-rankers, and safety/toxicity classifiers.
  - Hybrids: generative LLMs used as discriminators via log-likelihood scoring; pretrain generative, then add a discriminative head for classification.
  
- What are transformers? How do self-attention and multi-head attention work?
  - Transformers: attention-centric sequence models that replace recurrence with parallel attention. Layers = [Self-Attention → Feed-Forward] with residual connections + LayerNorm; add positional info (sinusoidal, learned, or RoPE). Architectures: encoder–decoder (seq2seq), encoder-only (BERT), decoder-only (GPT).
  - Self-attention (per layer): for tokens X ∈ R^{n×d_model}
    - Compute projections: `Q = X W_Q`, `K = X W_K`, `V = X W_V`
    - Weights: `A = softmax((Q K^T) / sqrt(d_k) + mask)`; Output: `A V`
    - Use padding masks; decoder uses a causal mask (no peeking at future tokens).
  - Multi-head attention: run h attention “heads” in parallel
    - For each head i: `head_i = softmax((Q_i K_i^T)/sqrt(d_k)) V_i` with its own `W_Q^i, W_K^i, W_V^i`
    - Concatenate and project: `Concat(head_1..head_h) W_O`
    - Benefit: different heads capture different relations (positions, syntax, long-range deps).
  - Notes: O(n^2) time/memory in sequence length n; long-context variants use sparse/linear attention.

- Explain tokenization in LLMs. How do Byte Pair Encoding (BPE) and SentencePiece differ?

- Explain tokenization in LLMs. How do Byte Pair Encoding (BPE) and SentencePiece differ?
  - Tokenization: converts text to tokens (subwords) that models process. Goals: handle any input (no OOV), keep vocab small, and keep sequence lengths reasonable.
  - BPE (Byte Pair Encoding):
    - Greedy merges of most frequent symbol pairs until vocab size is reached.
    - Often uses whitespace pre-tokenization; GPT-2/RoBERTa use byte-level BPE (operates on bytes 0–255) to avoid OOV entirely.
    - Deterministic segmentation; fast; can over-fragment rare words/morphemes.
  - SentencePiece:
    - Toolkit that trains directly on raw text without external pre-tokenization; encodes spaces with a meta symbol (e.g., ▁).
    - Supports two algorithms: Unigram LM (common: T5, LLaMA) and BPE; Unigram selects an optimal subword set via likelihood and enables subword regularization (stochastic sampling) for robustness.
    - Language-agnostic, good for languages without spaces; stable handling of mixed scripts.
  - Differences in practice:
    - Pre-tokenization: BPE often relies on whitespace; SentencePiece does not (handles spaces internally).
    - Algorithm: BPE = greedy merges; SentencePiece Unigram = probabilistic model with optional sampling.
    - Coverage: Byte-level BPE guarantees coverage of any byte; SentencePiece covers Unicode with learned pieces and byte-fallback if enabled.
    - Trade-offs: BPE is simple/deterministic; Unigram can yield fewer tokens and better generalization, especially multilingual.

- What are pre-training, fine-tuning, and instruction-tuning in LLMs?
  - Pre-training (self-supervised): train on massive unlabeled corpora with next-token prediction (causal LM) or masked LM. Outcome: broad language/world knowledge and general skills. Examples: GPT/LLaMA pre-trained on web/books/code.
  - Continual/domain-adaptive pre-training (optional): keep pretraining on in-domain text with the same objective to adapt to a new domain (legal/biomed/code) before task tuning.
  - Supervised fine-tuning (SFT): update the model on a narrower task/domain using labeled or synthetic data. Objectives: task cross-entropy; methods: full FT or PEFT (LoRA/adapters). Examples: fine-tune for code completion or NER on company data.
  - Instruction-tuning: SFT on instruction–response pairs so the model follows natural-language instructions (multi-task). Often followed by preference tuning (RLHF/DPO) to align tone, helpfulness, and safety.
  - When to use:
    - Pre-train: only if you control huge compute/data.
    - Continual pre-train: large domain shift with plenty of unlabeled text.
    - Fine-tune: specific tasks/domains with labeled data.
    - Instruction + preference tuning: change behavior/formatting and alignment.

- What are embeddings, and how do they help in semantic search and RAG?

  - Embeddings: dense vectors that encode semantic meaning of text/code/images so semantically similar items are close in vector space.
  - Produced by embedding models (e.g., sentence-transformers, E5/GTE, OpenAI text-embedding-3, Cohere; multilingual variants). Typical dims: 384–3072.
  - Similarity: cosine or dot product (L2-normalize for cosine). Used to rank relevance.

  - Semantic search workflow:
    - Ingest: split docs into chunks, compute embeddings, store vectors + metadata in a vector DB (FAISS, Qdrant, Pinecone, Weaviate) using ANN indexes (HNSW/IVF).
    - Query: embed the user query, run ANN search, return top-k nearest chunks.
    - Options: hybrid search (BM25 + vectors), multi-vector/late interaction (e.g., ColBERT), and cross-encoder re-ranking for precision.

  - In RAG:
    - Retrieve: embeddings fetch semantically relevant chunks for the question.
    - Re-rank/filter: optional cross-encoder re-ranker, metadata filters, time decay.
    - Generate: build the prompt with retrieved chunks so the LLM answers grounded in sources, reducing hallucinations and enabling citations.

  - Practical tips:
    - Pick domain- and language-aligned models; tune chunk size/overlap.
    - Store metadata (source, section, timestamp); dedupe and normalize vectors.
    - Re-embed when models or data shift; cache to cut cost/latency.
    - Evaluate with recall@k, MRR/NDCG; for RAG, measure groundedness/faithfulness.


### LLM Mechanics


- Explain how positional encoding works in transformer models.

  - Why needed: self-attention is order-agnostic; positional info injects sequence order.
  - Absolute (sinusoidal): PE[pos,2i] = sin(pos/10000^(2i/d)), PE[pos,2i+1] = cos(...). Added to token embeddings (often only at the input). No params; good extrapolation.
  - Learned absolute: trainable position vectors up to a max length. Simple and strong, but limited extrapolation unless extended/interpolated.
  - Relative positions:
    - Shaw et al.: add learned bias a(i−j) to attention logits (picks up relative offsets).
    - RoPE (rotary): rotate Q,K in 2D planes by angle θ(pos), making attention depend on relative offsets; strong long-context behavior.
    - ALiBi: add head-specific linear bias m_h·(i−j) to logits; cheap and extrapolates.
  - Practical notes:
    - Apply positions once at input (common in decoder-only) or per layer (some encoders).
    - Use padding/causal masks; positions count only real tokens.
    - Long-context: RoPE scaling/NTK-aware interpolation, position shifting.
    - 2D/axial encodings for images; time-axis encodings for audio.

- What is the difference between causal (decoder-only) vs. seq2seq (encoder-decoder) LLMs?

- What is the difference between causal (decoder-only) vs. seq2seq (encoder-decoder) LLMs?
  - Architecture:
    - Decoder-only: single stack with causal self-attention (no lookahead). Condition on left context only. Examples: GPT, LLaMA, Mistral.
    - Seq2seq: encoder (bidirectional self-attention) + decoder (causal self-attention + cross-attention to encoder). Examples: T5/FLAN-T5, BART, mT5.
  - Training objective:
    - Decoder-only: next-token prediction on concatenated text (causal LM).
    - Seq2seq: map input→output (supervised) or denoising (e.g., span corruption in T5), training decoder to generate conditioned on encoder outputs.
  - Strengths:
    - Decoder-only: excels at open-ended generation, chat, long-context prompting, simple serving; very scalable.
    - Seq2seq: strong at input→output tasks (translation, summarization, structured generation) due to full bidirectional encoding of the source.
  - Trade-offs:
    - Decoder-only: all conditioning must fit in the prompt; longer prompts increase cost; no explicit cross-attention memory.
    - Seq2seq: extra encode pass + cross-attention adds latency/params, but gives stronger conditioning and shorter prompts.
  - When to use:
    - Use decoder-only for chat/agent/RAG with long prompts and general generation.
    - Use seq2seq for supervised transformations where the output tightly depends on an input document.


- Define perplexity. Why is it used to evaluate LLMs?

- Define perplexity. Why is it used to evaluate LLMs?
  - Definition: Perplexity (PPL) = exp(cross-entropy per token). For tokens x₁..x_N, PPL = exp(−(1/N) Σ log p(x_t | x_<t)). Lower is better.
  - Intuition: average branching factor—the effective number of choices per next token.
  - Why used:
    - Aligned with the LM training objective; sensitive to modeling gains.
    - Simple offline metric; track train vs. val to detect overfitting and domain shift.
    - Comparable only when dataset and tokenizer are identical.
  - Practical notes:
    - Report tokenizer, dataset, and whether per-token or per-word PPL.
    - Weak correlation with instruction-following/helpfulness; use task metrics or human eval for those.
    - Not comparable across different tokenizations or heavy prompt formatting.

- Explain LoRA (Low-Rank Adaptation) and why it’s efficient for fine-tuning.

- Define perplexity. Why is it used to evaluate LLMs?
  - Definition: Perplexity (PPL) = exp(cross-entropy per token). For tokens x₁..x_N, PPL = exp(−(1/N) Σ log p(x_t | x_<t)). Lower is better.
  - Intuition: average branching factor—the effective number of choices per next token.
  - Why used:
    - Aligned with the LM training objective; sensitive to modeling gains.
    - Simple offline metric; track train vs. val to detect overfitting and domain shift.
    - Comparable only when dataset and tokenizer are identical.
  - Practical notes:
    - Report tokenizer, dataset, and whether per-token or per-word PPL.
    - Weak correlation with instruction-following/helpfulness; use task metrics or human eval for those.
    - Not comparable across different tokenizations or heavy prompt formatting.

- How does parameter-efficient fine-tuning (PEFT) differ from full fine-tuning?

- How does parameter-efficient fine-tuning (PEFT) differ from full fine-tuning?
  - Full fine-tuning:
    - Update all model weights (100%). Highest capacity; best for large domain shifts.
    - Expensive: optimizer states ≈ 2–3× params; large VRAM; store a full model per task.
    - Risk of catastrophic forgetting without mixed/data-regularization.
  - PEFT:
    - Freeze base model; train small add-ons or deltas (<<1% params).
    - Methods: LoRA/QLoRA (low-rank adapters on W_Q/W_K/W_V/FFN), Adapters, Prefix/Prompt tuning, BitFit.
    - Benefits: 10–100× lower train memory/compute; fast; cheap to store/swap adapters per task; works with 4/8-bit (QLoRA).
    - Trade-offs: slightly lower ceiling on tough domain shifts; choice of layers and rank r matters.
  - Deployment:
    - Full FT: one heavy checkpoint per task.
    - PEFT: load base once; hot-swap small adapters or merge LoRA into base for inference.
  - When to use:
    - Use PEFT for most downstream tasks, limited compute, or multi-tenant setups.
    - Use full FT for deep domain adaptation or when you must change core representations.


### RAG & AI Agents

- What problem does RAG solve compared to plain prompting?
  - Problem with plain prompting: LLMs lack up-to-date/domain-specific facts; hallucinate when info isn’t in weights; limited context window; no citations or access control.
  - RAG solution: retrieve top‑k relevant chunks from external sources at query time and ground the generation in them.
    - Benefits: fresher knowledge, fewer hallucinations, shorter prompts than pasting whole docs, citations/traceability, scoped access (per user/tenant).
  - When to use: dynamic or proprietary corpora; compliance/audit needs; per-user knowledge partitions.
  - Trade-offs: retrieval quality becomes the bottleneck (chunking, embeddings, re-ranking); added latency/cost; prompt injection from retrieved text; must evaluate recall and faithfulness.

- Explain chunking strategies in RAG pipelines. Trade-offs of large vs. small chunks.

  - Goal: keep enough local context for answering while maximizing retrieval recall.
  - Common strategies:
    - Fixed-length token chunks with overlap (sliding window). Simple, fast, predictable.
    - Natural-boundary chunking: split by sentences/paragraphs/headers (Markdown/HTML/PDF), keep headings with the chunk.
    - Structure-aware: keep tables as whole units; code-aware splits by function/class/file blocks; notebooks by cell.
    - Hierarchical (parent-child): big parent sections (e.g., 800–1500 tokens) plus overlapping leaf chunks (150–400) for retrieval; fetch parent for context.
    - Query-time expansion: after retrieving a leaf, also pull left/right neighbors or the parent; optional contextual compression (LLM filters/summarizes to the query).
  - Large vs small chunks:
    - Large (400–1200 tok): + more context, fewer boundary breaks, better multi-sentence reasoning; − diluted embeddings, lower recall for narrow queries, higher prompt cost, risk of off-topic content.
    - Small (100–300 tok): + precise matches, higher recall, granular citations; − fragmented context, may require neighbor fetching/re-ranking, larger index and latency.
  - Overlap:
    - Use 10–25% overlap (e.g., 32–128 tok). Too little → boundary loss; too much → duplication, higher cost.
  - Practical defaults (start here, then tune):
    - Prose/docs: 200–400 tok, 10–20% overlap; preserve headings in each chunk.
    - FAQs/emails: 80–200 tok; sentence-aware splitting.
    - Code: 50–120 tok but prefer AST/function-level chunks; include signature/imports.
    - Tables: keep intact; add header row and caption to text representation.
  - Tips:
    - Store rich metadata (title → section path → page → offsets); stable chunk IDs for dedup/versioning.
    - Evaluate with recall@k and answer faithfulness; grid-search chunk size/overlap.
    - Use re-rankers (cross-encoder) or multi-vector methods when using larger chunks.

- How does multi-vector (hybrid) search improve retrieval over embeddings-only search?


- Describe the role of re-ranking models (BERT, ColBERT, NeMo Retriever) in RAG.
- What is an AI Agent? How do frameworks like CrewAI, LangGraph, AutoGen differ?
- Explain tool use (function calling) in LLMs. How does it enable agents?
- How would you implement memory in an agent system?

### Evaluation & Safety
- Common evaluation metrics for LLMs (BLEU, ROUGE, F1, RAGAS).
- Explain hallucination in LLMs. Strategies to reduce it.
- What is groundedness in GenAI evaluation? How do you test for it?
- How does constitutional AI enforce alignment?
- What are prompt injection and data poisoning attacks? Mitigation strategies?
- How to monitor bias, toxicity, fairness in a GenAI app?
- Difference between adversarial prompts and jailbreak prompts.

### Deployment & MLOps
- How to design a scalable LLM inference system?
- Compare hosting on OpenAI API / Bedrock / Azure AI Foundry vs self-hosted LLaMA/Mistral.
- How does quantization (INT8/4-bit) help in deployment? Trade-offs.
- Difference between online vs batch inference for LLM workloads.
- Implement caching strategies to reduce LLM cost/latency.
- Role of vector databases in production LLM apps (FAISS, Qdrant, Pinecone, Weaviate).

---

## 2. System Design & Scenario Questions
- Design an enterprise chatbot integrating 25+ REST APIs.
- Build a GenAI-powered personal assistant with speech-to-text → LLM → text-to-speech while keeping latency low.
- Design a multi-cloud RAG pipeline (AWS Bedrock + Azure AI Search).
- Mitigate hallucinations in financial data GenAI apps.
- Integrate real-time analytics dashboards (PowerBI + AI Copilot) with LLM pipelines.
- Water quality prediction system with ML + GenAI report generation.
- Implement human-in-the-loop feedback to improve LLM answers.
- On-premise LLM deployment: hardware, software, MLOps considerations.
- Decide between full fine-tuning vs adapters/LoRA.
- Secure integration of enterprise documents (PII, compliance, RBAC).

---

## 3. Behavioral / Teamwork Questions
- Tell us about a time you optimized an ML/GenAI pipeline.
- Describe a GenAI project you led. Challenges and solutions.
- How do you collaborate with non-technical stakeholders?
- Example of handling constructive feedback on GenAI work.
- How do you stay updated with GenAI advancements?

---

## 4. Basic ML Questions for GenAI

### ML Fundamentals
- Supervised vs. Unsupervised vs. RL (link to embeddings, fine-tuning, RLHF)
- Overfitting vs. Underfitting (how to detect in fine-tuned LLM)
- Precision, Recall, F1 (RAG evaluation)
- Bias-Variance tradeoff (embedding models)
- Regularization (LoRA / PEFT)

### Feature Engineering & Data
- Feature scaling (vector DBs)
- Handling missing values (RAG ingestion)
- Categorical encoding (metadata)
- Dimensionality reduction (PCA/t-SNE/UMAP on embeddings)
- Imbalanced datasets (safety fine-tuning)

### Algorithms
- Linear vs Logistic Regression (pre-routing queries)
- Decision Trees vs Random Forest vs Gradient Boosting (document classification)
- K-Means Clustering (embedding clustering)
- Cosine Similarity vs Euclidean (embedding search)
- TF-IDF vs Word Embeddings

### Model Evaluation
- Confusion Matrix (toxic content classifier)
- ROC-AUC vs PR-AUC (class imbalance)
- Cross-validation (small fine-tuning datasets)
- Ranking evaluation (NDCG, MRR for re-rankers)
- Adversarial robustness (prompt injection tests)

### ML in GenAI Context
- Embedding vs classification models
- Vector similarity search
- SVM/BERT re-rankers in RAG
- ML + LLM hybrid pipelines
- Few-shot learning (parallels prompt engineering)

---

## 5. Coding Questions (Python / GenAI Basics)

1. **Token Counter**: Count tokens in a prompt → dict of token frequencies.
2. **Chunk Splitter**: Split text into n-word chunks.
3. **Cosine Similarity**: Compute cosine similarity between two vectors (no libs).
4. **Top-k Retriever**: Return top-k docs from a dict of scores.
5. **Stopword Filter**: Remove stopwords from a query string.
6. **Safe Prompt Validator**: Check if prompt contains forbidden keywords.

---

## 6. API-Oriented Coding Questions

1. **Call a Public API and Parse JSON**
2. **Retry API Call with Error Handling**
3. **Build a Simple FastAPI Endpoint**
4. **Call OpenAI (or Mock API) and Return Response**
5. **Async API Calls (Parallel Requests)**
6. **REST API Wrapper Class (TodoAPI)**

---

## 7. Difficulty Guide

- **Easy:** Token Counter, Stopword Filter, Public API Call, FastAPI Endpoint  
- **Medium:** Chunk Splitter, Top-k Retriever, Retry API Call, API Wrapper Class  
- **Advanced:** Cosine Similarity, OpenAI API Call, Async API Calls
