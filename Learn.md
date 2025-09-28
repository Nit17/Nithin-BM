
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
  - Idea: freeze the large weight W and learn a low‑rank update ΔW instead of updating all params.
    - Parameterization: $W' = W + s \cdot BA$, where $A \in \mathbb{R}^{r \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r}$, rank $r \ll \min(d_\text{in}, d_\text{out})$, scale $s$.
    - Forward: $W'x = Wx + s \cdot B(Ax)$. Initialize $B$ near 0 so training starts from the base model.
  - Where applied: usually to attention projections (W_Q, W_K, W_V, W_O) and sometimes MLP layers. Base weights stay frozen; only A,B train.
  - Why efficient:
    - Far fewer trainable params (often <1%) → 10–100× lower optimizer memory and faster training.
    - Compatible with quantization (QLoRA): keep W in 4‑bit, train LoRA in 16‑bit → fit bigger models on smaller GPUs.
    - Modular: per‑task adapters are tiny; hot‑swap or merge into W for inference ($W \leftarrow W + sBA$).
  - Hyperparams/tips:
    - Rank r: 4–64 common; larger r = more capacity/cost. Use scaling α (LoRA α) and small lora_dropout for regularization.
    - Target only the most impactful layers first (attention > MLP) to save memory.
    - Use a re‑ranker/validation set to tune r, α, dropout. Monitor loss and task metrics.
  - Trade‑offs:
    - Slightly lower ceiling vs full FT on large domain shifts; performance sensitive to which layers are adapted and chosen rank.
    - Multiple adapters can be composed (sum of ΔW), but may need calibration.


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
  - What it is:
    - Hybrid sparse+dense: combine lexical signals (BM25/SPLADE) with dense embeddings.
    - Multi-vector representations: index multiple vectors per doc/passage (e.g., token/phrase-level as in ColBERT, or multiple field/chunk vectors).
  - Why it helps:
    - Recovers exact terms (rare entities, codes, numbers, negations) that dense-only can miss, while still capturing semantic similarity.
    - Better recall on short/keyword-y or out-of-domain queries; more robust to multi-intent queries.
    - Typically improves top-k recall and downstream answer quality before re-ranking.
  - Common variants:
    - Sparse + dense fusion: score = α·cosine_dense + (1−α)·BM25 (normalize scores first) or use Reciprocal Rank Fusion (RRF).
    - Late interaction (ColBERT-style): represent passages with many token vectors; score = sum/max of token-level sims → preserves lexical precision with semantic matching.
    - Multi-field/multi-chunk: separate vectors for title/body/code/comments or per-chunk; aggregate (max/mean) at doc level.
  - Implementation notes:
    - Normalize per-retriever scores (z-score or min–max) before weighted fusion; tune α on a validation set.
    - RRF avoids calibration issues: RRF(doc) = Σ_i 1/(k + rank_i).
    - Add a cross-encoder re-ranker on top-N to boost precision.
  - Trade-offs:
    - More storage/compute and higher latency than dense-only; fusion weight tuning required.
    - Late interaction/multi-vector can be memory-heavy (many vectors per doc).
  - When to use:
    - Domains with jargon/IDs (legal, medical, code), short queries, safety-critical retrieval where missing exact terms is costly, or multilingual/noisy inputs.

- Describe the role of re-ranking models (BERT, ColBERT, NeMo Retriever) in RAG.
  - Role in RAG: second-stage precision filter. After a fast retriever (BM25/dense) returns top-N, a re-ranker scores (q, passage) pairs to select the final k passages used in the prompt. Boosts precision@k, reduces off-target context and hallucinations.
  - BERT cross-encoder (e.g., monoBERT, MS MARCO CE):
    - Architecture: joint encode “[CLS] query [SEP] passage [SEP]” → single relevance score.
    - Strength: highest precision; great for small N (e.g., N=50–200 → keep k=5–10).
    - Cost: slowest per pair; run on GPU, batch requests, truncate long passages or window them.
  - ColBERT (late interaction bi-encoder):
    - Idea: encode query and passage into token-level embeddings; score via MaxSim over tokens (preserves lexical precision with semantic matching).
    - Strength: better balance of speed/precision than cross-encoders; supports multi-vector indexing (good for code, entities).
    - Cost: bigger storage (many vectors per doc); custom index and scoring (but scalable with ANN).
  - NeMo Retriever (NVIDIA):
    - Toolkit with dense retrievers and cross-encoder re-rankers; supports domain adaptation, distillation, multi-GPU, and Triton deployment.
    - Strength: production-friendly, optimized inference, easy to compose hybrid pipelines (sparse+dense→re-rank).
  - Practical tips:
    - Use re-ranking on the top-N from your retriever; tune N and final k on validation (e.g., N=100, k=10).
    - For fusion (sparse+dense), normalize scores or use Reciprocal Rank Fusion before re-ranking.
    - Chunk-level re-ranking often beats doc-level; de-duplicate and diversify (MMR) to avoid near-duplicates.
    - Evaluate with NDCG/MRR/Recall@k; watch latency—fallback to lighter models (DistilBERT CE) if needed.

- What is an AI Agent? How do frameworks like CrewAI, LangGraph, AutoGen differ?
  - What is an AI Agent?
    - An LLM-driven actor with a goal, memory/state, and the ability to plan and take actions via tools/APIs. Core pieces: (1) policy/reasoner (LLM prompts), (2) tools/functions, (3) memory (short/long-term), (4) planning/control loop, (5) feedback/evaluation.
  - CrewAI
    - Opinionated multi-agent “crews” with roles and tasks; simple to stand up (Python/YAML). Built-in patterns (sequential/parallel/hierarchical manager-worker), handoffs, and tool registries.
    - Best for: quick multi-agent workflows, role-based collaboration, low boilerplate. Trade-off: less fine-grained control over state/flow than graph-first frameworks.
  - LangGraph (LangChain)
    - Graph/state-machine orchestration for agents and tools. Explicit nodes/edges, loops, interrupts, and persistence/checkpointing; typed shared state and streaming.
    - Best for: production reliability, deterministic control over loops/retries, complex routing and hybrid RAG pipelines. Trade-off: more engineering effort to design the graph/state.
  - AutoGen
    - Message-passing multi-agent framework (chat among agents). Primitives like Assistant, UserProxy, GroupChat; strong support for code execution and tool use inside the loop (e.g., CodeExecutor).
    - Best for: research/experiments with agent conversations, self-correction/debate, code-centric tasks. Trade-off: conversational loops can be costly; less structured than graph-first orchestration.
  - Selection tips
    - Need fast multi-agent prototypes with role/task semantics → CrewAI.
    - Need robust control, retries, and stateful workflows in prod → LangGraph.
    - Exploring multi-agent dialogues, code-gen with execution and debate → AutoGen.
  - Common concerns
    - Cost/latency control (limit tool calls/turns), observability (logs/traces), safety (tool whitelists, input/output checks), and memory design (episodic vs vector store vs summarized history).

- Explain tool use (function calling) in LLMs. How does it enable agents?
  - What it is:
    - The LLM emits a structured “function call” (tool name + JSON args matching a schema). The runtime executes the tool (API/DB/code), returns an observation, and the LLM continues with that result. Loop until a final answer.
  - Why it enables agents:
    - Extends the model beyond its weights: browse/query/compute/act (e.g., call APIs, run code, query SQL/vector DBs, control apps).
    - Supports planning→acting→observing loops (ReAct/Plan-and-Execute), letting the model decompose tasks, use tools, and self-correct.
    - Structured I/O (JSON) makes integrations reliable and composable across multiple steps.
  - Typical components:
    - Tool registry: name, description, JSON Schema, auth/permissions.
    - Router/selector: the model decides which tool (or none) and constructs valid args.
    - Executor: runs the tool, enforces timeouts/quotas, returns a compact result.
    - Memory/state: short-term (chat history) + long-term (vector store/DB) tools for recall and persistence.
  - Common patterns:
    - ReAct: think → act (tool call) → observe; iterate.
    - Planner–executor: a planner drafts steps, an executor performs tool calls.
    - Hybrid RAG: retrieve (vector/sparse) → optionally re-rank → generate; retrieval is “just a tool.”
  - Design tips:
    - Write precise tool descriptions with small examples; use JSON Schema constraints (enums, min/max) to guide arguments.
    - Validate args server-side; return machine-friendly, truncated results (ids, fields) plus a “get_by_id” tool to fetch details on demand.
    - Cap tool calls/turns; add retries/backoff; normalize/calibrate scores for fused retrievers; log calls/latency/errors for observability.
  - Safety & reliability:
    - Principle of least privilege, per-user/tenant scoping, allowlists, rate limits, and sandboxed execution for code/tools.
    - Guard against prompt injection from retrieved text; sanitize outputs; filter PII; audit logs for actions with side effects.
    - Prevent infinite loops (max steps), handle tool errors with explicit error payloads, and require structured outputs to avoid JSON drift.

- How would you implement memory in an agent system?
  - Goals: let the agent remember salient facts, preferences, decisions, and past tool results to improve future reasoning while controlling cost, drift, and privacy.
  - Memory types:
    - Short‑term/working: recent turns, scratchpad (ReAct) for current plan/variables.
    - Episodic: time‑stamped events (who/what/when/where) from prior sessions.
    - Semantic: distilled facts/preferences/skills the agent has learned.
    - Tool/cache: results of expensive queries, embeddings, auth state, schemas.
    - Shared/blackboard (multi‑agent): team-visible notes with access controls.
  - Read pipeline (at inference):
    1) Identify memory need (heuristic or model hint: “retrieve profile/project facts”).
    2) Query stores:
       - Vector store for semantically relevant memories (k=5–20, cosine/dot).
       - Key–value/Redis for recent session state, TTL caches.
       - RDBMS/graph DB for structured facts and relations.
    3) Re-rank/curate: dedupe, diversify (MMR), time decay, source filtering.
    4) Compress to fit context: summarize or “contextual compression” to the query.
  - Write pipeline (after each step/turn):
    1) Extract candidates (LLM tagger or rules) → entities, preferences, decisions, failures.
    2) Gate by salience: write only if novel, important, and likely reusable (score threshold).
    3) Normalize to schema:
       - event_id, user_id/tenant_id, agent_id, ts, type {fact|pref|task|error}, text, structured fields, embeddings, PII flags, TTL/policy.
    4) Persist to:
       - Vector DB (text + embedding), KV/Redis (recent/TTL), SQL (structured).
    5) Schedule consolidation: periodic “reflection” jobs that summarize episodic logs into stable semantic facts; archive or delete raw logs.
  - Retention/forgetting:
    - Sliding windows per topic/session; TTLs by type; time‑decay scoring.
    - Quotas per user/tenant; background summarization to keep total tokens bounded.
    - Guard against contamination: label provenance; don’t promote low‑confidence info to semantic memory.
  - Safety & governance:
    - PII detection/redaction; encryption at rest; per‑tenant indexes; RBAC on reads.
    - Prompt‑injection defense: never write memories from untrusted text without gating; keep “instructions” separate from “facts.”
    - Audit logs for memory writes/reads; right‑to‑be‑forgotten workflows.
  - Practical defaults:
    - Embeddings: E5/GTE 768–1024d; chunk 200–400 tokens; HNSW index; k=10; MMR λ≈0.5.
    - Summarize when thread >1–2k tokens or daily; keep a 5–20 turn short‑term window.
    - Store profile/preferences as structured rows with an embedding mirror for retrieval.
  - Multi‑agent notes:
    - Shared blackboard with typed channels (plan, findings, blockers); per‑agent private memory.
    - Mediator enforces who can write/read which channel; summarize long threads to keep latency low.
  - Evaluation:
    - Ablate memory reads/writes; track answer accuracy, groundedness, latency, and tool call count.
    - Spot‑check for drift and privacy leaks; test deletion/RTBF works end‑to‑end.

### Evaluation & Safety

- Common evaluation metrics for LLMs (BLEU, ROUGE, F1, RAGAS).

  - BLEU (Machine Translation-style):
    - What: n-gram precision with a brevity penalty (typically up to 4-grams).
    - Use for: MT or tasks where exact phrasing overlap matters.
    - Notes: apply smoothing; case/segmentation matter; weak on faithfulness/semantics for open-ended gen.

  - ROUGE (Summarization-style):
    - What: recall-focused n-gram overlap. ROUGE-1/2 (uni/bi-gram recall), ROUGE-L/Lsum (Longest Common Subsequence).
    - Use for: summarization; content coverage vs reference.
    - Notes: can reward verbose outputs; doesn’t guarantee factuality.

  - F1 (Precision/Recall harmonic mean):
    - What: token- or span-level Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R).
    - Use for: extractive QA, information extraction, classification; often paired with Exact Match (EM) for QA.
    - Notes: specify micro vs macro averaging; tokenizer and normalization (lowercase, punctuation, stopwords) change scores.

  - RAGAS (RAG-specific):
    - What: LLM-graded metrics tailored to retrieval-augmented generation:
      - Faithfulness: are claims grounded in retrieved context?
      - Answer Relevance: does the answer address the question?
      - Context Precision/Recall (and Relevance): is retrieved context on-topic and sufficient?
      - Answer Correctness (optional): semantic correctness vs reference.
    - Use for: end-to-end RAG evaluation beyond overlap metrics.
    - Notes: relies on an LLM judge → prompt/config sensitivity, variance, and cost; fix temperature (e.g., 0), seed, and judge model; audit a sample manually.

  - Practical tips:
    - Match metric to task: MT → BLEU; summarization → ROUGE/BERTScore; extractive QA → EM/F1; RAG → RAGAS + retrieval metrics (Recall@k, MRR, NDCG).
    - Standardize preprocessing (casing, stemming, detokenization) for fair comparisons.
    - Add human eval for faithfulness/helpfulness; overlap metrics don’t capture factuality.
    - Report dataset, metric variants, and configs (n-gram order, smoothing, judge model) for reproducibility.

  - Also consider:
    - BERTScore/BLEURT (semantic similarity), QAFactEval/FactCC (factual consistency) as stronger semantic/faithfulness checks.

- Explain hallucination in LLMs. Strategies to reduce it.

  - What is hallucination: model outputs that are unsupported by evidence (extrinsic), contradict facts, or internally inconsistent (intrinsic); includes fabricated citations/IDs/numbers.
  - Why it happens: next-token training rewards fluency over factuality; missing/ stale knowledge; ambiguous prompts; long/irrelevant context; aggressive sampling/decoding.

  - Reduction strategies:
    - Grounding (RAG): retrieve top‑k relevant chunks; use strong retrievers + re‑rankers; instruct “answer only from context; say ‘not found’ otherwise”; require citations (source ids/spans).
    - Prompting: be specific about format and refusal policy; ask for stepwise evidence checks without revealing chain-of-thought; request an “I don’t know” option.
    - Tools over guessing: calculator for math, code exec for logic, web/API/db for facts; function calling with strict JSON schema.
    - Constrained generation: schemas/grammars, closed‑book vs open‑book modes, entity lists/KB constraints, allowlists for domains.
    - Decoding controls: lower temperature/top‑p; avoid wide beam search for factual tasks; cap max tokens; penalize repetition.
    - Model & tuning: choose domain‑aligned models; SFT on grounded QA; preference tuning (RLHF/DPO) for truthfulness/abstention; penalize unsupported claims in rewards.
    - Context hygiene: good chunking, overlap, metadata filters, time decay; dedupe; keep prompts short and on-topic.

  - Detection & evaluation:
    - Automatic: RAGAS (faithfulness, context precision/recall), QAFactEval/FactCC, entailment (NLI) between answer and sources.
    - Retrieval metrics: Recall@k, MRR/NDCG; track “answerable from context” rate and abstention rate.
    - Human audits on sampled outputs; log citations and verify they support each claim.


- What is groundedness in GenAI evaluation? How do you test for it?

  - Definition: the degree to which an answer’s claims are supported by the provided context (retrieved documents, tables, APIs), not just by the model’s parametric knowledge. Distinct from “factuality” (truth w.r.t. the real world); groundedness is support w.r.t. supplied sources.

  - How to test (practical recipe):
    1) Require citations: have the model output claims with source_ids (and optionally span/quote offsets).
    2) Claim extraction: break the answer into atomic claims (rules or an LLM tagger).
    3) Evidence retrieval: for each claim, gather cited spans; if missing, search the provided context (sentence-level).
    4) Entailment check: run an NLI/citation verifier (e.g., DeBERTa/BERT NLI or LLM-as-judge) to see if evidence supports the claim.
    5) Score:
       - Attribution Precision/Recall/F1 = over claims, what fraction are supported by cited context vs missing/contradicted.
       - RAGAS-style: Faithfulness (claims supported), Context Precision/Recall (context on-topic and sufficient), Answer Relevance.
       - Report unsupported rate, abstention rate (“I don’t know”), and citation coverage.

  - Tools/metrics:
    - RAGAS: Faithfulness, Context Precision/Recall, Answer Relevance, (optional) Answer Correctness.
    - NLI/QAFactEval/FactCC: entailment between claim and cited span(s).
    - Retrieval coupling: Recall@k/NDCG for whether gold spans were retrieved; low recall caps groundedness.
    - Human audit: sample and verify that cited passages actually support each claim.

  - Guardrails to improve groundedness:
    - Prompting: “Answer only from the context; if not present, say ‘Not found’” and require inline citations (e.g., [doc_id#para]).
    - Generation constraints: schema with fields {claim, citation_ids, quote}; penalize outputs without valid citations.
    - Pipeline: strong retriever + re-ranker; contextual compression; filter stale/off-domain docs; time decay.
    - Decoding: lower temperature for factual tasks; cap max tokens to reduce drift.
    - Training: SFT on grounded QA with citations; preference tuning (RLHF/DPO) rewarding supported claims/abstention.

  - Reporting:
    - Always specify judge model/prompt (if LLM-as-judge), temperature=0, seed, and datasets.
    - Include both answer-level and claim-level metrics, plus retrieval metrics, for a full picture.


- How does constitutional AI enforce alignment?

  - Definition: A training and inference-time framework where an explicit set of principles (a “constitution”) guides the model to be helpful, harmless, and honest. The rules are written as natural-language guidelines the model can reference.

  - Mechanisms (how it enforces alignment):
    - Inference-time critique-and-rewrite:
      1) Draft: the model produces an initial answer.
      2) Self-critique: prompt the model to check its output against the constitution (e.g., safety, privacy, fairness) and list violations.
      3) Revise: the model rewrites the answer to comply; optionally iterate or do rejection sampling then rewrite.
    - Training-time with AI feedback (RLAIF):
      - Generate pairs: produce initial and revised answers; use a judge prompt grounded in the constitution to prefer the safer/more aligned response.
      - Preference tuning: train with AI preference labels (e.g., reward modeling + RLHF, or DPO/ORPO using the preferred vs rejected outputs).
      - SFT with critiques: optionally include “critique → revision” traces, then distill to a single-step policy that internalizes the rules.

  - Example principles (constitution):
    - Safety/harms: refuse illegal, dangerous, or self-harm instructions; offer safe alternatives.
    - Privacy: avoid revealing PII; don’t infer sensitive attributes; minimize data exposure.
    - Honesty: avoid fabrications; acknowledge uncertainty; cite sources when asked.
    - Fairness/neutrality: avoid biased or demeaning language; maintain neutral tone.
    - Professional boundaries: no medical/legal/financial advice beyond general information; encourage consulting professionals.

  - Where rules live: system prompts (policy), judge prompts (for preferences), and training data (preferred vs rejected responses). They can be versioned per product/locale and audited.

  - Why it works: converts vague “be safe” into consistent, auditable instructions that the model can apply to police itself; scales alignment signals via AI feedback instead of relying solely on costly human labels.

  - Limitations/trade-offs: possible over-refusals or under-refusals; rule conflicts and prompt-injection risks; dependence on judge/policy quality; distribution shifts (language/domain) can degrade adherence; may curb creativity if overly strict.

  - Evaluation tips: red-team pass rate, harmful-advice rate, refusal accuracy (should refuse when required, comply when safe), helpfulness with safe alternatives, and policy-citation quality; combine with groundedness checks for factual tasks.



- What are prompt injection and data poisoning attacks? Mitigation strategies?

  - Prompt injection (at inference):
    - What: An adversary crafts inputs (user prompts or retrieved content in RAG) that hijack instructions, e.g., “Ignore previous rules and …”, exfiltrate secrets, or force unsafe actions via tool calls.
    - Variants: direct user prompt injection; indirect/jailbreak via retrieved webpages/docs; instruction smuggling in code blocks/HTML comments/metadata; CSS/data-URI attacks on UIs; tool-output “reflection” attacks.
    - Risks: policy override, data exfiltration (keys, system prompt), unsafe tool actions, reputational/ compliance breaches.
  - Data poisoning (at training/ingestion):
    - What: Malicious or low-quality data inserted into training corpora, embedding stores, or evaluation sets to bias behavior or trigger backdoors (e.g., special tokens cause harmful responses).
    - Variants: pretraining/finetuning poison; RAG index poison (malicious chunks dominate retrieval); prompt templates or evals with contaminated references.
    - Risks: persistent misbehavior, targeted backdoors, degraded retrieval precision, miscalibrated evaluations.

  - Mitigation strategies:
    - Defense-in-depth policy:
      - Separate instruction channels: keep system/developer prompts immutable and out of the model’s editable context; label provenance of each context segment.
      - Content provenance & trust tiers: mark retrieved text as untrusted; apply different parsing/rules; never let untrusted text set policy.
      - Principle of least privilege for tools: fine-grained allowlists, per-user scopes, dry-run/confirm steps for side-effects; timeouts and quotas.
    - Prompting & parsing:
      - Sandwich prompts: policy → user → policy reminder; restate rules after user content; require structured JSON outputs with strict schemas/grammars.
      - Delimiters and fenced blocks: treat retrieved text as data, not instructions; instruct the model to ignore any instructions inside user/retrieved content.
      - Output validation: JSON schema validation; type/enum checks; allow only known commands/IDs; post-processors to block suspicious patterns (URLs, secrets).
    - Retrieval/RAG hygiene:
      - Source filtering: domain allowlists, blocklists, freshness windows; strip scripts/HTML; sanitize Markdown links; disable remote resource loading.
      - Contextual compression and re-ranking: reduce attack surface by passing only necessary spans; diversify to avoid single-source domination.
      - Citation requirement: force evidence-linked answers to discourage following injected instructions.
    - Detection & monitoring:
      - Heuristics: patterns like “ignore previous”, base64/hex blobs, long code blocks with instructions; anomaly detectors on token distributions.
      - Safety classifiers: toxicity/hate/self-harm, data exfiltration detectors; tool-call anomaly detection (rate/arg patterns).
      - Red-teaming and canaries: seeded prompts and hidden markers to detect exfiltration and policy bypasses.
    - Data pipeline protections (poisoning):
      - Data contracts and lineage: track source, hashes, versioning; quarantine unverified data; dual-approval for schema/prompt template changes.
      - Automated quality gates: deduplication, PII scrub, profanity filters, language/id checks; outlier detection on embeddings.
      - Backdoor screening: test triggers (rare tokens/phrases) and differential behavior; hold-out clean validation sets.
      - RAG index controls: de-dup, per-tenant isolation, signed documents, moderation at ingest, periodic re-embedding with vetted models.
    - Organizational controls:
      - Secrets management: never put secrets/system prompts in user-visible contexts; use KMS/secret managers; rotate keys.
      - Access controls & auditing: RBAC on tools/data; immutable logs of prompts, tool calls, and outputs; incident response playbooks.

  - Practical defaults:
    - Temperature 0–0.3 for factual tasks; strict JSON schemas; max tokens limits; reject responses lacking required citations when in RAG mode.
    - Retrievers: hybrid (BM25+dense), top-k small with re-ranking, context windows 1–2k tokens with compression; sanitize HTML/JS.
    - Regularly run a prompt-injection test suite and a poisoned-data sentinel set; track refusal correctness and exfiltration rate as KPIs.




- How to monitor bias, toxicity, fairness in a GenAI app?

  - Goals: detect and reduce harmful or systematically skewed outputs (toxicity, hate, stereotyping, demographic performance gaps) while preserving utility and minimizing false positives (over-refusals).

  - Key dimensions:
    - Toxicity / Hate / Harassment
    - Violence / Self-harm encouragement
    - Sexual content / Adult / Minors
    - Demographic bias: disparities in tone, refusal rate, sentiment, or accuracy across groups (gender, ethnicity, age, locale, disability, religion, etc.)
    - Stereotype & association bias: e.g., occupation ↔ demographic pairings
    - Fairness in task performance: precision/recall/latency differences across segments
    - Over-refusal bias: safe queries from sensitive groups incorrectly blocked more often

  - Dataset strategy:
    - Benchmark sets: RealToxicityPrompts, ToxiGen, CivilComments (for toxicity); HolisticBias / CrowS-Pairs / StereoSet (stereotypes); BOLD for open-ended generation.
    - Synthetic augmentation: use templated or LLM-generated prompts covering intersectional groups (e.g., profession + demographic + neutral/harmful intents).
    - Shadow traffic sampling: log real user prompts (after PII scrubbing) and stratify by intent categories.
    - Balanced eval slices: ensure enough samples per group for statistical power; use bootstrapping for confidence intervals.

  - Instrumentation & logging:
    - Capture: prompt, model version, decoding params, context docs IDs, tool calls, final answer, refusal flag, safety scores.
    - Tagging: infer or map demographic references (with caution)—store hashed pseudo-identifiers; track only aggregated stats to reduce privacy risk.
    - Versioning: store policy/prompt template versions to attribute shifts.

  - Metrics (per group g and overall):
    - Toxicity rate: P(toxicity_score ≥ τ)
    - Refusal rate & unjustified refusal rate (refused when no violation) → compare Δg vs baseline.
    - Accuracy / relevance (for task outputs) by group; disparity = max_g − min_g or ratio.
    - Sentiment / politeness deltas across groups.
    - Stereotype association score: probability(model links group to stereotyped attribute) vs neutral baseline.
    - Calibration: AUC / Brier for classifier-style outputs (if classification tasks).
    - Utility vs safety frontier: plot toxicity rate vs answer acceptance to detect over-tuning.

  - Detection models & scoring:
    - Use ensemble: open-source classifiers (Perspective API-like models, Detoxify), LLM-as-judge (temp=0) with strict rubric, regex/keyword filters (high-precision for banned slurs).
    - Calibrate thresholds (τ) via ROC to balance false positives; maintain separate thresholds per language domain if needed.
    - Periodically re-score historical samples after model/policy upgrades.

  - Bias analytical methods:
    - Counterfactual evaluation: swap demographic terms ("he"→"she", group names) and measure output deltas (toxicity, sentiment, refusal change).
    - Paired prompts (CrowS-Pairs style) comparing stereotyped vs anti-stereotyped contexts; compute log-likelihood bias score.
    - Representation parity: embedding clustering distribution across groups (for internal embedding models).
    - Performance gap significance: two-proportion z-test or bootstrap CIs; alert if disparity exceeds policy threshold (e.g., >5 p.p.).

  - Mitigation techniques:
    - Prompt/policy: explicit safety instructions + neutral tone style guides; require citing evidence for claims about groups.
    - Decoding controls: lower temperature and top-p for safety-critical flows; early toxicity token blocking (constrained decoding) using a banned token trie.
    - Post-generation filtering: cascade of fast filters → classifier → LLM rewrite (attempt detoxification) → final refuse if unresolved.
    - Fine-tuning / preference tuning: include detoxified rewrites and counter-stereotype examples; apply group-balanced sampling; penalize toxic continuations in reward model.
    - Representation debiasing (if controlling embeddings): iterative nullspace projection / INLP / contrastive debiasing (apply carefully—avoid erasing legitimate distinctions causing accuracy loss).
    - Retrieval guardrails (RAG): exclude high-toxicity sources; classify retrieved chunks; mask or summarize sensitive content before prompt inclusion.
    - Over-refusal correction: maintain a “benign sensitive” test list; if refusal rate drifts up without toxicity increase, relax policy or adjust refusal heuristics.

  - Monitoring pipeline (continuous):
    1) Ingest logs → de-identify → route to metrics computation.
    2) Batch scoring jobs (hourly/daily) compute toxicity & fairness metrics; persist time series.
    3) Drift detection: KS test on prompt category distribution; alert on shifts that could mask bias.
    4) Alerting: thresholds for (a) toxicity rate spike, (b) disparity > policy limit, (c) refusal surge.
    5) Human review queue: sample borderline or newly toxic categories for labeling.

  - Governance & documentation:
    - Maintain a Safety Model Card: model version, safety datasets, known limitations, last audit date.
    - Change management: require fairness impact review before deploying new policies or fine-tunes.
    - Incident response: run root-cause (data shift? classifier drift? decoding change?) and backfill metrics post-fix.

  - Practical defaults (starter numbers; tune per domain):
    - Toxicity threshold τ around 0.7 (Detoxify scale 0–1) initially; adjust to keep FP manageable.
    - Sample 1–5% of production prompts for deep fairness analysis daily (ensure min 200 per key group/week).
    - Disparity alert: toxicity_rate_gap > 5 p.p. or refusal_rate_ratio > 1.5× baseline.
    - Quarterly comprehensive bias audit; monthly lightweight checkpoint.

  - Pitfalls:
    - Over-sanitization harming usefulness; metric gaming (model avoids group terms → false fairness perception); reliance on a single classifier (concept drift); ignoring intersectionality; storing raw PII in logs.

  - Success criteria:
    - Stable or reduced toxicity with preserved task accuracy.
    - Disparity metrics within defined policy bounds over rolling windows.
    - Fast detection/rollback (<24h) for regressions.
    - Transparent reporting: audit logs + periodic fairness summary shared with stakeholders.

-- Difference between adversarial prompts and jailbreak prompts.

  - Adversarial prompts (umbrella): deliberately crafted inputs aimed at triggering ANY failure mode: unsafe or policy-violating output, misinformation, biased/ stereotyped responses, secret or system prompt leakage, logic derailment, denial-of-service via mass refusals, retrieval/tool misuse, or subtle quality degradation (lower factuality/groundedness).
  - Jailbreak prompts (subset): specialized adversarial prompts whose primary goal is to bypass safety & alignment guardrails to elicit explicitly disallowed content (e.g., weapon construction, self-harm facilitation, hate speech, doxxing, targeted harassment). Success criterion = model emits content it should have refused.

  - Typical jailbreak tactics:
    - Role / persona override: “Act as an uncensored model… ignore previous instructions.”
    - Indirection & hypotheticals: frame as fiction, satire, research, translation, or “explain what someone COULD do.”
    - Multi-turn priming: early benign turns weaken/rewrite policy context before the harmful ask.
    - Obfuscation: leet (w3ap0n), homoglyphs, zero‑width chars, base64/hex, spaced letters to evade filters.
    - Split payload: distribute the instruction over several messages or code/data blocks.
    - Instruction smuggling: hide override text inside quotes, tables, code fences, HTML comments, or retrieved RAG passages.

  - Broader adversarial patterns (not necessarily jailbreak):
    - Context window flooding / truncation (push system prompt out of window).
    - Retrieval poisoning (malicious chunk inserted so RAG cites false or unsafe instructions).
    - Tool steering / escalation (coax use of powerful tool with crafted rationale).
    - Formatting confusion (malformed JSON / nested quotes to break parsers or validators).
    - Logic traps & contradiction loops to degrade trust / consistency.

  - Differences at a glance:
    - Scope: adversarial = any exploit vector; jailbreak = explicit safety-policy circumvention.
    - Success signal: adversarial may just reduce quality or induce bias; jailbreak yields disallowed output.
    - Defense emphasis: adversarial requires anomaly + integrity + retrieval/tool protections; jailbreak emphasizes semantic safety & refusal robustness.
    - Attack leverage: jailbreak leans on social engineering & semantic persuasion; broader adversarial may exploit structural mechanics (context, formatting, indexing, tooling).

  - Relation to other terms:
    - Prompt injection: instruction override vector (esp. via retrieved content) enabling either jailbreak or other adversarial goals (exfiltration, manipulation).
    - Data poisoning: upstream contamination (training/index) that increases downstream exploitability.
    - Red-teaming: sanctioned generation of adversarial / jailbreak prompts for measurement & improvement.

  - Layered mitigations:
    1) Immutable policy reinforcement: reattach / reassert system safety preamble every turn; detect token displacement (system prompt index monitoring).
    2) Input normalization: Unicode NFKC, remove zero-width chars, canonicalize homoglyphs, detect/flag base64 & hex blocks, de-leet mapping.
    3) Safety gating cascade: fast pattern / keyword blacklist (high precision) → lightweight classifiers (toxicity, self-harm, PII) → LLM safety judge (temp=0) for nuanced semantics → refusal / rewrite.
    4) Structured output constraints: JSON/function schemas + strict validation to reduce free-form exploit surface; reject malformed responses.
    5) Continuous adversarial training refresh: incorporate newest jailbreak styles (role-play, obfuscated, multilingual) into preference / DPO datasets.
    6) Provenance labeling: tag segments as system | user | retrieved | tool; ignore or down-weight instructions from untrusted segments.
    7) Tool mediation & rationale verification: require explicit reason + intended tool; secondary checker validates rationale vs policy before execution.
    8) Streaming safety monitor: parallel classifier/LLM inspects partial generation; early-abort on unsafe trajectory.
    9) Session escalation: throttling & stricter guardrails after repeated near-miss jailbreak attempts; potential human review trigger.
    10) Automated red-team harness: mutate & fuzz known jailbreak prompts; track bypass rate trend; feed escapes back into training.

  - KPIs:
    - Jailbreak bypass rate = disallowed outputs released / attempted high-risk prompts (trend ↓).
    - False refusal rate (benign prompts refused) – maintain acceptable utility/safety trade-off.
    - Mean time to patch (detection → mitigation deployed).
    - Tactic coverage (# distinct jailbreak tactic classes exercised in last 7 / 30 days).
    - System prompt displacement attempts blocked (% prevented).

  - Practical defaults:
    - Temperature ≤0.4 for safety-critical endpoints; max token caps; standardized refusal template language.
    - Mandatory citations in RAG answers (uncited injected instructions stand out & can be filtered).
    - Nightly adversarial regression suite; fail build if bypass rate exceeds threshold.

  - Pitfalls:
    - Overfitting to static regex/keywords → obfuscation bypasses.
    - Excessive refusals harming UX → users escalate creativity.
    - Single safety classifier → concept drift blind spot.
    - Ignoring multilingual / code-mixed / emoji-based attacks.

  - Mnemonic:
    - Adversarial (big circle) ⊃ Jailbreak (policy violation). Injection = door. Poisoning = upstream seed. Red-team = radar.

### Deployment & MLOps
- How to design a scalable LLM inference system?

  - Objectives: minimize tail latency (p95/p99), maximize throughput (tokens/sec/GPU), reduce cost ($/1k tokens), maintain reliability (SLOs), secure multi-tenancy, enforce safety & observability.

  - Core components:
    1) API Gateway: auth (API key/JWT), rate limiting, idempotency keys, request normalization.
    2) Orchestrator / Router: model selection (policy-based, quality tiers), batching queue, A/B & canary, fallback logic.
    3) Pre-processing: prompt templating, context assembly (RAG retrieval + compression), safety pre-check, tokenization.
    4) Inference Workers (accelerated): dynamic micro-batching, KV cache (reuse across turns), quantized or mixed-precision model runtime (vLLM / FasterTransformer / TensorRT-LLM).
    5) Post-processing: streaming assembly, safety post-filter, citation validation, formatting (JSON/schema), truncation.
    6) Caching Layers: prompt/full-completion, prefix/KV sharing, embedding cache, metadata TTL logic.
    7) Observability: metrics (latency, queue depth), traces, structured logs, usage billing, anomaly detection.
    8) Feedback & Data Loop: user ratings, red-team outcomes, regression evaluation harness, fine-tune sample capture.

  - Request lifecycle (happy path):
    Ingest → Auth/Quota → Normalize & Hash → Prompt Cache lookup → (Miss) Retrieve (parallel dense+sparse) → Compress context (MMR / summarization) → Assemble prompt → Batch Scheduler → Forward pass (speculative or standard) → Stream tokens (with online safety scanning) → Post-filter & format → Emit → Log & metrics & feedback enqueue.

  - Latency & throughput tactics:
    - Dynamic micro-batching: accumulate for ~5–20 ms or batch-size limit; pack by similar token lengths to reduce padding.
    - Speculative decoding: small draft model proposes k tokens; large model verifies → 1.3–2× speedup typical.
    - Multi-query / grouped-query attention to reduce K/V duplication per head.
    - FlashAttention / fused kernels; ensure overlap compute/IO; pin memory.
    - Prefix/KV cache reuse for shared system prompts or conversation history.

  - Memory & model efficiency:
    - Quantization: INT8/FP8/4-bit (QLoRA or AWQ/GPTQ) for higher batch or longer context; watch accuracy regression (evaluate golden set).
    - Mixed precision (BF16/FP16) vs FP32; gradient-free inference avoids optimizer states.
    - Sharding: tensor/pipeline parallel for > single GPU capacity; prefer fewer partitions for lower latency if memory allows.
    - KV eviction: LRU across sessions when memory watermark reached; track cache hit ratio.

  - Caching strategy:
    - Prompt exact hash (normalized JSON/system prompt stable) → completion.
    - Prefix dedup: detect longest common prefix in active batch and compute once.
    - Embedding cache (content hash) for RAG ingestion & re-retrieval.
    - Tiered: RAM (hot), Redis/memcached (warm), object store (cold). Evict on model version or policy change.

  - Routing & multi-model policy:
    - Quality tiers: fast small model for straightforward queries, larger model for complex reasoning (decided by classifier or heuristics: length, complexity keywords, required factuality).
    - Canary rollout: <5% traffic; compare latency, refusal accuracy, task metrics; auto rollback on regressions.
    - Fallback: timeout or error threshold triggers smaller backup model or different region.

  - Retrieval (RAG) integration:
    - Parallel sparse (BM25/SPLADE) + dense (vector) retrieval; fuse scores (RRF or weighted) → re-rank top-N (cross-encoder) → contextual compression (drop low-contribution sentences) → enforce token budget.
    - Query intent classifier: skip retrieval for chit-chat / personal preference queries.
    - Citation enforcement: tag chunks with IDs; post-check answer cites only retrieved IDs.

  - Safety & governance:
    - Pre & in-stream filtering (toxicity, self-harm, PII, jailbreak heuristics) with early abort on severity.
    - Policy templates: jurisdiction-aware content filters (regional compliance toggles).
    - Logging: store minimal hashed identifiers; retention TTL; secure vault for encryption keys.
    - Tool sandbox: network/file isolation, execution time caps, JSON schema arg validation.

  - Observability & metrics:
    - Core: p50/p95/p99 end-to-end, queue wait, model compute time, tokens/sec/GPU, batch size distribution, cache hit %, retrieval latency breakdown, safety block rate.
    - Quality: refusal correctness (true positive vs false positive), groundedness (if RAG), answer length distribution, hallucination proxy metrics.
    - Cost: $/1k output tokens, GPU utilization %, idle capacity %, cache savings (token skip %).
    - Tracing: correlation ID across gateway → retrieval → inference → safety; sample traces for long-tail latency.

  - Reliability patterns:
    - Circuit breakers around retrieval or external tools (open after error spike, degrade gracefully).
    - Warm standby replicas (preloaded weights) to absorb bursts; rolling deploy with pre-warm handshake.
    - Graceful degradation ladder: (1) turn off speculative decoding, (2) reduce max_new_tokens, (3) route to smaller model, (4) switch to offline queue.
    - SLA partitions: reserve GPU slices for premium tenants; avoid noisy neighbor starvation.

  - Cost optimization levers:
    - Right-size hardware portfolio: H100/A100 for large context; L4 / consumer GPUs for smaller models / embedding tasks.
    - Mixed pool autoscaling: scale cheaper nodes for baseline; scale high-end only when high complexity demand.
    - Prune rarely used models; lazy load on first request (with TTL unload timer).
    - Nightly efficiency report: tokens/sec/GPU, cache hit improvement, speculative success rate.

  - Feedback & continuous improvement:
    - Golden evaluation set (diverse tasks, safety cases) run on each build; block deploy on metric regression thresholds.
    - Red-team prompt suite (jailbreak/injection) executed nightly; track bypass rate trend line.
    - User rating & flag pipeline feeding labeled dataset; active learning surface (low confidence + high impact samples).

  - Capacity planning formula (simplified):
    - Required GPUs ≈ (Incoming_tokens_per_sec / (Effective_tokens_per_sec_per_GPU × Utilization_target))
    - Effective tokens/sec accounts for average batch size × model throughput × (1 - overhead_fraction).
    - Maintain headroom (e.g., 30%) for burst + failover.

  - Example starter SLOs (tune per product):
    - p95 latency (short prompt <1k tokens) < 800 ms.
    - Availability (no 5xx) ≥ 99.5% monthly.
    - Safety false negative (audited) < 0.1%; false refusal < 3%.
    - Cache hit (prompt + KV) ≥ 25% after warm-up.

  - Scalability mnemonic: BATON-CRAFT
    - Batching, Autoscaling, Token efficiency, Orchestration, Networking locality
    - Caching, Retrieval optimization, Alignment (safety), Feedback loop, Token economics

  - Common pitfalls:
    - Ignoring queue wait vs model compute breakdown.
    - Over-long prompts (history bloat) harming throughput and cost.
    - Stale caches after model or policy update (missing invalidation hooks).
    - One-size-fits-all model (no routing) inflating cost.
    - Lack of golden regression → silent quality regressions post optimization.

  - Implementation bootstrap (phased):
    1) Baseline: single model + streaming + metrics.
    2) Add dynamic batching + prompt cache + tracing.
    3) Introduce RAG & safety cascade.
    4) Multi-model router + canary + fallback.
    5) Performance: speculative decoding, quantization, KV sharing.
    6) Governance: nightly eval gates, red-team automation, cost dashboards.
    7) Advanced: autoscale heuristics tuning, A/B quality scoring, active learning feedback ingestion.


- Compare hosting on OpenAI API / Bedrock / Azure AI Foundry vs self-hosted LLaMA/Mistral.

  - Framing: Choice = (Managed proprietary / managed multi‑model) vs (Self-hosted open weights). Trade-off triangle: (Control & Customization) vs (Operational Burden) vs (Time-to-Value & Managed Compliance).

  - Managed API (OpenAI / Azure OpenAI / Bedrock / Azure AI Foundry):
    - Pros:
      - Time-to-value: zero infra; SLA-backed uptime; autoscaling handled.
      - Cutting-edge models: immediate access to latest GPT / Claude / Sonnet / Mistral variants (Bedrock multi-vendor, Azure integrates OpenAI + Phi + others, Foundry adds orchestration & eval tooling).
      - Ecosystem features: safety filters, logging dashboards, key-based auth, quota management, usage analytics, latent eval tools (Azure AI Safety, Bedrock Guardrails).
      - Compliance & certifications: SOC2, ISO, HIPAA, GDPR data processing agreements already in place; regional hosting options.
      - Reliability primitives: global region failover, managed retries, model version pinning.
      - Security defaults: no raw weights exposure; network isolation abstracted; built-in abuse monitoring.
      - Fast iteration: new model versions, modalities (vision, audio), and performance optimizations land automatically.
    - Cons:
      - Cost per token higher (pay margin + premium for frontier R&D); egress & context inflation can be expensive.
      - Limited deep customization: cannot modify architecture, training data, tokenizer, internal safety stack; fine-tuning often restricted or gated.
      - Latency variance: multi-tenant queueing; region congestion; sometimes higher p95 vs tuned on-prem.
      - Data governance concerns: although vendors promise not to train (when configured), some enterprises still want absolute isolation; need contractual clarity on retention & logging.
      - Vendor lock-in risk: proprietary model behaviors & eval metrics; prompt engineering not always portable; rate-limit & pricing changes outside control.
      - Observability granularity: limited access to low-level kernel timings, KV cache stats, GPU utilization.

  - Self-hosted open models (LLaMA, Mistral, Mixtral, Phi, Qwen, etc.):
    - Pros:
      - Cost efficiency at scale: amortize GPU/accelerator cost; with > moderate steady token volume, $/1k tokens can drop 2–10× (after infra + ops).
      - Customization: full control of weights (parameter-efficient fine-tuning, domain adaptation, safety alignment, structured decoding tweaks, custom sampling code, retrieval integration in-graph).
      - Privacy & sovereignty: data stays in VPC/on-prem (regulatory or confidential IP use cases); audit every component.
      - Performance tuning: quantization strategy, speculative decoding pair, batching scheduler, hardware topology (NVLink vs PCIe) — all adjustable for p95 reduction.
      - Model diversity: can host multiple tailored distilled variants (e.g., small routing model + domain expert fine-tunes) optimizing cost/performance mix.
      - Safety + policy transparency: implement/iterate custom guardrails (regex + classifier + LLM judge cascade) and measure internals.
      - Offline / air-gapped capability: critical for highly regulated sectors (defense, healthcare, finance) or edge scenarios.
    - Cons:
      - Operational complexity: MLOps pipelines (model packaging, container images, weight distribution, GPU scheduling, autoscaling, hot reload), observability (traces, tokens/sec, memory), incident response.
      - Talent & maintenance: need engineers for CUDA/runtime, security patching, scaling, vulnerability management (e.g., supply chain scanning of containers, driver updates).
      - Upgrade cadence friction: evaluating new open checkpoints takes benchmarking harness & regression tests; risk of model drift/regression between versions.
      - Hidden costs: GPU idle time, overprovisioning for burst, networking (intra-cluster), storage (multiple quantization variants), monitoring stack, patch cycles.
      - Compliance overhead: build/maintain audit logs, encryption-in-transit/at-rest, key management, access controls, data retention policies yourself.
      - Reliability risk: need to architect redundancy, failover, health probes, autoscaling triggers; single-shard outages can degrade service.

  - Dimension-by-dimension comparison:
    - Latency (median): Self-hosted (properly tuned) can be lower (no cross-tenant queue) especially for small/quantized models; managed can shine for large frontier models with proprietary kernel optimizations.
    - Tail latency (p95/p99): Managed vendors invest heavily in smoothing; self-hosting requires careful batching caps + backpressure + multi-queue scheduling.
    - Cost at low volume: Managed cheaper (no fixed infra). Cost at high stable volume: Self-hosted wins after breakeven (often when GPU utilization > 40–50% sustained, depending on hardware amortization and ops cost).
    - Feature velocity: Managed highest (new modalities, updates). Self-hosted requires manual integration.
    - Model optionality: Bedrock/Azure multi-vendor catalog; self-hosted unlimited open-source but each adds ops overhead.
    - Fine-tuning depth: Self-hosted unlimited (full/PEFT). Managed: often adapters or limited aspects; some vendors restrict dataset types & size.
    - Governance / audit: Self-hosted offers deepest transparency; managed provides standardized attestations but less internal process visibility.
    - Data residency: Self-hosted precise control (choose region/cloud/on-prem). Managed depends on region availability & contractual terms.
    - Lock-in: Managed encourages dependency on proprietary behavior & pricing; self-hosted: risk is internal complexity, but portability between clouds possible.
    - Security surface: Managed reduces attack surface (no direct weight/infra mgmt). Self-hosted expands surface (supply chain, container, GPU driver, lateral movement risk) needing hardening.
    - Observability depth: Self-hosted full (kernel traces, cache metrics). Managed limited to exposed metrics.
    - Scaling agility: Managed near-infinite elastic (within quotas). Self-hosted requires capacity planning, pre-warming, cluster autoscale.
    - SLA & support: Managed provides contractual SLA / enterprise support tiers. Self-hosted: internal SRE / vendor hardware support only.

  - When to favor Managed APIs:
    - Early-stage / MVP or uncertain demand; low eng bandwidth.
    - Need frontier reasoning/creative performance (latest GPT/Claude) quickly.
    - Strict compliance requirement already met by vendors (FedRAMP Moderate/High, HIPAA BAA) reducing internal audit scope.
    - Highly spiky traffic patterns; desire to externalize scaling risk.
    - Rapid experimentation across multiple model families for evaluation.

  - When to favor Self-hosted LLaMA/Mistral:
    - Predictable or large token throughput where cost optimization matters.
    - Domain adaptation & custom safety policy beyond vendor allowances.
    - Sensitive IP / regulated data requiring strict isolation / data minimization guarantees.
    - Need deterministic or modified decoding (grammar constraints, structured generation libs) not supported by vendor.
    - Multi-agent or retrieval pipeline optimization requiring deep control of KV cache, batching scheduler, or custom kernels.

  - Hybrid Strategy (common in mature orgs):
    - Routing layer: simple/low-risk queries → small self-hosted model; complex reasoning → managed frontier model.
    - Fallback: self-hosted primary (cost); managed as reliability fallback or overflow (burst spillover during peak).
    - Evaluation: use managed high-quality model as judge/critic to improve self-hosted fine-tunes (distillation). Avoid sending sensitive user content to judge if privacy constraints.
    - Progressive migration: start on managed → collect logs/datasets → fine-tune open model → gradually shift traffic behind quality guardrails.

  - Cost modeling (simplified):
    - Managed: Cost ≈ Σ(tokens_in + tokens_out)/1k × provider_rate + premium features (guardrails, evals). Zero fixed GPU capex.
    - Self-hosted: Cost ≈ (GPU_hourly_rate × hours × utilization_factor) + storage + ops labor + networking + amortized capital. Effective $/1k tokens ↓ as utilization ↑ and quantization improves throughput.
    - Breakeven heuristic: If (Monthly managed spend) > (1.4–1.6 × fully-loaded self-host hosted cluster cost) for ≥2–3 months, evaluate migration.

  - Risk considerations:
    - Managed: sudden pricing changes, API deprecations, region outages, model behavior shifts on silent upgrades (mitigate: version pin, regression tests).
    - Self-hosted: security patches lag, model performance regression from poorly validated fine-tune, scaling incidents (OOM, GPU fragmentation), single-team bus factor.

  - Due diligence checklist:
    - Managed: data usage policy, retention window, region & failover strategy, rate limit escalation path, versioning & deprecation schedule, safety feature roadmap.
    - Self-hosted: benchmark harness (latency/quality/safety), infra as code, autoscaling policy, observability (metrics/traces/log retention), security hardening (image scanning, RBAC, network policies), disaster recovery (backup weights, multi-zone).

  - Decision mnemonic: FACTORS → (F)ine-tuning depth, (A)nnual token volume, (C)ompliance isolation, (T)alent available, (O)bservability needs, (R)outing complexity, (S)ensitivity of data.

  - Interview framing example answer (concise):
    - "Start with managed (OpenAI/Bedrock/Azure) for speed, frontier quality, and compliance offload. As usage stabilizes and customization or cost efficiency become priorities, introduce a self-hosted Mistral/LLaMA fine-tuned variant behind a router. Maintain hybrid fallback for resilience; continuously benchmark latency, quality (groundedness, refusal accuracy), and $/1k tokens to guide traffic shifts."

  - Pitfalls when migrating:
    - Comparing raw quality without domain fine-tuning (open model looks weaker initially).
    - Underestimating ops overhead (24/7 on-call, GPU monitoring) → hidden cost.
    - Ignoring eval drift: not updating golden set to reflect new domain behaviors.
    - Incomplete safety parity: assuming managed guardrails replicated automatically; must rebuild cascade.

  - Practical migration plan (phased):
    1) Log & label production prompts/responses on managed (privacy scrubbed).
    2) Train PEFT fine-tunes of chosen open model on curated dataset + safety preference data.
    3) Stand up inference stack (quantized, batching, cache) in staging; run side-by-side evaluations (quality, safety, latency, cost).
    4) Canary route 5% low-risk traffic with guardrails + shadow compare responses; monitor deltas.
    5) Ramp traffic gated by SLO adherence & safety metrics; keep managed as overflow + regression oracle.
    6) Periodic re-benchmark frontier managed models; decide if hybrid ratio still optimal.

  - Summary: Managed APIs optimize for speed, breadth, compliance, and cutting-edge capability; self-hosting optimizes for cost control, deep customization, privacy, and performance tuning. Mature systems blend both with a routing & evaluation layer.


 - How does quantization (INT8/4-bit) help in deployment? Trade-offs.
   
   - Core idea: Represent model weights (and sometimes activations + KV cache) with lower precision integers (INT8, INT4, or FP8/FP16 hybrids) instead of FP16/FP32 to reduce memory footprint and increase effective throughput (tokens/sec) by fitting more model copies or larger batch/context into limited GPU memory & memory bandwidth.

   - Why it helps (benefits):
     - Memory reduction: FP16 → INT8 halves weight storage; INT4 (weight-only) gives ~4× reduction vs FP16 (plus minor scale metadata). Enables hosting larger context windows or multiple concurrent models on same hardware.
     - Bandwidth bound speedup: Transformer inference is often memory bandwidth limited (loading weights each layer). Smaller weights = fewer bytes transferred → better arithmetic intensity; can yield 1.2–1.8× wall-clock speed improvements (depends on kernels & hardware support).
     - Higher batch & sequence length: freed VRAM allows larger micro-batches or longer prompts before OOM, improving GPU utilization & amortizing overhead.
     - Cost efficiency / consolidation: fewer GPUs for same traffic or more QPS per GPU lowers $/1k tokens; can enable edge / CPU / low-power deployments (8-bit/4-bit AWQ + CPU int8 matmuls).
     - Enables parameter-efficient fine-tuning (QLoRA): keep 4-bit base weights frozen, train small rank adapters in higher precision → big VRAM savings during training.

   - Key quantization types (inference focus):
     1) Post-Training Dynamic INT8 (activation dynamic range estimated on the fly) – easy, moderate accuracy.
     2) Post-Training Static/Calibrated INT8 (calibration dataset to fix scales) – better accuracy; requires representative sample.
     3) Weight-only quantization (W8A16 or W4A16): quantize weights; keep activations in FP16/BF16 (dominant for LLMs). Methods: GPTQ (error-aware per-column reconstruction), AWQ (Activation-Aware Weight Quantization selects salient channels to keep higher precision / better scaling), RPTQ, SpQR (semi-structured).
     4) 4-bit formats: INT4, NF4 (normal float 4)—data-driven codebook capturing distribution of weight values (QLoRA). NF4 preserves distributional shape, improving accuracy over uniform INT4.
     5) FP8 (E4M3 / E5M2): emerging hardware (H100, MI300) mixed-precision; retains floating semantics with tighter dynamic range; often used for activations + weights with scaling groups.
     6) Quantization-Aware Training (QAT): simulate quantization during fine-tuning to adapt weights; best accuracy but more training cost.
     7) KV cache quantization: compress stored key/value tensors (often INT8/FP8) to reduce memory for long chats; can add minor perplexity increase if naive.

   - Implementation building blocks:
     - Granularity: per-tensor, per-channel (column-wise for matmul W), per-group (e.g., 128-group). Finer granularity → better accuracy, slightly larger scale metadata.
     - Zero-points & scales: map real range [min, max] to integer range; outlier channels can blow range causing precision loss (handled by channel-wise scaling or outlier splitting to higher precision paths).
     - Symmetric vs asymmetric: symmetric (range [-S, S]) simplifies hardware; asymmetric improves dynamic range for skewed distributions (less common for weight-only LLM quantization).
     - Activation outliers: early embedding and attention output layers may have heavy tails; often kept in higher precision (FP16) while middle MLP weights are quantized.
     - SmoothQuant / Outlier Suppression: redistribute activation range into weights (pre-processing) enabling activation quantization with less error.

   - Accuracy considerations:
     - Sensitivity varies by layer: embeddings, final LM head, layer norms, attention output projections are more sensitive.
     - 8-bit weight-only: near-lossless (<0.1 perplexity delta) typically.
     - 4-bit weight-only (AWQ/GPTQ/NF4): small perplexity increase (e.g., +0.2–0.8) but acceptably low for many chat/RAG use cases; may slightly degrade long-range reasoning or factual recall.
     - Stacked degradations: combining aggressive quantization + long context + low-rank fine-tune can accumulate quality loss; need composite eval.
     - Evaluate on: perplexity (wiki subset), task benchmarks (MMLU subset, domain QA), safety refusal accuracy (ensure not regressing), hallucination proxy metrics if relevant.

   - Performance notes:
     - INT8 widely supported (TensorRT-LLM, FasterTransformer, bitsandbytes, vLLM partial). INT4 kernels vary; speed gains depend on vendor libs (some show memory savings but limited compute speedup due to dequant overhead).
     - FP8 emerging: better balance (accuracy close to FP16 with improved throughput) but hardware limited; good for end-to-end weight+activation quantization.
     - Dequantization overhead: naive per-token dequant can offset gains; fused matmul kernels perform on-the-fly dequant in registers to minimize cost.
     - KV cache quantization: reduces memory scaling O(L * d * layers); particularly impactful for long conversation or streaming with large concurrency.

   - Trade-offs / risks:
     - Quality regression: subtle reasoning, code generation, math precision degrade earlier than generic chat; 4-bit may underperform for complex multi-hop.
     - Calibration data dependency: non-representative calibration → scale mismatch → accuracy cliff.
     - Tool & ecosystem maturity: debugging harder (can’t just inspect FP weights); some ops (layer norm, softmax) still in higher precision → limited total benefit.
     - Numerical stability: extreme logits (rare tokens) susceptible to rounding; could alter decoding probabilities, impacting determinism.
     - Fine-tuning constraints: training on quantized weights needs special flows (QLoRA) to avoid precision accumulation error; full QAT more compute.
     - Hardware variance: speedups inconsistent across GPU generations (INT4 acceleration better on some architectures; others bottleneck on memory anyway).
     - Safety / bias drift: minor shifts in distribution might change borderline refusal thresholds or classifier embeddings; must re-run safety eval.
     - Maintenance overhead: multiple quantized variants (4-bit, 8-bit, FP16) multiplies artifact management + CI benchmarks.

   - Evaluation checklist (before/after quantization):
     1) Perplexity delta (≤ +0.5 acceptable for many prod chat tasks; stricter for high-precision domains).
     2) Task accuracy sample set (≥ 95–98% of FP16 baseline on key tasks).
     3) Latency & throughput: tokens/sec, p95 latency improvement (report both cold & warm).
     4) Memory usage: VRAM resident size + max batch size / context length at same OOM threshold.
     5) Safety metrics: refusal correctness, toxicity classification alignment.
     6) Regression tests for determinism (if required) with fixed seeds & decoding params.

   - Practical deployment patterns:
     - Start with 8-bit weight-only (near lossless) → validate → move hot paths to 4-bit for cost-critical endpoints if quality passes SLO.
     - QLoRA fine-tune: base model in NF4 + adapters in FP16/BF16; after tuning, optionally merge (if method supports) or keep adapters separate to swap.
     - Mixed policy: High-tier (enterprise) traffic served by FP16/BF16 model for maximal quality; bulk/low-priority traffic by 4/8-bit variant.
     - Maintain golden output corpus & nightly diff: compare quantized vs reference answers (semantic similarity + safety classification) to catch drift.
     - Use grouping (e.g., AWQ group size 128) to balance accuracy vs overhead; experiment with group size & rounding mode.

   - Selection guide:
     - If memory bound & small quality tolerance → INT4 (AWQ/GPTQ) or NF4 (QLoRA) weight-only.
     - If want minimal risk & quick win → INT8 weight-only.
     - If hardware supports & need further gains with stability → FP8 mixed precision.
     - If extreme memory constraints (edge) → 4-bit + aggressive KV cache quant + context compression.
     - For multi-lingual / code tasks (sensitive) → stay at INT8 unless validated.

   - Interactions with other optimizations:
     - Speculative decoding + quantization: ensure draft & target models have compatible output quality; quantization noise can reduce acceptance rate slightly.
     - LoRA + quantization: freeze quantized base, train LoRA in higher precision (QLoRA pipeline) → preserves adaptation quality; evaluate merging carefully.
     - Caching: quantized weights reduce load time; still keep KV cache maybe in FP16 unless memory pressure severe; quantizing KV too aggressively can hurt long-context coherence.

   - Common pitfalls:
     - Evaluating only perplexity (not user-facing tasks) → hidden reasoning regression.
     - Ignoring activation outliers leading to saturation & degraded layers.
     - Not isolating random seed differences when comparing latencies (variance noise misattributed to quantization).
     - Serving quantized & full precision with same autoscaler assumptions (different memory footprints affect scaling triggers).
     - Skipping recalibration after fine-tune (distribution shift invalidates earlier scales).

   - Metrics to monitor post-deploy:
     - Throughput gain % vs baseline, VRAM utilization %, acceptance rate in speculative decoding, safety refusal accuracy delta, user satisfaction / rating drift.
     - Quality guardrail: roll back if key task accuracy < threshold (e.g., <97% of baseline) for two consecutive evaluations.

   - Quick interview summary:
     - "Quantization shrinks model weight & activation precision (INT8/INT4/FP8) to cut memory & bandwidth, increasing batch size and lowering latency/cost. Weight-only INT8 is near-lossless; 4-bit (GPTQ/AWQ/NF4) gives ~4× memory reduction with small accuracy trade-offs. Risks: quality regression on reasoning, calibration errors, limited kernel support, safety drift. Mitigate with layer-wise/group scales, outlier handling (AWQ/SmoothQuant), and rigorous before/after evaluation across perplexity, task accuracy, safety, and latency."

   - Mnemonic: SCALE-Q → (S)peed, (C)ost, (A)fford larger context, (L)ayer sensitivity matters, (E)valuate thoroughly, (Q)uality trade-off.

- Difference between online vs batch inference for LLM workloads.

  - Core distinction: Online (real-time) inference serves individual requests with immediate response requirements; batch inference processes multiple inputs together, optimizing for throughput over latency. Different optimization strategies, infrastructure patterns, and cost models.

  - Online inference characteristics:
    - Request pattern: synchronous user-facing queries (chat, API calls, interactive apps) requiring <1–3s response times.
    - Optimization goal: minimize p95/p99 latency while maintaining reasonable throughput (tokens/sec/GPU).
    - Architecture: always-on inference servers, dynamic batching (micro-batches of 1–32 concurrent requests), request queuing with SLA timeouts.
    - Resource usage: consistent GPU memory allocation, idle capacity for burst handling, warm model weights, KV cache retention across conversations.
    - Scaling: horizontal (more GPU replicas) + vertical (faster GPUs/more VRAM per instance); autoscaling based on queue depth & latency SLOs.
    - Caching: aggressive prompt/completion caching, prefix sharing, KV cache reuse to cut repeated computation.
    - Concurrency model: async request handling, streaming responses, connection pooling, load balancing across replicas.

  - Batch inference characteristics:
    - Request pattern: offline workloads (document processing, dataset analysis, periodic reports, model evaluation) where results can wait minutes/hours.
    - Optimization goal: maximize total throughput (tokens/hour/GPU) and minimize cost per token processed.
    - Architecture: job queue systems (Kubernetes jobs, Airflow, SQS+Lambda), large static batches (64–512+ prompts), scheduled execution.
    - Resource usage: burst GPU allocation during jobs, can preempt lower-priority tasks, optimize for GPU utilization % over idle time.
    - Scaling: can use spot instances, preemptible VMs, and cheaper hardware (older GPUs, CPUs for smaller models); schedule during off-peak hours.
    - Parallelization: data parallelism (split large batches across GPUs), pipeline parallelism (multi-stage processing), embarrassingly parallel workloads.
    - Storage integration: direct read/write to object storage (S3, GCS), compressed input formats, result aggregation and reporting.

  - Performance optimization strategies:
    - Online optimizations:
      - Dynamic batching: accumulate requests for 5–20ms to form mini-batches; balance latency vs throughput.
      - Speculative decoding: draft model + verification for 1.3–2× speedup while preserving quality.
      - KV cache management: persist conversation state, prefix sharing for system prompts, LRU eviction.
      - Model quantization: INT8/FP8 for reduced memory + higher batch capacity.
      - Flash attention & fused kernels: optimize attention computation and memory access patterns.
      - Request routing: complexity-based routing (small model for simple queries, large for complex reasoning).
    - Batch optimizations:
      - Large static batches: pack 128–1024+ prompts with similar token lengths to minimize padding waste.
      - Sequence packing: concatenate multiple short sequences into one batch element (with attention masks).
      - Mixed precision: aggressive quantization (INT4) acceptable since no real-time constraints.
      - Pipeline parallelism: overlap data loading, inference, and result writing across multiple stages.
      - Checkpointing: save intermediate results to handle job failures and resume processing.
      - Output streaming: write results incrementally rather than accumulating in memory.

  - Infrastructure & cost patterns:
    - Online serving:
      - Always-on costs: 24/7 GPU reservation even during low traffic; over-provisioning for peak capacity.
      - Premium hardware: H100/A100 for low latency, high-memory GPUs for large context windows.
      - Multi-region: global deployment for latency optimization, cross-region failover, CDN integration.
      - Monitoring: real-time metrics (p95 latency, error rates, queue depth), alerting, SLO tracking.
      - Auto-scaling: responsive but conservative (avoid cold start delays); maintain warm standby capacity.
    - Batch processing:
      - On-demand costs: spin up resources only when jobs run; use spot/preemptible instances (50–90% cost savings).
      - Cost-optimized hardware: older GPU generations, CPU inference for smaller models, mixed instance types.
      - Single-region: process where data resides, minimize egress costs, leverage reserved capacity.
      - Scheduling: run during off-peak hours, coordinate with other workloads, batch multiple datasets together.
      - Elasticity: aggressive scaling (100s–1000s of instances), tolerate startup delays, optimize for total job completion time.

  - Use case mapping:
    - Online inference ideal for:
      - Interactive chatbots, customer support agents, real-time content generation.
      - API services integrating with web/mobile apps requiring <2s response times.
      - Code completion, search query expansion, real-time translation, live transcription.
      - Multi-turn conversations requiring state retention (KV cache) and personalization.
      - A/B testing, canary deployments, and gradual rollouts requiring precise traffic control.
    - Batch inference ideal for:
      - Document corpus processing (summarization, extraction, classification of millions of files).
      - Periodic report generation (weekly insights, monthly analyses, quarterly summaries).
      - Dataset labeling, synthetic data generation, model evaluation on large test sets.
      - ETL pipelines with LLM-based transformation (clean text, extract entities, enrich metadata).
      - Research experiments, hyperparameter sweeps, fine-tuning data preparation.
      - Compliance scanning (PII detection, policy violation checks across document repositories).

  - Hybrid patterns (common in practice):
    - Tiered serving: online for interactive, near-real-time queue for "fast batch" (results in 1–10 mins), offline batch for bulk processing.
    - Overflow routing: online capacity exhausted → route excess traffic to batch queue with async notification.
    - Preprocessing pipeline: batch jobs precompute embeddings/summaries → online service fetches precomputed results.
    - Model distillation: use batch processing with large model to generate training data for faster online model.
    - Feature engineering: batch jobs create prompt templates, retrieve context, generate training samples → online inference uses prepared inputs.

  - Technology stack differences:
    - Online serving stacks:
      - Inference servers: vLLM, TensorRT-LLM, FasterTransformer, Triton Inference Server.
      - Orchestration: Kubernetes (HPA, VPA), Docker Swarm, managed services (AWS SageMaker, GCP Vertex AI).
      - Load balancing: NGINX, HAProxy, cloud load balancers with health checks and circuit breakers.
      - Monitoring: Prometheus, Grafana, DataDog, custom metrics for tokens/sec, cache hit rates.
    - Batch processing stacks:
      - Job orchestration: Apache Airflow, Kubernetes Jobs, AWS Batch, Azure Batch, Google Cloud Dataflow.
      - Data processing: Apache Spark (for large-scale preprocessing), Dask, Ray (distributed Python).
      - Storage integration: direct S3/GCS APIs, data lake formats (Parquet, Delta Lake), streaming (Kafka, Pub/Sub).
      - Resource management: spot fleet management, preemptible instance pools, cluster autoscalers.

  - Latency vs throughput trade-offs:
    - Online serving: prioritize consistent low latency → smaller batches, more replicas, premium hardware, aggressive caching.
    - Batch processing: prioritize total throughput → larger batches, fewer replicas, cost-optimized hardware, minimal caching overhead.
    - SLO examples:
      - Online: p95 latency <800ms, availability >99.9%, burst capacity 3× baseline.
      - Batch: complete 10K documents within 4 hours, cost <$0.50 per 1K tokens, fault tolerance with <5% job failure rate.

  - Monitoring & observability:
    - Online metrics: request latency distribution, throughput (RPS, tokens/sec), error rates, queue depth, cache hit ratios, GPU utilization.
    - Batch metrics: job completion time, total throughput (tokens/hour), cost per token, resource utilization over time, failure/retry rates.
    - Shared concerns: model quality drift, safety metrics (toxicity rate), resource costs, data lineage.

  - Development & testing considerations:
    - Online: staging environment mimicking production load, canary deployments, blue-green rollouts, real-time monitoring.
    - Batch: development can use smaller sample datasets, integration testing with full pipeline, regression testing on golden datasets.
    - Shared: version control of models/configs, reproducible builds, rollback capabilities, audit logs.

  - Decision framework:
    - Choose Online when: user-facing interaction, latency <3s required, traffic patterns unpredictable, conversation state needed, real-time feedback loops.
    - Choose Batch when: processing existing datasets, results can wait >10 minutes, cost optimization critical, embarrassingly parallel workload, scheduled/periodic execution.
    - Consider Hybrid when: mixed workload types, need overflow capacity, preprocessing + real-time serving, cost vs latency optimization across different user tiers.

  - Common migration patterns:
    1) Start online-only → identify batch-suitable workloads → offload to batch processing → optimize costs.
    2) Batch-first → add online endpoint for interactive features → gradually shift more traffic online as latency requirements tighten.
    3) Build batch foundation (data pipelines, evaluation) → add online API layer → scale online based on demand patterns.

  - Cost optimization strategies:
    - Online: right-size instance types, implement request-based autoscaling, use caching aggressively, employ model routing (small model for simple requests).
    - Batch: maximize spot instance usage, schedule during off-peak hours, optimize batch sizes for hardware, use data-local processing.
    - Both: monitor GPU utilization, use quantized models where quality permits, implement circuit breakers to avoid cascade failures.

  - Interview summary: "Online inference optimizes for low-latency user-facing requests with always-on infrastructure, dynamic batching, and aggressive caching. Batch inference optimizes for high-throughput offline workloads using larger static batches, cost-optimized hardware, and scheduled execution. Choose based on latency requirements, traffic predictability, and cost constraints. Many production systems use hybrid approaches with tiered serving and overflow routing."

  - Selection mnemonic: BOLT → (B)atch for Bulk processing, (O)nline for user Operations, (L)atency vs throughput trade-offs, (T)iered hybrid approaches.

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


--Core Python (must-know)

What are Python’s data structures (list, tuple, dict, set) and when would you use each?

How does Python handle memory management (Garbage Collection, reference counting)?

Difference between deepcopy and shallow copy?

Explain Python decorators and give an example use case in ML pipelines.

What are generators? Why are they useful when working with large datasets for training?

Explain how Python’s Global Interpreter Lock (GIL) affects multi-threading.

What is the difference between @staticmethod, @classmethod, and instance methods?

How would you handle exceptions in a long-running model training job?

Explain context managers (with statement) and where they’re useful in ML workflows.

Difference between is and == in Python.

--NumPy / Pandas for Data Handling

How do you efficiently handle large datasets in Pandas without running out of memory?

Difference between loc[], iloc[], and ix[] (deprecated) in Pandas.

How does broadcasting work in NumPy?

How to vectorize operations in NumPy instead of using loops?

Given a huge CSV file, how would you load it in chunks and preprocess it for model training?
