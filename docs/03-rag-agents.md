# 3. RAG & AI Agents

## RAG Problem & Solution
Problem: LLMs hallucinate, lack up-to-date proprietary facts, have finite context, no citations/access control.
Solution: Retrieve top‑k relevant chunks at query time → ground generation with sources (fresher knowledge, fewer hallucinations, traceability, scoped access).
Trade-offs: retrieval quality bottleneck; added latency/cost; prompt injection risk; evaluation expands (retrieval + generation metrics).

## Chunking Strategies
- Fixed token windows with overlap.
- Natural boundary (paragraphs, headings, sentences).
- Structure-aware (code functions/classes; keep tables intact; notebook cells).
- Hierarchical parent-child (large parent 800–1500 + leaf 150–400 tokens).
- Query-time expansion of neighbors; contextual compression to prune irrelevant sentences.
- Sizes: prose 200–400 tokens (10–20% overlap), FAQs 80–200, code 50–120 (function-level), tables whole.
Evaluate: recall@k, answer faithfulness; tune chunk & overlap via grid search.

## Hybrid / Multi-Vector Retrieval
- Sparse + Dense fusion (BM25/SPLADE + embeddings) improves recall of rare terms & semantics.
- Multi-vector (ColBERT late interaction, per-field vectors) for finer granularity.
- Fusion: weighted normalized scores α·dense + (1−α)·BM25 or RRF; re-rank with cross-encoder.
- Trade-offs: more latency/storage; tune α; memory heavy for per-token vectors.

## Re-Ranking Models
- Cross-encoder (BERT) highest precision; joint encode query+passage; use on top N (e.g., 100) keep k (e.g., 10).
- ColBERT late interaction balances precision/efficiency; token-level MaxSim.
- Distilled/light models for latency-sensitive paths.
- Metrics: NDCG/MRR/Recall@k; ensure latency budget respected.

## AI Agent Frameworks
- CrewAI: opinionated multi-agent crews (roles, tasks); quick prototypes; less granular control.
- LangGraph: graph/state-machine orchestration; explicit nodes/edges, persistence, retries; production reliability.
- AutoGen: conversational multi-agent message passing; strong code execution & debate loops; research friendly.
Selection: prototypes → CrewAI; complex production flows → LangGraph; experimental debate/code loops → AutoGen.

## Tool / Function Calling
Structured function calls (name + JSON args) extend model capabilities (APIs, DB, retrieval, code execution). Enables reasoning-act-observe loops (ReAct, Plan-Execute).
Design: clear descriptions, JSON Schema constraints, output validation, tool allowlists, retries/backoff, logging.
Safety: sandbox execution, argument validation, limit steps, detect injection in retrieved text.

## Agent Memory Design
Memory types: short-term (recent turns), episodic (events), semantic (facts/preferences), tool/cache (expensive results), shared blackboard (multi-agent).
Read pipeline: detect need → query vector store/KV/DB → re-rank/diversify (MMR) → compress.
Write pipeline: extract candidates → salience gating → normalize schema (id, type, ts, text, embedding) → persist → periodic reflection summarization.
Retention: sliding windows, TTL, time decay, quotas, summarization; prevent contamination (provenance labels).
Safety: PII detection/redaction, encryption, RBAC, right-to-be-forgotten workflows.
Evaluation: ablation tests (with/without memory) measuring accuracy, groundedness, latency, tool calls.

---
End of RAG & Agents.
