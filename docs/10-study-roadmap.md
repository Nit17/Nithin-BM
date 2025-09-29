# 10. Study & Practice Roadmap

## Phase 1: Foundations (Week 1)
- Read: Core Concepts, Glossary.
- Goals: Explain transformer attention & tokenization differences; compute simple perplexity example.
- Exercise: Build small retrieval index (FAISS Flat) over a handful of docs; run a query.

## Phase 2: Model Adaptation & Mechanics (Week 2)
- Read: LLM Mechanics, Quantization intro sections.
- Implement: Mini LoRA on a small open model (e.g., 7B) for a classification task.
- Evaluate: Compare full FT vs LoRA parameter count & validation loss.

## Phase 3: Retrieval & Agents (Week 3)
- Read: RAG & Agents, Vector Databases sections.
- Build: Hybrid retrieval (BM25 + dense) prototype; add cross-encoder re-ranker.
- Agent: Simple tool-calling loop (calculator + wiki lookup) with memory stub.

## Phase 4: Evaluation & Safety (Week 4)
- Read: Evaluation & Safety.
- Create: Small red-team prompt list; measure refusal accuracy & hallucination rates.
- Implement: Claim grounding check using NLI or lightweight entailment model.

## Phase 5: Deployment & Ops (Week 5)
- Read: Deployment & MLOps, Caching, Inference Modes.
- Prototype: Micro-batching + prompt cache + semantic cache metrics dashboard.
- Quant: Introduce INT8 quantized model; collect latency & quality diffs.

## Phase 6: Advanced Optimization (Week 6)
- Add: Speculative decoding (draft small model) and measure acceptance rate.
- Tune: HNSW parameters for recall vs latency; add PQ compression experiment.
- Safety: Automate nightly regression (recall, groundedness, refusal accuracy).

## Continuous Practice
- Weekly: Add 5 new adversarial prompts & refine defenses.
- Monthly: Re-benchmark embeddings; drift detection.
- Quarterly: Revisit cost per 1K tokens & architectural hot spots.

## Capstone Project Ideas
1. End-to-end RAG QA system with semantic cache + groundedness evaluator.
2. Multi-model router (small vs large) with dynamic complexity classifier.
3. Quantization evaluation harness generating approval report (PPL, tasks, safety).
4. Agent with memory summarization + tool governance & audit log.

## Assessment Checklist
- Can you justify each retrieval stage (fuse → re-rank → compression)?
- Can you quantify quality impact of quantization & caching with metrics?
- Can you detect and mitigate basic prompt injection attempts?
- Can you design a rollback plan for an embedding model migration?

## Suggested Reading Order Recap
Glossary → Core → Mechanics → RAG & Agents → Evaluation & Safety → Deployment → Quantization → Caching → Vector DBs → Roadmap (this file).

---
Use this roadmap iteratively; maintain a journal of latency, quality, and safety experiments.
