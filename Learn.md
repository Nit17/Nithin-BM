
# GenAI Knowledge Index

The previous monolithic content has been decomposed into concise, topic-focused markdown files under the `docs/` directory for easier reading, maintenance, and extension. This file now serves purely as an index.

## Table of Contents
1. Core GenAI Concepts – `docs/01-core-genai.md`
2. LLM Mechanics – `docs/02-llm-mechanics.md`
3. RAG & AI Agents – `docs/03-rag-agents.md`
4. Evaluation & Safety – `docs/04-eval-safety.md`
5. Deployment & MLOps – `docs/05-deployment-mlops.md`
6. Quantization (SCALE-Q) – `docs/06-quantization.md`
7. Inference Modes (BOLT) – `docs/07-inference-modes.md`
8. Caching Strategies (CACHE-STACK) – `docs/08-caching.md`
9. Vector Databases (VECTOR-FIT) – `docs/09-vector-databases.md`

## Mnemonics Summary
- BATON-CRAFT: Scalable inference system design.
- FACTORS: Hosting decision framework.
- SCALE-Q: Quantization evaluation & trade-offs.
- BOLT: Online vs Batch inference framing.
- CACHE-STACK: Layered caching design & governance.
- VECTOR-FIT: Vector DB selection & tuning.

## Navigation Tips
- Start with Core (01) if you want foundations.
- Jump directly to optimization or infra domains (05–09) as needed.
- Use global search across `docs/` for specific terms (e.g., "speculative decoding", "MMR", "LoRA").

## Suggested Future Enhancements
- Cross-link related sections (e.g., retrieval metrics ↔ groundedness).
- Add a glossary of key terms & acronyms.
- Incorporate lightweight architecture diagrams.
- Provide example code snippets (retrieval pipeline, caching middleware, adapter loading).

---
All detailed material now lives in the modular files above. This index intentionally remains concise.

