# 9. Vector Databases (VECTOR-FIT)

## Roles
RAG semantic retrieval, personalization/memory, semantic deduplication, safety exemplar lookup, evaluation drift detection, semantic caching reuse.

## Data Model
Chunk: id, doc_id, vector(s), text, metadata (source, section, lang, timestamp, access_level, pii_flags, version, tags), embedding_model_version. Multi-vector fields (title/body/code) for weighted fusion.

## Index Types
- Flat (exact) – small scale / benchmarking.
- HNSW – high recall/low latency; tune M, ef_search.
- IVF / IVF-PQ – partition + quantize; tune nlist, nprobe.
- PQ/OPQ – compress memory 4–16×; trade recall.
- DiskANN / Vamana – SSD-backed billion-scale.
- Late interaction (ColBERT) – token-level multi-vector; higher storage.

## Hybrid Retrieval
Sparse (BM25/SPLADE) + dense fusion (weighted/z-score or RRF) → cross-encoder re-rank → diversity (MMR) → compression.

## Metadata Filtering
Pre-filter narrows candidates; ensure balanced recall. Filters: access control, recency, language, type, tags. Integrated filtering (Qdrant/Weaviate) vs wrapper logic (FAISS) vs managed (Pinecone).

## Ingestion & Versioning
Pipeline: clean → chunk → dedupe (MinHash/embedding) → embed (batch) → quality gates (PII/lang/length) → bulk upsert → alias swap. Version: dual-write old+new embeddings; shadow eval recall before cutover.

## Deletions & Compliance
Soft delete + periodic compaction; mapping doc_id→vector_ids for hard delete/right-to-be-forgotten; verify absence via paraphrase probes.

## Scaling & Cost
Shard by doc_id/time; monitor skew (<1.25 max/mean). Replicas for HA & throughput. Memory overhead target ≤2× raw vectors. Compression: FP16, INT8, PQ. Tiered hot (recent) vs cold (archival) indices.

## Evaluation & Observability
Metrics: Recall@k, nDCG@k, MRR, coverage, duplication, p50/p95 latency, ingestion lag, memory per M vectors, cost/query. Nightly recall regression, centroid drift monitoring, duplicate surge alerts.

## Tuning Knobs
HNSW: ef_search until recall plateau. IVF: increase nlist for partition granularity; nprobe for recall. PQ: adjust m & bits/subvector; validate recall drop ≤2–3%. Fusion weights α via validation optimizing nDCG@k or groundedness.

## RAG Integration Best Practices
Dynamic retrieval budget; start k=8–12; expand on low confidence. Diversity (MMR λ≈0.3–0.5), track supported sentence fraction, keep retrieval p95 <120 ms.

## Governance & Change Management
Schema version, embedding model hash, chunking params stored with index manifest. Dual alias rollback; access separation for read vs write. Audit logs of queries (minimized/hashed).

## Vendor Snapshot
FAISS: library control; needs custom durability/filtering.
Qdrant: OSS, HNSW + filtering, quantization, WAL, snapshots.
Pinecone: managed, automatic scaling, hybrid support.
Weaviate: OSS + cloud, GraphQL, hybrid built-in, module ecosystem.

## Selection (VECTOR-FIT)
Volume & velocity, Embedding versioning, Cost & compression, Tail latency & throughput, Operational overhead, Recall & relevance metrics – Filtering complexity, Isolation/security, Tuning flexibility.

## Roadmap Phases
Prototype (Flat/HNSW) → Hybrid + re-rank → Compression & param tuning → Ingestion pipeline + alias swaps → Dual-version indexing + drift dashboards → Cost optimization (dim reduce, tiered) → Advanced (multi-vector, recency boosts, A/B embedding tests).

## Practical Thresholds
Recall@10 ≥0.9 baseline; p95 <50–80 ms; ingestion lag p95 <60 s; memory overhead ≤2×; full re-index 100M <24h.

---
End of Vector Databases.

---
## Quick Reference
- Index Selection: HNSW (versatile), IVF-PQ (memory save), Flat (small set), DiskANN (billion+), ColBERT (late interaction granularity).
- Retrieval Pipeline: Sparse + Dense → Merge → Re-rank → MMR → Compress.
- Migration: Dual-write old+new embeddings → shadow eval → alias swap.
- Key Metrics: Recall@k, nDCG@k, p95 latency, ingestion lag, memory overhead factor.

## Common Pitfalls
| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| Embedding model upgrade without re-index | Recall collapse | Dual index + shadow eval |
| Over-chunking tiny fragments | Poor semantic cohesion | Minimum chunk size & hierarchical approach |
| Ignoring drift | Silent relevance decay | Nightly recall & centroid drift job |
| Excessive nprobe/ef_search chasing 1% | Latency spike | Stop at recall plateau curve |
| No deletion mapping | Compliance failure | Maintain doc_id → vector_ids map |

## Interview Checklist
1. Choose index for 300M docs, moderate updates, low-latency requirement.
2. Design recall regression harness.
3. Hybrid fusion weighting method.
4. Strategy to reduce storage 4× with minimal recall hit.
5. Guarantee fresh content visible <60s after ingest.

## Cross-Links
- Poisoning & safety: see [Evaluation & Safety](04-eval-safety.md#prompt-injection--data-poisoning).
- Caching integration: see [Caching](08-caching.md#layers).

## Further Reading
- HNSW paper, DiskANN
- ColBERT v2
- FAISS wiki & PQ tutorials

## Exercise
- Provide an index parameter tuning plan for scaling from 10M → 100M vectors.

---
