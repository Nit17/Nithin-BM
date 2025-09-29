## 0. Glossary of Key Terms

| Term | Definition |
|------|------------|
| Generative Model | Learns data distribution p(x)/p(x,y) enabling sampling of new examples. |
| Discriminative Model | Models conditional p(y|x) or decision boundary for prediction. |
| Transformer | Neural architecture using self-attention + FFN blocks with positional encodings. |
| Token | Subword unit produced by a tokenizer consumed by the model. |
| BPE | Greedy merge-based subword tokenization algorithm. |
| SentencePiece Unigram | Probabilistic subword model optimizing likelihood over candidate vocab. |
| Embedding | Dense vector representing semantic meaning of text/code/etc. |
| RAG | Retrieval-Augmented Generation: retrieve external context to ground LLM outputs. |
| Chunking | Splitting documents into retrievable segments with metadata. |
| Hybrid Retrieval | Combining sparse lexical (BM25) + dense embeddings (and/or late interaction). |
| Re-ranking | Second-stage precise scoring of query–passage pairs (e.g., cross-encoder). |
| MMR | Max Marginal Relevance; balances relevance & diversity in ranking selection. |
| LoRA | Low-Rank Adaptation method adding low-rank weight updates to frozen base. |
| PEFT | Parameter-Efficient Fine-Tuning umbrella (LoRA, Adapters, Prefix Tuning, BitFit). |
| Perplexity | Exponentiated average negative log-likelihood; lower = better LM fit. |
| Hallucination | Unsupported or fabricated model output content. |
| Groundedness | Degree answer claims are supported by provided context sources. |
| KV Cache | Stored attention key/value tensors enabling fast continuation. |
| Micro-Batching | Aggregating small requests briefly to improve GPU utilization. |
| Speculative Decoding | Draft model proposes tokens; target model verifies to accelerate generation. |
| Quantization | Lower precision representation (INT8/4-bit) of weights/activations to save memory. |
| QLoRA | Quantized base (4-bit NF4) with LoRA adapters trained in higher precision. |
| GPTQ | Post-training weight quantization minimizing reconstruction error. |
| AWQ | Activation-aware weight quant focusing on salient channels. |
| PQ | Product Quantization compressing vectors via subspace codebooks. |
| HNSW | Graph-based ANN index enabling logarithmic-like search complexity. |
| IVF | Inverted File index clustering vectors then searching selected lists. |
| nprobe / ef_search | Parameters controlling recall vs latency in IVF / HNSW. |
| Recall@k | Fraction of queries where at least one ground-truth item appears in top-k. |
| nDCG | Normalized Discounted Cumulative Gain; rank quality metric emphasizing order. |
| MRR | Mean Reciprocal Rank; average inverse first relevant result rank. |
| Alias Swap | Changing logical pointer (alias) to a newly built index for zero-downtime migration. |
| Dual-Write | Writing to old + new index versions during migration for validation. |
| Data Poisoning | Malicious data injection to manipulate model behavior or retrieval. |
| Prompt Injection | Crafted input aiming to override instructions or exfiltrate secrets. |
| Jailbreak | Prompt that bypasses safety guardrails to elicit disallowed content. |
| RLHF | Reinforcement Learning from Human Feedback for preference alignment. |
| DPO | Direct Preference Optimization alternative to RLHF using pairwise comparisons. |
| Adapter Merge | Incorporating trained adapter weights into base weights permanently. |
| Semantic Cache | Approximate reuse of previous answers via embedding similarity. |
| Stale-While-Revalidate | Serve cached response then refresh asynchronously. |
| Time Decay | Reducing weight/score of older documents in retrieval ranking. |
| Drift | Distributional change causing performance degradation if unaddressed. |
| Groundedness Score | Metric quantifying proportion of claims supported by retrieved context. |

---
See also: [Core Concepts](01-core-genai.md), [RAG & Agents](03-rag-agents.md), [Vector DBs](09-vector-databases.md).
