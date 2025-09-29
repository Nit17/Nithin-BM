
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
