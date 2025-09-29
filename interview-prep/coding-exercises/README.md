# Coding Exercises (Core GenAI / Python Basics)

This set strengthens fundamentals often probed in ML / GenAI screens. Each task includes: problem, considerations, reference solution outline, complexity, and extension ideas.

## 1. Token Counter
Count token (word) frequencies in a prompt.

### Considerations
- Normalize case? (usually lower)
- Strip punctuation? configurable
- Handle empty / whitespace-only strings
- Definition of token: split on whitespace vs tokenizer (simplify here)

### Outline
1. Validate input string.
2. Split by whitespace.
3. Optional normalization (lower, strip punctuation using regex or str.translate).
4. Accumulate counts in dict.

### Complexity
- Time: O(n) tokens
- Space: O(k) unique tokens

### Extensions
- Plug in a real tokenizer (tiktoken) for LLM contexts.
- Return top-N.

## 2. Chunk Splitter
Split text into n-word chunks (preserving order).

### Considerations
- Last chunk may be shorter.
- Handle n <= 0 (raise ValueError).
- Multiple spaces / newlines.

### Outline
1. Pre-clean (split). 
2. Iterate in steps of n; join slice back to string.

### Complexity
- Time: O(n) words
- Space: O(n) for list of words + chunks

### Extensions
- Overlapping windows (stride < n).
- Character-length based with word boundary preservation.

## 3. Cosine Similarity (No Libraries)
Compute cosine similarity between two numeric vectors of equal length.

cosine(a,b) = (Σ a_i b_i) / ( sqrt(Σ a_i^2) * sqrt(Σ b_i^2) )

### Considerations
- Vectors must be same length.
- Zero vector → similarity undefined (return 0 or raise). Here: raise ValueError.
- Accept list/tuple.

### Complexity
- Time: O(n)
- Space: O(1)

### Extensions
- Add type enforcement.
- Support streaming (generator consumption).

## 4. Top-k Retriever
Return top-k docs from a dict of scores.

### Considerations
- If k >= len dict → return all sorted.
- Stability: tie-breaking via key.
- Large dict: use heapq.nsmallest / nlargest to reduce memory.

### Complexity
- Sorting approach: O(n log n)
- Heap approach: O(n log k)

### Extensions
- Support min-threshold filter.
- Return (doc_id, score) tuples.

## 5. Stopword Filter
Remove stopwords from a query string.

### Considerations
- Case normalization.
- Punctuation removal before filtering.
- Custom stopword set injection.

### Complexity
- Time: O(n) tokens
- Space: O(n) for filtered tokens

### Extensions
- Keep original casing for non-stopwords.
- Track removed words count.

## 6. Safe Prompt Validator
Check if prompt contains forbidden keywords; return boolean + offending terms.

### Considerations
- Case-insensitive matching.
- Whole word vs substring (prefer whole word to avoid partials like 'class' in 'classification').
- Configurable forbidden list.

### Complexity
- Time: O(n) tokens
- Space: O(m) matches

### Extensions
- Regex patterns (PII detection placeholders).
- Severity levels per keyword.

---

## Reference Usage Example
Implementations live in individual modules:
- `token_counter.py`
- `chunk_splitter.py`
- `cosine_similarity.py`
- `top_k_retriever.py`
- `stopword_filter.py`
- `safe_prompt_validator.py`

Legacy aggregate re-exports remain in `solutions.py` for backward compatibility. Run `tests_basic.py` to validate.

## General Best Practices Applied
- Input validation with descriptive errors.
- Pure functions (no side effects) ease testing.
- Clear docstrings & type hints improve readability.
- Edge cases covered (empty input, zero division, invalid k).

## Further Practice Ideas
- Add Jaccard similarity for token sets.
- Implement BM25 scoring stub.
- Add a sliding window tokenizer preserving punctuation indices.

## Interview Tip
Explain tradeoffs (e.g., heap vs full sort for top-k) and articulate failure modes (zero vector in cosine similarity) proactively.
