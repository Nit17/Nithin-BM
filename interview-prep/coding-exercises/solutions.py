"""Standalone aggregate with duplicate definitions for direct execution.

Note: Individual per-problem modules exist; this file maintains backward
compatibility and avoids package-relative imports to simplify running
`python solutions.py` directly.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Iterable, Tuple, Set
import math
import re
import heapq

_WORD_RE = re.compile(r"\w+")

def token_counter(prompt: str, *, lowercase: bool = True, strip_punct: bool = True) -> Dict[str, int]:
    if not prompt or prompt.strip() == "":
        return {}
    text = prompt.lower() if lowercase else prompt
    tokens = _WORD_RE.findall(text) if strip_punct else text.split()
    out: Dict[str,int] = {}
    for t in tokens:
        out[t] = out.get(t,0) + 1
    return out

def chunk_split(text: str, n: int) -> List[str]:
    if n <= 0:
        raise ValueError("n must be > 0")
    words = [w for w in text.split() if w]
    return [" ".join(words[i:i+n]) for i in range(0, len(words), n)]

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    if not a:
        raise ValueError("Vectors must be non-empty")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Zero vector encountered")
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

def top_k(scores: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    if k <= 0:
        return []
    n = len(scores)
    if k >= n:
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return heapq.nlargest(k, scores.items(), key=lambda x: (x[1], -ord(x[0][0]) if x[0] else 0))

def stopword_filter(query: str, stopwords: Iterable[str], *, lowercase: bool = True) -> str:
    if not query:
        return "" 
    stop: Set[str] = {s.lower() if lowercase else s for s in stopwords}
    tokens = query.split()
    out: List[str] = []
    for t in tokens:
        key = t.lower() if lowercase else t
        if key not in stop:
            out.append(t if not lowercase else key)
    return " ".join(out)

def safe_prompt_validator(prompt: str, forbidden: Iterable[str], *, whole_word: bool = True, ignore_case: bool = True) -> Tuple[bool, List[str]]:
    if not prompt:
        return True, []
    flags = re.IGNORECASE if ignore_case else 0
    found: Set[str] = set()
    if whole_word:
        escaped = [re.escape(w) for w in forbidden]
        if not escaped:
            return True, []
        pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", flags)
        for m in pattern.finditer(prompt):
            term = m.group(1)
            found.add(term.lower() if ignore_case else term)
    else:
        text = prompt.lower() if ignore_case else prompt
        for w in forbidden:
            term = w.lower() if ignore_case else w
            if term and term in text:
                found.add(term)
    return (len(found) == 0, sorted(found))

if __name__ == "__main__":
    print(token_counter("Hello hello world!"))
    print(chunk_split("one two three four five", 2))
    print(cosine_similarity([1,2,3],[1,2,3]))
    print(top_k({"a":1.0,"b":2.0,"c":0.5}, 2))
    print(stopword_filter("This is a simple test", {"is","a"}))
    print(safe_prompt_validator("please drop database", ["drop", "truncate"]))
