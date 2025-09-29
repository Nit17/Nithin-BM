"""Stopword Filter Exercise

Problem: Remove stopwords from a query string.
Focus: normalization, set membership.
Complexity: O(n) tokens.
Extensions: preserve original casing, return removed list.
"""
from __future__ import annotations
from typing import Iterable, Set

def stopword_filter(query: str, stopwords: Iterable[str], *, lowercase: bool = True) -> str:
    if not query:
        return ""
    stop: Set[str] = {s.lower() if lowercase else s for s in stopwords}
    tokens = query.split()
    out = []
    for t in tokens:
        key = t.lower() if lowercase else t
        if key not in stop:
            out.append(t if not lowercase else key)
    return " ".join(out)

if __name__ == "__main__":
    print(stopword_filter("This is a simple test", {"is","a"}))
