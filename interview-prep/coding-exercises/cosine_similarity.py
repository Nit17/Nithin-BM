"""Cosine Similarity Exercise

Problem: Compute cosine similarity between two numeric vectors.
Focus: numeric stability, validation.
Complexity: O(n) time, O(1) space.
Extensions: streaming tolerance, sparse vectors.
"""
from __future__ import annotations
from typing import Sequence
import math

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

if __name__ == "__main__":
    print(cosine_similarity([1,2,3],[1,2,3]))
