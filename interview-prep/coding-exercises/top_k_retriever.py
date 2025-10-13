"""Top-k Retriever Exercise

Problem: Return top-k docs by score.
Focus: algorithmic complexity tradeoff (sort vs heap).
Complexity: sort O(n log n) or heap O(n log k).
Extensions: threshold filtering, stable ordering.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import heapq

def top_k(scores: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    if k <= 0:
        return []
    n = len(scores)
    if k >= n:
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return heapq.nlargest(k, scores.items(), key=lambda x: (x[1], -ord(x[0][0]) if x[0] else 0))

if __name__ == "__main__":
    print(top_k({"a":1.0,"b":3.0,"c":2.0}, 2))

