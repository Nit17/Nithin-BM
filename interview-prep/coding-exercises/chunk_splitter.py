"""Chunk Splitter Exercise

Problem: Split text into n-word chunks.
Focus: bounds handling, validation.
Complexity: O(n) words.
Extensions: overlapping windows, stride.
"""
from __future__ import annotations
from typing import List

def chunk_split(text: str, n: int) -> List[str]:
    if n <= 0:
        raise ValueError("n must be > 0")
    words = [w for w in text.split() if w]
    return [" ".join(words[i:i+n]) for i in range(0, len(words), n)]

if __name__ == "__main__":
    print(chunk_split("one two three four five", 2))
