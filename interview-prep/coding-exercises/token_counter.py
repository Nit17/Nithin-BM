"""Token Counter Exercise

Problem: Count token (word) frequencies in a prompt.
Focus: string normalization, basic parsing, dictionary accumulation.
Complexity: O(n) time, O(k) space.
Extensions: plug in true tokenizer, return top-N.
"""
from __future__ import annotations
from typing import Dict
import re

_WORD_RE = re.compile(r"\w+")

def token_counter(prompt: str, *, lowercase: bool = True, strip_punct: bool = True) -> Dict[str, int]:
    if not prompt or prompt.strip() == "":
        return {}
    text = prompt.lower() if lowercase else prompt
    tokens = _WORD_RE.findall(text) if strip_punct else text.split()
    counts: Dict[str,int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return counts

if __name__ == "__main__":  # quick demo
    print(token_counter("Hello hello, world!"))
