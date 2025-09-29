"""Safe Prompt Validator Exercise

Problem: Detect forbidden keywords in prompt.
Focus: regex boundaries, case folding.
Complexity: O(n) tokens ~ O(length).
Extensions: severity tags, regex groups, partial matches.
"""
from __future__ import annotations
from typing import Iterable, List, Tuple, Set
import re

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
        for match in pattern.finditer(prompt):
            term = match.group(1)
            found.add(term.lower() if ignore_case else term)
    else:
        text = prompt.lower() if ignore_case else prompt
        for w in forbidden:
            term = w.lower() if ignore_case else w
            if term and term in text:
                found.add(term)
    return (len(found) == 0, sorted(found))

if __name__ == "__main__":
    print(safe_prompt_validator("please drop table", ["drop","truncate"]))
