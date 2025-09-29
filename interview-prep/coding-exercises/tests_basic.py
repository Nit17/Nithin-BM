"""Basic tests for coding exercises.
Run manually: python tests_basic.py

Imports directly from solutions.py for simplicity.
"""
from solutions import (
    token_counter,
    chunk_split,
    cosine_similarity,
    top_k,
    stopword_filter,
    safe_prompt_validator,
)


def test_token_counter():
    assert token_counter("") == {}
    assert token_counter("Hello hello") == {"hello": 2}
    assert token_counter("Hello, world! hello") == {"hello": 2, "world":1}


def test_chunk_split():
    assert chunk_split("a b c d", 2) == ["a b", "c d"]
    assert chunk_split("a b c", 2) == ["a b", "c"]


def test_cosine_similarity():
    assert round(cosine_similarity([1,0],[1,0]), 5) == 1.0
    assert round(cosine_similarity([1,0],[0,1]), 5) == 0.0


def test_top_k():
    scores = {"a":1.0,"b":3.0,"c":2.0}
    top2 = top_k(scores, 2)
    assert len(top2) == 2
    # first is highest score key b
    assert top2[0][0] == 'b'


def test_stopword_filter():
    assert stopword_filter("This is a test", {"is"}) == "this a test"
    assert stopword_filter("", {"a"}) == ""


def test_safe_prompt_validator():
    safe, found = safe_prompt_validator("please drop table", ["drop","truncate"])
    assert not safe and found == ["drop"]
    safe2, found2 = safe_prompt_validator("all good", ["drop"]) 
    assert safe2 and found2 == []


def run_all():
    test_token_counter()
    test_chunk_split()
    test_cosine_similarity()
    test_top_k()
    test_stopword_filter()
    test_safe_prompt_validator()
    print("All tests passed.")

if __name__ == "__main__":
    run_all()
