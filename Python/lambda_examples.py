"""
Lambda functions in Python: quick, anonymous, single-expression functions.

Theory (read this first):
- A lambda creates a tiny, unnamed function defined by a single expression. Use them when
    passing a short callback (key functions, predicates) improves readability.
- Prefer def for anything non-trivial: multiple expressions, statements (if/for/try), or
    when you’ll reuse the function. Named functions are easier to debug and document.
- Lambdas capture variables by reference; to freeze a value at definition time, use a default
    argument (lambda x, captured=val: ...).
"""
from __future__ import annotations
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Tuple

# 1) Basic syntax and quick inline functions
add = lambda a, b: a + b  # simple adder
square = lambda x: x * x

# 2) Sorting with key functions
words = ["Banana", "apple", "cherry", "date"]
# Sort case-insensitively
sorted_ci = sorted(words, key=lambda s: s.casefold())
# Sort by length then case-insensitive for ties
sorted_len_ci = sorted(words, key=lambda s: (len(s), s.casefold()))

# 3) Mapping and filtering data
nums = list(range(10))
squares = list(map(lambda x: x * x, nums))
evens = list(filter(lambda x: x % 2 == 0, nums))

# 4) Reducing (with built-in sum as typical alternative)
from functools import reduce
product = reduce(lambda acc, x: acc * x, nums[1:] or [1], 1)

# 5) Inline adapters for APIs taking callables
# e.g., custom key extractors or simple predicates
people: List[Dict[str, Any]] = [
    {"name": "Alice", "age": 30},
    {"name": "bob", "age": 25},
    {"name": "Charlie", "age": 35},
]
# Sort people by lowercase name
people_by_name = sorted(people, key=lambda p: p["name"].casefold())
# Filter adults
adults = list(filter(lambda p: p["age"] >= 30, people))

# 6) Closures with lambda (capturing outer variables)
multiplier = 3
triple = lambda x, m=multiplier: x * m  # capture via default to freeze value at definition time

# 7) Conditional logic in a single expression
sign = lambda x: 0 if x == 0 else (1 if x > 0 else -1)

# 8) Small dispatch tables (dictionary of lambdas)
ops: Dict[str, Callable[[int, int], int]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}
apply_op = lambda op, a, b: ops[op](a, b)

# 9) With partial to pre-configure
pow2 = partial(pow, exp=2)
# Alternative with lambda
pow3 = lambda x: pow(x, 3)

# 10) With sorted for complex data (tuples, nested)
data: List[Tuple[str, Tuple[int, int]]] = [
    ("taskA", (2, 5)),
    ("taskB", (1, 7)),
    ("taskC", (2, 3)),
]
sorted_data = sorted(data, key=lambda item: (item[1][0], item[1][1], item[0]))

# 11) Key functions that normalize
unique_words = {w.casefold(): w for w in words}  # map normalized to original form

# 12) Lambdas in list/dict comprehensions (use sparingly)
# Usually prefer named def for readability, but small one-liners are ok.
labels = [(x, (lambda n: "even" if n % 2 == 0 else "odd")(x)) for x in range(6)]


if __name__ == "__main__":
    print("Lambda examples demo:\n")

    print("1) add(2, 3) ->", add(2, 3))
    print("   square(5) ->", square(5))

    print("\n2) sorted case-insensitive ->", sorted_ci)
    print("   sorted by (len, ci)     ->", sorted_len_ci)

    print("\n3) squares ->", squares)
    print("   evens   ->", evens)

    print("\n4) product(1..9) ->", product)

    print("\n5) people by name ->", [p["name"] for p in people_by_name])
    print("   adults         ->", [p["name"] for p in adults])

    print("\n6) triple(4) ->", triple(4))

    print("\n7) sign(-5), sign(0), sign(8) ->", [sign(x) for x in (-5, 0, 8)])

    print("\n8) apply_op('*', 6, 7) ->", apply_op("*", 6, 7))

    print("\n9) pow2(5), pow3(2) ->", pow2(5), pow3(2))

    print("\n10) sorted_data ->", sorted_data)

    print("\n11) unique_words keys (normalized) ->", list(unique_words.keys()))

    print("\n12) labels 0..5 ->", labels)

