"""
Generators & Iterators in Python

Theory (read this first):
- Iterators provide a uniform way to consume sequences lazily via next(). Generators are a
    concise way to write iterators using the yield keyword.
- Laziness means values are produced on demand — saving memory and enabling infinite or
    very large streams. Compose filters/maps without materializing intermediate lists.
- Use itertools for building pipelines (islice, chain, groupby, etc.). Prefer generator
    functions/expressions over custom iterator classes unless you need fine control.

This script shows:
- How to write generator functions with yield
- Generator expressions vs list comprehensions
- Itertools utilities (islice, count, chain, groupby)
- Lazy pipelines for large/streaming data
- Custom iterator class (for completeness)
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import islice, count, chain, groupby
from typing import Generator, Iterable, Iterator, List, Tuple

# 1) Basic generator function

def countdown(n: int) -> Generator[int, None, None]:
    """Yield n, n-1, ..., 1 lazily."""
    while n > 0:
        yield n
        n -= 1

# 2) Generator expression vs list comprehension
# Both square numbers 0..9; one is lazy (genexp), the other is eager (list comp)
nums = range(10)
SquaresGen = (x * x for x in nums)  # generator expression (lazy)
SquaresList = [x * x for x in nums]  # list comprehension (eager)

# 3) Lazy pipelines for large/streaming data
# Compose filters and maps without materializing intermediate lists
stream = count(1)  # infinite stream: 1, 2, 3, ...
first_ten_even_squares = list(
    islice(
        (x * x for x in stream if x % 2 == 0),  # filter even and square lazily
        10,
    )
)

# 4) Itertools: chain and groupby
letters1 = ["a", "b", "c"]
letters2 = ["c", "b", "a"]
chained = list(chain(letters1, letters2))

words = ["apple", "apricot", "banana", "blueberry", "cherry", "clementine"]
# groupby requires pre-sorted by the same key
words_sorted = sorted(words, key=lambda w: w[0])
by_initial = {k: list(g) for k, g in groupby(words_sorted, key=lambda w: w[0])}

# 5) Custom iterator class (rarely needed, but instructive)
@dataclass
class StepRange:
    start: int
    stop: int
    step: int = 1

    def __iter__(self) -> Iterator[int]:
        cur = self.start
        while (self.step > 0 and cur < self.stop) or (self.step < 0 and cur > self.stop):
            yield cur
            cur += self.step

# 6) Using yield from to delegate to sub-iterables

def flatten(nested: Iterable[Iterable[int]]) -> Generator[int, None, None]:
    for seq in nested:
        yield from seq


if __name__ == "__main__":
    print("Generators & Iterators demo:\n")

    print("1) countdown(5) ->", list(countdown(5)))

    print("\n2) SquaresGen (first 5) ->", list(islice(SquaresGen, 5)))
    print("   SquaresList         ->", SquaresList)

    print("\n3) first_ten_even_squares ->", first_ten_even_squares)

    print("\n4) chained ->", chained)
    print("   grouped by initial ->", by_initial)

    print("\n5) StepRange(0,10,3) ->", list(StepRange(0, 10, 3)))

    print("\n6) flatten ->", list(flatten([[1, 2], (3, 4), range(5, 8)])))
