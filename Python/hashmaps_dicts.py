"""
Hash maps in Python (dict) – practical, runnable examples.

Theory (read this first):
- A dict is a hash map: it hashes the key to find a bucket, so lookups/inserts/deletes are
    O(1) on average. Collisions are handled internally; pathological worst case can be O(n),
    but Python includes randomization and a robust table design to avoid easy collisions.
- Keys must be hashable: their hash value must not change during their lifetime and must be
    consistent with equality. Immutable built-ins like str, int, float, bytes, and tuples of
    immutables are hashable. Lists/dicts/sets are unhashable. Don’t mutate objects used as
    keys in a way that changes their equality/hash.
- Insertion order is preserved (a language guarantee since Python 3.7). That means iterating
    a dict returns keys in the order they were first inserted, which is useful for predictable
    output and simple ordered algorithms.
- Dict views (keys/values/items) are dynamic: they reflect later mutations of the dict.
- Common pitfalls: using a mutable as a key; relying on floating-point NaN (NaN != NaN);
    forgetting that later assignments overwrite earlier values for the same key.

What you'll learn:
- Core dict operations: create, access, update, delete, membership
- Iteration patterns and dict views (keys/values/items)
- Comprehensions and conditional builds
- Merging strategies (| operator, ** unpacking, update)
- defaultdict and Counter for counting/grouping
- setdefault vs get and when to use each
- Sorting dicts (by key/value) and maintaining order
- Custom objects as keys (hash/eq), tuple keys
"""
from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Iterable


# ---------- Basics ----------

def basics() -> None:
    user = {"id": 1, "name": "Alice", "roles": ["admin", "editor"]}
    print("user:", user)
    print("user['name']:", user["name"])  # KeyError if missing
    print("user.get('email'):", user.get("email"))  # None if missing (or provide default)

    user["email"] = "alice@example.com"  # insert
    user["roles"].append("viewer")        # update nested
    print("updated user:", user)

    # membership and removal
    print("has 'id' key?", "id" in user)
    removed = user.pop("email", None)  # remove with default
    print("popped email:", removed)
    del user["roles"]
    print("after del roles:", user)


# ---------- Iteration & views ----------

def iter_and_views() -> None:
    data = {"a": 1, "b": 2, "c": 3}
    print("keys view:", list(data.keys()))
    print("values view:", list(data.values()))
    print("items view:", list(data.items()))

    # Views are dynamic
    kv = data.keys()
    data["d"] = 4
    print("after adding 'd', keys view:", list(kv))

    for k, v in data.items():
        print(f"{k} -> {v}")


# ---------- Comprehensions ----------

def comprehensions() -> None:
    nums = list(range(10))
    squares = {n: n * n for n in nums}
    evens_sq = {n: n * n for n in nums if n % 2 == 0}
    print("squares:", squares)
    print("evens_sq:", evens_sq)


# ---------- Merging ----------

def merging() -> None:
    a = {"x": 1, "y": 2}
    b = {"y": 99, "z": 3}
    merged_pipe = a | b            # Python 3.9+: right-hand wins on conflicts
    merged_unpack = {**a, **b}     # older style: right-most wins
    c = a.copy(); c.update(b)      # in-place update
    print("a|b:", merged_pipe)
    print("{**a, **b}:", merged_unpack)
    print("a updated with b:", c)


# ---------- defaultdict and Counter ----------

def counting_and_grouping() -> None:
    words = "to be or not to be that is the question".split()

    # Counting with Counter
    counts = Counter(words)
    print("Counter counts:", counts)
    print("most common 3:", counts.most_common(3))

    # Grouping with defaultdict(list)
    by_len: defaultdict[int, list[str]] = defaultdict(list)
    for w in words:
        by_len[len(w)].append(w)
    print("grouped by length:", dict(by_len))


# ---------- setdefault vs get ----------

def setdefault_vs_get() -> None:
    data: dict[str, list[int]] = {}

    # Using setdefault: concise for multi-insert
    for i in [1, 2, 3, 1, 2]:
        data.setdefault("nums", []).append(i)
    print("with setdefault:", data)

    # Using get then assign back (more explicit)
    data2: dict[str, list[int]] = {}
    for i in [1, 2, 3, 1, 2]:
        lst = data2.get("nums")
        if lst is None:
            lst = []
            data2["nums"] = lst
        lst.append(i)
    print("with get+assign:", data2)


# ---------- Sorting dicts ----------

def sorting_dicts() -> None:
    prices = {"banana": 3, "apple": 4, "cherry": 2, "date": 5}
    by_key = dict(sorted(prices.items()))
    by_value_asc = dict(sorted(prices.items(), key=lambda kv: kv[1]))
    by_value_desc = dict(sorted(prices.items(), key=lambda kv: kv[1], reverse=True))
    print("by_key:", by_key)
    print("by_value_asc:", by_value_asc)
    print("by_value_desc:", by_value_desc)


# ---------- Custom keys ----------

@dataclass(frozen=True)
class Coord:
    x: int
    y: int
    # dataclass(frozen=True) auto-generates __hash__ and __eq__


def custom_keys() -> None:
    grid: dict[Coord, str] = {
        Coord(0, 0): "origin",
        Coord(1, 2): "A",
        Coord(2, 1): "B",
    }
    print("grid[Coord(1,2)]:", grid[Coord(1, 2)])

    # Tuple keys are also common
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    def add_edge(a: tuple[int, int], b: tuple[int, int]) -> None:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    add_edge((0, 0), (0, 1))
    add_edge((0, 1), (1, 1))
    print("neighbors:", neighbors)


# ---------- Edge cases & tips ----------

def edge_cases() -> None:
    # Unhashable keys raise TypeError
    d = {}
    try:
        d[[1, 2]] = "nope"  # list is unhashable
    except TypeError as e:
        print("TypeError for unhashable key:", e)

    # Keys must be unique; later assignments overwrite earlier ones
    dup = {"k": 1, "k": 2}
    print("duplicate key result:", dup)


# ---------- Main ----------

def main() -> None:
    print("== Basics =="); basics()
    print("\n== Iteration & views =="); iter_and_views()
    print("\n== Comprehensions =="); comprehensions()
    print("\n== Merging =="); merging()
    print("\n== defaultdict & Counter =="); counting_and_grouping()
    print("\n== setdefault vs get =="); setdefault_vs_get()
    print("\n== Sorting dicts =="); sorting_dicts()
    print("\n== Custom keys =="); custom_keys()
    print("\n== Edge cases =="); edge_cases()


if __name__ == "__main__":
    main()

