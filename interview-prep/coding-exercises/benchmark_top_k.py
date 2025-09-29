"""Benchmark: heap (nlargest) vs full sort for top-k retrieval.

Usage:
  python benchmark_top_k.py --n 100000 --k 10 --trials 5

Reports average wall time per method.
"""
from __future__ import annotations
import random
import time
import argparse
import heapq


def gen_scores(n: int) -> dict[str, float]:
    return {f"doc_{i}": random.random() for i in range(n)}


def top_k_sort(scores: dict[str, float], k: int):
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]


def top_k_heap(scores: dict[str, float], k: int):
    return heapq.nlargest(k, scores.items(), key=lambda x: x[1])


def bench(fn, scores, k, trials: int) -> float:
    start = time.perf_counter()
    for _ in range(trials):
        fn(scores, k)
    end = time.perf_counter()
    return (end - start) / trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=20000, help='number of docs')
    parser.add_argument('--k', type=int, default=20, help='top k')
    parser.add_argument('--trials', type=int, default=10, help='repeat count')
    parser.add_argument('--seed', type=int, default=42, help='rng seed')
    args = parser.parse_args()

    random.seed(args.seed)
    scores = gen_scores(args.n)

    t_sort = bench(top_k_sort, scores, args.k, args.trials)
    t_heap = bench(top_k_heap, scores, args.k, args.trials)

    ratio = t_sort / t_heap if t_heap else float('inf')

    print(f"n={args.n} k={args.k} trials={args.trials}")
    print(f"sort: {t_sort*1e3:.3f} ms avg")
    print(f"heap: {t_heap*1e3:.3f} ms avg")
    print(f"speedup (sort/heap): {ratio:.2f}x")

    if args.k >= args.n // 2:
        print("Note: When k is large relative to n, full sort can be comparable or faster.")
    else:
        print("Note: For small k, heap-based approach typically wins.")

if __name__ == '__main__':
    main()
