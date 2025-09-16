"""
Decorators in Python: wrap functions/classes to add behavior without changing their callers.

This file includes:
- Basic function decorator (logging)
- Timing decorator
- Retry decorator (with backoff)
- Parameterized decorator (decorator factory)
- lru_cache from functools
- cached_property for expensive attribute computation
"""
from __future__ import annotations
import functools
import time
from dataclasses import dataclass
from functools import lru_cache, cached_property
from typing import Any, Callable, Iterable, Optional, TypeVar, Tuple

F = TypeVar("F", bound=Callable[..., Any])

# 1) Basic decorator: log calls

def log_calls(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[LOG] {func.__name__} args={args} kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"[LOG] {func.__name__} -> {result}")
        return result

    return wrapper  # type: ignore[return-value]


# 2) Timing decorator

def timeit(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            dt = (time.perf_counter() - t0) * 1000
            print(f"[TIME] {func.__name__}: {dt:.2f} ms")

    return wrapper  # type: ignore[return-value]


# 3) Retry decorator with optional backoff

def retry(times: int = 3, delay: float = 0.1, backoff: float = 2.0, exceptions: Tuple[type[BaseException], ...] = (Exception,)):
    """Retry the wrapped function if it raises, up to `times` attempts.

    Args:
        times: Total attempts (including the first one).
        delay: Initial delay between attempts in seconds.
        backoff: Multiplier for delay after each failure.
        exceptions: Exception types to catch and retry on.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            wait = delay
            while True:
                attempts += 1
                try:
                    return func(*args, **kwargs)
                except exceptions as e:  # type: ignore[misc]
                    if attempts >= times:
                        print(f"[RETRY] {func.__name__} failed after {attempts} attempts: {e}")
                        raise
                    print(f"[RETRY] {func.__name__} attempt {attempts} failed ({e}); retrying in {wait:.2f}s")
                    time.sleep(wait)
                    wait *= backoff

        return wrapper  # type: ignore[return-value]

    return decorator


# 4) Parameterized decorator example: scale return values

def scale_result(factor: float):
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return factor * func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# 5) Built-in decorators: lru_cache and cached_property

@lru_cache(maxsize=128)
def fib(n: int) -> int:
    """Naive Fibonacci with caching to become fast."""
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


@dataclass
class ExpensiveResource:
    seed: int

    @cached_property
    def value(self) -> int:
        print("[cached_property] computing value once...")
        time.sleep(0.1)
        return self.seed * 42


# Demo functions using the decorators

@log_calls
@timeit
def compute_sum(n: int) -> int:
    return sum(range(n))


counter = {"calls": 0}

@retry(times=3, delay=0.05, backoff=2.0)
def flaky() -> str:
    counter["calls"] += 1
    if counter["calls"] < 2:
        raise RuntimeError("intermittent failure")
    return "ok"


@scale_result(10)
def base_value() -> int:
    return 7


if __name__ == "__main__":
    print("Decorators demo:\n")

    print("1) compute_sum(100000) ->", compute_sum(100_000))

    print("\n2) fib(30) with lru_cache ->", fib(30))

    print("\n3) flaky() with retry ->", flaky())

    r = ExpensiveResource(seed=5)
    print("\n4) cached_property value 1st ->", r.value)
    print("   cached_property value 2nd ->", r.value)

    print("\n5) scale_result(10) * base_value() ->", base_value())

