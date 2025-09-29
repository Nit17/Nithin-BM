# Python Advanced Concepts

## 1. Iterables, Iterators, Generators
| Concept | Definition | Key Methods | Pitfall |
|---------|------------|-------------|---------|
| Iterable | Object returning iterator | __iter__ | Re-iterating may create fresh iterators |
| Iterator | State machine producing values | __next__, __iter__ | Exhaustion not auto-reset |
| Generator | Function with yield producing iterator | send, throw, close | Hidden StopIteration propagation |

### Custom Iterator Example
```python
class Countdown:
    def __init__(self, start):
        self.current = start
    def __iter__(self):
        return self
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1
```

### Generator Delegation
```python
def sub():
    yield from range(3)
```

## 2. Comprehensions & Generator Expressions
- List vs Generator memory tradeoff.
```python
squares_list = [i*i for i in range(10_000)]
squares_gen = (i*i for i in range(10_000))
```
- Prefer generator in streaming pipelines.

## 3. Functional Tools
| Tool | Use Case |
|------|----------|
| map/filter | Transform / predicate filter |
| functools.lru_cache | Memoization (pure functions) |
| functools.partial | Specialize functions |
| itertools.islice | Windowing / pagination |
| itertools.groupby | Group consecutive keys |

## 4. Decorators
### Basic Decorator
```python
import time, functools

def timing(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        t0 = time.time()
        r = fn(*a, **kw)
        dt = (time.time() - t0)*1000
        print(f"{fn.__name__} took {dt:.2f} ms")
        return r
    return wrapper
```

### Parameterized Decorator
```python
def retry(times=3):
    def outer(fn):
        @functools.wraps(fn)
        def inner(*a, **kw):
            last = None
            for _ in range(times):
                try:
                    return fn(*a, **kw)
                except Exception as e:
                    last = e
            raise last
        return inner
    return outer
```

## 5. Context Managers
```python
from contextlib import contextmanager
@contextmanager
def managed_resource():
    acquire()
    try:
        yield
    finally:
        release()
```

## 6. Performance Micro-Patterns
| Pattern | Tip |
|---------|-----|
| Membership | Use set for frequent contains |
| String concat | Use ''.join(list) |
| Large loops | Move lookups local (attr → local var) |
| Data structure | Choose deque for appendleft |

## 7. Async & Concurrency Snapshot
| Mechanism | Use Case | Limits |
|----------|----------|-------|
| threading | I/O bound | GIL for CPU tasks |
| multiprocessing | CPU bound | Overhead serialization |
| asyncio | Many concurrent I/O | Requires async libs |

## 8. Type Hints & Static Analysis
```python
from typing import Iterable, Iterator, TypeVar
T = TypeVar('T')

def flatten(list_of_lists: Iterable[Iterable[T]]) -> Iterator[T]:
    for seq in list_of_lists:
        for item in seq:
            yield item
```

## 9. Common Pitfalls
| Pitfall | Example | Fix |
|---------|---------|-----|
| Mutable default arg | def f(x, acc=[]) | Use None sentinel |
| Late binding in loops | lambdas in for | default arg capture |
| Shadowing builtins | list = [] | Rename variable |
| Floating point surprises | 0.1+0.2!=0.3 | Use Decimal for money |

## 10. Practice Questions
1. Implement an LRU cache manually.
2. Write a context manager that times a block.
3. Convert nested loops to a generator pipeline.
4. Create a decorator that enforces type hints at runtime.
5. Implement bounded parallel fetch with asyncio + semaphore.

## 11. Checklist
- [ ] Comfortable writing custom iterators
- [ ] Understand generator exhaustion
- [ ] Can design parameterized decorators
- [ ] Apply lru_cache appropriately
- [ ] Avoid mutable default args

## 12. Interview Soundbite
"I leverage Python’s iterator/generator protocol and functional constructs to build memory-efficient, composable data pipelines; decorators and context managers encapsulate cross-cutting concerns like retries, timing, and resource safety."

## 13. Suggested Deep Reading
| Area | Reference | Why |
|------|----------|-----|
| Iteration Protocol | Python Docs (Data Model) | Internal mechanics (__iter__/__next__) |
| Generators & Coroutines | PEP 342 / 380 | Understanding send/throw/yield from |
| Asyncio Event Loop | Python Docs / Trio blog | Concurrency model limits |
| Typing & Gradual Types | PEP 484 / mypy docs | Static analysis leverage |
| Performance | PyPy vs CPython blog | Interpreter tradeoffs |

## 14. Extended Practice Prompts
1. Build a generic retry/backoff decorator supporting async + sync functions.
2. Profile a slow ETL script; refactor using generators + itertools.
3. Implement a context manager that captures stdout and restores it.
4. Create an async rate limiter (tokens per second) using asyncio primitives.
5. Write a caching layer with time-based invalidation using functools.lru_cache + wrapper.

## 15. Memory & Object Model Snapshot
- Small integer caching; string interning implications.
- Reference counting + cyclic GC generation thresholds.
- sys.getsizeof vs true container footprint.

## 16. Concurrency Pitfalls
| Pitfall | Explanation | Mitigation |
|---------|------------|-----------|
| Shared mutable state | Race conditions under threads | Use locks / immutability |
| Blocking sync I/O in async | Event loop stalls | Use aiohttp / run_in_executor |
| Oversubscription CPU | Excess processes thrash | Bound process pool size |

## 17. Quick Links
| Topic | Link (doc keyword) |
|-------|--------------------|
| Data Model | "Python data model" |
| Contextlib | "contextlib docs" |
| Asyncio | "asyncio event loop" |
| Typing | "mypy cheat sheet" |
| Profiling | "cProfile tutorial" |