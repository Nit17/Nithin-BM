"""
Asyncio Basics: practical concurrency with async/await

Covers:
- Defining async functions and running them
- Concurrency with asyncio.gather
- Timeouts with asyncio.timeout
- Cancellation handling
- Limiting concurrency with Semaphore
- Offloading CPU/blocking work with asyncio.to_thread
- TaskGroup (Python 3.11+) structured concurrency
"""
from __future__ import annotations
import asyncio
import math
import random
from typing import List


async def pretend_io(task: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{task} done in {delay:.2f}s"


async def demo_gather() -> List[str]:
    tasks = [pretend_io(f"job-{i}", random.uniform(0.05, 0.2)) for i in range(5)]
    return await asyncio.gather(*tasks)


async def demo_timeout() -> str:
    try:
        # asyncio.timeout is new since Python 3.11
        async with asyncio.timeout(0.1):
            return await pretend_io("slow", 0.5)
    except TimeoutError:
        return "timed out"


async def demo_cancellation() -> str:
    async def worker():
        try:
            await asyncio.sleep(0.5)
            return "finished"
        except asyncio.CancelledError:
            return "cancelled"

    task = asyncio.create_task(worker())
    await asyncio.sleep(0.1)
    task.cancel()
    return await task


async def cpu_bound(n: int) -> int:
    # Deliberate CPU-heavy calculation (sum of primes up to n)
    def is_prime(x: int) -> bool:
        if x < 2:
            return False
        r = int(math.sqrt(x))
        for i in range(2, r + 1):
            if x % i == 0:
                return False
        return True

    def _work() -> int:
        return sum(i for i in range(n + 1) if is_prime(i))

    return await asyncio.to_thread(_work)


async def limited_fetch(names: List[str]) -> List[str]:
    # Limit concurrency to at most 3 at a time
    sem = asyncio.Semaphore(3)

    async def do_one(name: str) -> str:
        async with sem:
            return await pretend_io(name, random.uniform(0.05, 0.2))

    return await asyncio.gather(*(do_one(n) for n in names))


async def demo_taskgroup() -> List[str]:
    results: List[str] = []
    # TaskGroup automatically cancels siblings on error
    async with asyncio.TaskGroup() as tg:  # Python 3.11+
        for i in range(3):
            tg.create_task(pretend_io(f"tg-{i}", 0.05 * (i + 1)))
        # Collect via a small helper
        async def collect():
            results.extend(await demo_gather())
        tg.create_task(collect())
    return results


async def main() -> None:
    print("Asyncio Basics demo:\n")

    print("1) gather ->", await demo_gather())

    print("\n2) timeout ->", await demo_timeout())

    print("\n3) cancellation ->", await demo_cancellation())

    print("\n4) to_thread CPU-bound sum(primes<=20000) ->", await cpu_bound(20_000))

    print("\n5) limited concurrency (Semaphore) ->", await limited_fetch(["A", "B", "C", "D", "E"]))

    print("\n6) TaskGroup ->", await demo_taskgroup())


if __name__ == "__main__":
    asyncio.run(main())
