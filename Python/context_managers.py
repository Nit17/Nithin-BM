"""
Context Managers in Python

Theory (read this first):
- Context managers guarantee setup/teardown via the with-statement protocol: __enter__ runs
    before the block, __exit__ runs even if an exception occurs. This ensures resources like
    files, locks, and network connections are released safely.
- Use class-based context managers when you need stateful objects; use @contextmanager when
    a simple try/finally suffices. For dynamic sets of contexts, ExitStack composes multiple
    managers and unwinds them correctly.

This script shows:
- Class-based context managers (__enter__/__exit__)
- @contextmanager function style
- contextlib.suppress to ignore specific errors
- contextlib.ExitStack to manage dynamic groups of contexts
- redirect_stdout to capture prints
- nullcontext for optional contexts
- Temporary environment variable override
"""
from __future__ import annotations
import contextlib
import io
import os
from dataclasses import dataclass 
from typing import Optional

# 1) Class-based context manager
@dataclass
class Resource:
    name: str
    opened: bool = False

    def __enter__(self):
        self.opened = True
        print(f"[enter] open {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        print(f"[exit] close {self.name}")
        self.opened = False
        # return False to propagate exceptions (default); True would swallow them
        return False


# 2) Function style with @contextmanager
@contextlib.contextmanager
def temporary_file(path: str):
    try:
        print(f"[ctx] creating {path}")
        with open(path, "w") as f:
            f.write("hello\n")
        yield path
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)
            print(f"[ctx] removed {path}")


# 3) suppress: ignore specific exceptions

def might_fail(divisor: int) -> float:
    return 10 / divisor


# 4) ExitStack to manage dynamic/conditional contexts

@contextlib.contextmanager
def open_many(paths):
    stack = contextlib.ExitStack()
    files = []
    try:
        for p in paths:
            f = stack.enter_context(open(p, "w"))
            files.append(f)
        yield files
    finally:
        stack.close()


if __name__ == "__main__":
    print("Context Managers demo:\n")

    print("1) Class-based context manager")
    try:
        with Resource("resA") as r:
            print("   inside with, opened:", r.opened)
        print("   after with, opened:", r.opened)
    except Exception:
        pass

    print("\n2) @contextmanager temporary_file")
    with temporary_file("/tmp/cm_example.txt") as p:
        print("   wrote to:", p)
        print("   exists?", os.path.exists(p))
    print("   exists after?", os.path.exists(p))

    print("\n3) suppress division by zero")
    with contextlib.suppress(ZeroDivisionError):
        print("   result:", might_fail(0))  # suppressed
    print("   still running after suppression")

    print("\n4) ExitStack for many files")
    with open_many(["/tmp/f1.txt", "/tmp/f2.txt"]) as files:
        for i, f in enumerate(files, 1):
            f.write(f"file {i}\n")
        print("   wrote:", [f.name for f in files])

    print("\n5) redirect_stdout capture")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("   inside capture")
        print("   more lines")
    captured = buf.getvalue()
    print("   captured ->", captured.strip().splitlines())

    print("\n6) nullcontext for optional context")
    opt: Optional[contextlib.AbstractContextManager] = None
    cm = contextlib.nullcontext() if opt is None else opt
    with cm:
        print("   ran inside nullcontext")

    print("\n7) Temporary env override with @contextmanager")

    @contextlib.contextmanager
    def temp_env(key: str, value: str):
        old = os.environ.get(key)
        os.environ[key] = value
        try:
            yield
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    print("   before: DEMO_ENV=", os.environ.get("DEMO_ENV"))
    with temp_env("DEMO_ENV", "123"):
        print("   inside: DEMO_ENV=", os.environ.get("DEMO_ENV"))
    print("   after:  DEMO_ENV=", os.environ.get("DEMO_ENV"))

