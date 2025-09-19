"""
Exceptions & Error Handling in Python

Theory (read this first):
- Exceptions signal exceptional situations; they unwind the call stack until caught. Use
  try/except for expected failure modes; avoid blanket except that hides bugs.
- Prefer specific exception types; use except Exception as a last-resort boundary (e.g.,
  top-level CLI) where you log and exit gracefully.
- EAFP (Easier to Ask Forgiveness than Permission): try the operation and handle the failure,
  instead of pre-checks (LBYL) that can race. Combine with narrow except clauses.
- Use else for code that runs only when no exception was raised; finally for cleanup that
  always runs. Consider contextlib.suppress for targeted ignore.
- For libraries, define custom exception classes for clarity; use exception chaining to provide
  context.
"""
from __future__ import annotations
import contextlib
import json
from dataclasses import dataclass
from typing import Any


# ---------- Custom Exceptions ----------

class AppError(Exception):
    """Base application error."""

class ConfigError(AppError):
    pass

class NetworkError(AppError):
    pass


# ---------- Examples ----------

def parse_config(text: str) -> dict[str, Any]:
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError as e:
        # Add context using exception chaining
        raise ConfigError("Invalid JSON in config") from e

    # Validate required keys using EAFP
    try:
        host = cfg["host"]
        port = int(cfg.get("port", 80))
    except KeyError as e:
        raise ConfigError(f"Missing required key: {e.args[0]}") from e
    except (TypeError, ValueError) as e:
        raise ConfigError("Port must be an integer") from e

    return {"host": host, "port": port}


def fetch_data(url: str) -> str:
    # Simulate network; wrap non-network exceptions into NetworkError
    try:
        if not url.startswith("http"):
            raise ValueError("unsupported scheme")
        # Pretend to fetch
        return f"DATA({url})"
    except ValueError as e:
        raise NetworkError(f"Bad URL: {url}") from e


def read_file_len(path: str) -> int:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return 0  # Treat missing file as empty
    else:
        return len(data)
    finally:
        # Would run even if we returned early (cleanup happens before returning)
        pass


def maybe_int(x: str) -> int | None:
    # Targeted ignore of parse errors
    with contextlib.suppress(ValueError):
        return int(x)
    return None


# ---------- Demo ----------

def main() -> None:
    print("Exceptions & Error Handling demo:\n")

    print("1) parse_config ->")
    ok = '{"host": "example.com", "port": "8080"}'
    bad_json = '{host:"x"}'
    missing = '{"port": 10}'
    for t in (ok, bad_json, missing):
        try:
            print("   ", parse_config(t))
        except ConfigError as e:
            print("   ConfigError:", e)

    print("\n2) fetch_data ->")
    for url in ("http://site", "ftp://site"):
        try:
            print("   ", fetch_data(url))
        except NetworkError as e:
            print("   NetworkError:", e)

    print("\n3) read_file_len (missing is 0) ->", read_file_len("/tmp/definitely-missing.txt"))

    print("\n4) maybe_int ->", [maybe_int(s) for s in ("10", "x", "20")])


if __name__ == "__main__":
    main()
