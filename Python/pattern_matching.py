"""
Pattern Matching (match/case) in Python 3.10+

Theory (read this first):
- Structural pattern matching lets you match the shape and content of data (not just values),
  similar to switch/case in other languages but more powerful.
- Patterns include literals, OR-patterns (|), sequence and mapping patterns, class patterns,
  and capture names. You can add guards (if ...) for extra conditions.
- Use it to write clearer branching logic for nested data, ASTs, protocol messages, etc.

This tutorial demonstrates:
- Literal and OR patterns
- Capture and wildcard (_)
- Sequence and mapping patterns
- Class patterns (with dataclasses)
- Guards (case pattern if cond)
- Enums and exhaustiveness hints
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


# ---------- Examples ----------

def classify_http(status: int) -> str:
    match status:
        case 200 | 201 | 202:
            return "success"
        case 400 | 404:
            return "client_error"
        case 500 | 502 | 503:
            return "server_error"
        case _:
            return "unknown"


def describe_point(pt: tuple[int, int]) -> str:
    match pt:
        case (0, 0):
            return "origin"
        case (0, y):
            return f"y-axis at y={y}"
        case (x, 0):
            return f"x-axis at x={x}"
        case (x, y) if x == y:
            return f"diagonal x=y={x}"
        case (x, y):
            return f"point x={x}, y={y}"


def normalize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    # Mapping patterns match by keys
    match cfg:
        case {"host": host, "port": int(port)}:
            return {"host": host, "port": port}
        case {"url": url}:
            # derive host/port from url (toy example)
            host = url.split("://")[-1].split(":")[0]
            return {"host": host, "port": 80}
        case _:
            return {"host": "localhost", "port": 8000}


@dataclass
class Event:
    kind: str
    payload: Any


def handle_event(ev: Event) -> str:
    match ev:
        case Event(kind="login", payload={"user": user}):
            return f"login for {user}"
        case Event(kind="click", payload=(x, y)) if x >= 0 and y >= 0:
            return f"click at ({x},{y})"
        case Event(kind="error", payload=msg):
            return f"error: {msg}"
        case Event():
            return "unhandled event"


class Shape(Enum):
    CIRCLE = auto()
    RECT = auto()


def area(shape: Shape, dims: tuple[float, ...]) -> float | None:
    match shape, dims:
        case (Shape.CIRCLE, (r,)) if r >= 0:
            return 3.14159 * r * r
        case (Shape.RECT, (w, h)) if w >= 0 and h >= 0:
            return w * h
        case _:
            return None


# ---------- Demo ----------

def main() -> None:
    print("Pattern Matching demo:\n")

    print("1) classify_http ->", [classify_http(s) for s in (200, 201, 404, 500, 999)])

    print("\n2) describe_point ->")
    for pt in [(0, 0), (0, 5), (6, 0), (3, 3), (2, 5)]:
        print(f"   {pt}: {describe_point(pt)}")

    print("\n3) normalize_config ->")
    print("   ", normalize_config({"host": "example.com", "port": 8080}))
    print("   ", normalize_config({"url": "http://site"}))
    print("   ", normalize_config({}))

    print("\n4) class patterns (Event) ->")
    print("   ", handle_event(Event("login", {"user": "alice"})))
    print("   ", handle_event(Event("click", (10, 20))))
    print("   ", handle_event(Event("error", "oops")))
    print("   ", handle_event(Event("other", {})))

    print("\n5) enums ->")
    print("   circle area r=2 ->", area(Shape.CIRCLE, (2,)))
    print("   rect area 3x4 ->", area(Shape.RECT, (3, 4)))
    print("   invalid ->", area(Shape.RECT, (-1, 2)))


if __name__ == "__main__":
    main()
