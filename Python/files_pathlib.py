"""
Files & Pathlib in Python

Theory (read this first):
- Prefer pathlib.Path over os.path for readability and rich methods (exists, glob, read_text).
- Always use context managers (with open(...)) so files are closed even on errors.
- Text vs binary: open(..., 'w') for text; 'wb' for binary. Specify encoding for text.
- For safe writes, write to a temp file and replace() atomically to avoid partial files.
- Use json/csv modules for structured data instead of manual string handling.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import os
import tempfile
from typing import Iterable


OUT_DIR = Path("./out-files")


def ensure_outdir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


# ---------- Text I/O ----------

def write_text(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ---------- Binary I/O ----------

def write_binary(p: Path, data: bytes) -> None:
    p.write_bytes(data)

def read_binary(p: Path) -> bytes:
    return p.read_bytes()


# ---------- JSON / CSV ----------

def write_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def write_csv(p: Path, rows: Iterable[list[str]]) -> None:
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "age"])  # header
        w.writerows(rows)


def read_csv(p: Path) -> list[list[str]]:
    with p.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        return [row for row in r]


# ---------- Atomic write ----------

def atomic_write(p: Path, data: str, encoding: str = "utf-8") -> None:
    # Write to a temp file in the same directory then replace
    tmp_dir = p.parent
    with tempfile.NamedTemporaryFile("w", delete=False, dir=tmp_dir, encoding=encoding) as tf:
        tf.write(data)
        tmp_name = tf.name
    Path(tmp_name).replace(p)  # atomic on POSIX


# ---------- Path operations ----------

def demo_paths() -> dict[str, list[str]]:
    out = ensure_outdir()
    (out / "a").mkdir(exist_ok=True)
    (out / "b").mkdir(exist_ok=True)

    # Create some files
    for i in range(3):
        write_text(out / "a" / f"file{i}.txt", f"hello {i}\n")
    write_binary(out / "b" / "data.bin", b"\x00\x01\x02")

    # Glob
    txts = sorted(str(p) for p in (out / "a").glob("*.txt"))

    # Rename/replace
    src = out / "a" / "file0.txt"
    dst = out / "a" / "renamed.txt"
    if dst.exists():
        dst.unlink()
    src.rename(dst)

    # Atomic write example
    atomic_write(out / "notes.txt", "line1\nline2\n")

    return {
        "txts": txts,
        "renamed": [str(dst)],
        "notes": [str(out / "notes.txt")],
    }


# ---------- Demo ----------

def main() -> None:
    print("Files & Pathlib demo:\n")

    out = ensure_outdir()
    tpath = out / "message.txt"
    write_text(tpath, "Hello file!\n")
    print("1) read_text ->", read_text(tpath).strip())

    bpath = out / "blob.bin"
    write_binary(bpath, b"\x10\x20\x30")
    print("2) read_binary ->", list(read_binary(bpath)))

    jpath = out / "user.json"
    write_json(jpath, {"name": "Alice", "age": 30})
    print("3) read_json ->", read_json(jpath))

    cpath = out / "people.csv"
    write_csv(cpath, [["Bob", "25"], ["Cara", "28"]])
    print("4) read_csv ->", read_csv(cpath))

    print("\n5) path operations ->", demo_paths())


if __name__ == "__main__":
    main()
