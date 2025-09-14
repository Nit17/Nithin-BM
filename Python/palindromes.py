
from collections import deque
from typing import Callable, List, Tuple


def _normalize(s: str) -> str:
    """Lowercase and keep only alphanumeric characters (common interview variant)."""
    return "".join(ch for ch in s.casefold() if ch.isalnum())


def is_palindrome_slice(s: str, *, normalize: bool = False) -> bool:
    """Check palindrome using Python slicing.

    Args:
        s: Input string.
        normalize: If True, ignore case and non-alphanumerics.
    """
    s2 = _normalize(s) if normalize else s
    return s2 == s2[::-1]


def is_palindrome_reversed(s: str, *, normalize: bool = False) -> bool:
    """Check palindrome using built-in reversed() and str.join()."""
    s2 = _normalize(s) if normalize else s
    return s2 == "".join(reversed(s2))


def is_palindrome_two_pointers(s: str) -> bool:
    """Two-pointer technique (ignores case and non-alphanumerics without extra string)."""
    i, j = 0, len(s) - 1
    while i < j:
        # Move i to next alnum
        while i < j and not s[i].isalnum():
            i += 1
        # Move j to prev alnum
        while i < j and not s[j].isalnum():
            j -= 1
        if i < j and s[i].casefold() != s[j].casefold():
            return False
        i += 1
        j -= 1
    return True


def is_palindrome_recursive(s: str) -> bool:
    """Recursive palindrome check on a normalized string.

    Note: Recursion depth is O(n); for very long strings, prefer iterative methods.
    """
    s2 = _normalize(s)

    def helper(left: int, right: int) -> bool:
        if left >= right:
            return True
        return s2[left] == s2[right] and helper(left + 1, right - 1)

    return helper(0, len(s2) - 1)


if __name__ == "__main__":
    samples: List[Tuple[str, bool]] = [
        ("racecar", True),
        ("RaceCar", True),
        ("A man, a plan, a canal: Panama!", True),
        ("hello", False),
        ("", True),
        ("0P", False),
        ("Was it a car or a cat I saw?", True),
    ]

    funcs: List[Tuple[str, Callable[..., bool]]] = [
        ("slice(normalize=True)", lambda s: is_palindrome_slice(s, normalize=True)),
        ("reversed(normalize=True)", lambda s: is_palindrome_reversed(s, normalize=True)),
        ("two_pointers", is_palindrome_two_pointers),
        ("recursive", is_palindrome_recursive),
    ]

    print("Palindrome checks:\n")
    for text, expected in samples:
        results = [fn(text) for _, fn in funcs]
        ok = all(r == expected for r in results)
        status = "OK" if ok else "MISMATCH"
        print(f"{status:10} expected={expected!s:<5} text={text!r}")
    
    print("\nMethods used:")
    for name, _ in funcs:
        print(f" - {name}")
