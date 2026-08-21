from __future__ import annotations

from typing import Any, Iterable


def mean(xs: Iterable[Any]) -> float:
    xs = list(xs)
    return sum(float(x) for x in xs) / len(xs) if xs else 0.0


def median(xs: Iterable[Any]) -> float:
    xs = sorted(float(x) for x in xs)
    if not xs:
        return 0.0
    m = len(xs) // 2
    if len(xs) % 2:
        return xs[m]
    return 0.5 * (xs[m - 1] + xs[m])
