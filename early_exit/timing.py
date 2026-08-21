"""Tokenizer-accurate generate accounting.

A single wall-clock number is not enough to compare probe-and-stop with a
baseline: we need prefill vs decode, cache hit vs miss, and how many
generate calls ran.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GenerateResult:
    text: str
    n_tokens: int
    latency_s: float
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    n_calls: int = 1
    cache_hit: bool = False
    kind: str = "cold"  # cold | prefill | decode | probe
    token_ids: list[int] = field(default_factory=list)

    def as_log(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("text", None)
        d.pop("token_ids", None)
        return d


def empty_call_totals() -> dict[str, float]:
    return {
        "n_generate_calls": 0,
        "reason_prefill_tokens": 0,
        "reason_decode_tokens": 0,
        "reason_prefill_s": 0.0,
        "reason_decode_s": 0.0,
        "probe_prefill_tokens": 0,
        "probe_decode_tokens": 0,
        "probe_prefill_s": 0.0,
        "probe_decode_s": 0.0,
        "n_cache_hits": 0,
        "n_cache_misses": 0,
    }


def add_result(totals: dict[str, float], res: GenerateResult, bucket: str) -> None:
    """bucket is 'reason' or 'probe'."""
    totals["n_generate_calls"] += res.n_calls
    totals[f"{bucket}_prefill_tokens"] += res.prefill_tokens
    totals[f"{bucket}_decode_tokens"] += res.decode_tokens
    totals[f"{bucket}_prefill_s"] += res.prefill_s
    totals[f"{bucket}_decode_s"] += res.decode_s
    if res.cache_hit:
        totals["n_cache_hits"] += 1
    else:
        totals["n_cache_misses"] += 1
