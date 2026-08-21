"""Throughput metrics derived from a run summary.

Definitions (same for baseline, probe, parse-stop):

  wall              sum of per-example wall seconds
  examples_per_s    n / wall
  reason_tokens_per_s
                    generated answer/reason tokens / wall
  prefill_tokens_per_s
                    billed prefix tokens / prefill seconds (0 if lumped)
  decode_tokens_per_s
                    decode tokens / decode seconds (0 if lumped)
  prefix_tokens_billed
                    tokens sent through a full-prefix forward
                    (cold path recounts the prompt on every call)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _sum(rows: list[dict], *keys: str) -> float:
    total = 0.0
    for row in rows:
        for key in keys:
            if key in row and row[key] is not None:
                total += _f(row[key])
                break
    return total


def attach_throughput(result: dict[str, Any]) -> dict[str, Any]:
    """Add a `throughput` block. Does not remove existing keys."""
    rows = list(result.get("rows") or [])
    n = int(result.get("n") or len(rows) or 0)
    wall = _sum(rows, "total_latency_s", "latency_s")
    if wall <= 0:
        wall = _f(result.get("total_latency_s")) or _f(result.get("avg_total_latency_s")) * max(n, 1)

    reason_tok = _sum(rows, "reason_tokens", "tokens")
    if reason_tok <= 0:
        reason_tok = _f(result.get("avg_reason_tokens") or result.get("avg_tokens")) * max(n, 1)

    prefill_s = _sum(rows, "reason_prefill_s") + _sum(rows, "probe_prefill_s")
    if prefill_s <= 0:
        prefill_s = _sum(rows, "prefill_s")
    decode_s = _sum(rows, "reason_decode_s") + _sum(rows, "probe_decode_s")
    if decode_s <= 0:
        decode_s = _sum(rows, "decode_s")

    prefill_tok = _sum(rows, "reason_prefill_tokens") + _sum(rows, "probe_prefill_tokens")
    if prefill_tok <= 0:
        prefill_tok = _sum(rows, "prefill_tokens")
    decode_tok = _sum(rows, "reason_decode_tokens") + _sum(rows, "probe_decode_tokens")
    if decode_tok <= 0:
        decode_tok = _sum(rows, "decode_tokens")
        if decode_tok <= 0:
            decode_tok = reason_tok

    block = {
        "n": n,
        "wall_s": wall,
        "examples_per_s": (n / wall) if wall else 0.0,
        "reason_tokens": reason_tok,
        "reason_tokens_per_s": (reason_tok / wall) if wall else 0.0,
        "prefill_tokens_billed": prefill_tok,
        "prefill_s": prefill_s,
        "prefill_tokens_per_s": (prefill_tok / prefill_s) if prefill_s else 0.0,
        "decode_tokens": decode_tok,
        "decode_s": decode_s,
        "decode_tokens_per_s": (decode_tok / decode_s) if decode_s else 0.0,
        "avg_wall_s": (wall / n) if n else 0.0,
    }
    result["throughput"] = block
    result["examples_per_s"] = block["examples_per_s"]
    result["reason_tokens_per_s"] = block["reason_tokens_per_s"]
    return result


def _label(result: dict[str, Any], path: Path | None = None) -> str:
    if result.get("mode") == "parsestop":
        name = "parsestop"
    elif result.get("step") == 1:
        name = "baseline"
    elif result.get("k") is not None:
        name = f"probe k={result['k']} s={result.get('stability', '?')}"
    elif path is not None:
        name = path.stem.replace(".summary", "")
    else:
        name = f"step{result.get('step', '?')}"
    cache = result.get("cache_enabled")
    if cache is True:
        name += " cache"
    elif cache is False:
        name += " cold"
    return name


def row_from_result(result: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    if "throughput" not in result:
        attach_throughput(result)
    t = result["throughput"]
    return {
        "run": _label(result, path),
        "cache": {True: "on", False: "off"}.get(result.get("cache_enabled"), "-"),
        "n": t["n"],
        "acc": _f(result.get("accuracy")),
        "tokens": _f(result.get("avg_reason_tokens") or result.get("avg_tokens")),
        "wall_s": t["avg_wall_s"],
        "ex_per_s": t["examples_per_s"],
        "reason_tok_s": t["reason_tokens_per_s"],
        "prefill_billed": t["prefill_tokens_billed"],
        "decode_tok_s": t["decode_tokens_per_s"],
    }


def format_table(rows: list[dict[str, Any]], baseline: dict[str, Any] | None = None) -> str:
    headers = [
        "run",
        "cache",
        "n",
        "acc",
        "tok",
        "wall/ex",
        "ex/s",
        "reason tok/s",
        "prefix billed",
        "vs base",
        "vs cold",
    ]
    base_ex = baseline["ex_per_s"] if baseline and baseline.get("ex_per_s") else None
    cold_ex = None
    for row in rows:
        if row.get("cache") == "off" and "probe" in row.get("run", ""):
            cold_ex = row["ex_per_s"]
            break

    lines = [
        "  ".join(
            [
                f"{headers[0]:<28}",
                f"{headers[1]:>5}",
                f"{headers[2]:>3}",
                f"{headers[3]:>5}",
                f"{headers[4]:>5}",
                f"{headers[5]:>7}",
                f"{headers[6]:>6}",
                f"{headers[7]:>12}",
                f"{headers[8]:>13}",
                f"{headers[9]:>7}",
                f"{headers[10]:>7}",
            ]
        ),
        "  ".join(
            [
                "-" * 28,
                "-" * 5,
                "-" * 3,
                "-" * 5,
                "-" * 5,
                "-" * 7,
                "-" * 6,
                "-" * 12,
                "-" * 13,
                "-" * 7,
                "-" * 7,
            ]
        ),
    ]
    for row in rows:
        vs_base = "-"
        vs_cold = "-"
        if base_ex and row["ex_per_s"]:
            vs_base = f"{row['ex_per_s'] / base_ex:.2f}x"
        if cold_ex and row["ex_per_s"]:
            vs_cold = f"{row['ex_per_s'] / cold_ex:.2f}x"
        lines.append(
            "  ".join(
                [
                    f"{row['run']:<28}",
                    f"{row['cache']:>5}",
                    f"{int(row['n']):>3}",
                    f"{row['acc']:>5.2f}",
                    f"{row['tokens']:>5.1f}",
                    f"{row['wall_s']:>7.2f}",
                    f"{row['ex_per_s']:>6.3f}",
                    f"{row['reason_tok_s']:>12.2f}",
                    f"{row['prefill_billed']:>13.0f}",
                    f"{vs_base:>7}",
                    f"{vs_cold:>7}",
                ]
            )
        )
    return "\n".join(lines)


def load_run_dir(out_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(out_dir.glob("*.summary.json")):
        try:
            import json

            result = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(result, dict):
            continue
        if result.get("step") not in {1, 2} and "accuracy" not in result:
            continue
        rows.append(row_from_result(result, path))
    return rows


def compare_dir(out_dir: Path) -> str:
    rows = load_run_dir(out_dir)
    baseline = next((r for r in rows if r["run"].startswith("baseline")), None)
    body = format_table(rows, baseline)
    return f"Throughput comparison  ({out_dir})\n{body}\n"
