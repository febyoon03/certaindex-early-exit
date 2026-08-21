"""Step runners. Sequential rule is enforced on recorded gate values.

Call order for a full track:
  1. step0  — smoke generate("Reply with exactly: OK")
  2. step1  — baseline generate(task prompt) + real extractor vs gold
  3. step2  — chunked generate_continue + discarded side-branch probe
  or parsestop — single stream, stop when ANSWER: <int> is terminated
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from early_exit.backends import Backend
from early_exit.extract import answer_complete_int, extract_answer
from early_exit.io_util import read_json
from early_exit.stats import mean, median
from early_exit.tasks import Task
from early_exit.timing import add_result, empty_call_totals
from early_exit.throughput import attach_throughput


def step0(backend: Backend) -> dict[str, Any]:
    prompt = "Reply with exactly: OK"
    text, n_tok, dt = backend.generate(prompt, max_tokens=16)
    tps = n_tok / dt if dt > 0 else 0.0
    said_ok = "ok" in (text or "").strip().lower()
    return {
        "step": 0,
        "backend": backend.name,
        "model_id": backend.model_id,
        "load_s": getattr(backend, "load_s", None),
        "smoke_text": text,
        "smoke_tokens": n_tok,
        "smoke_latency_s": dt,
        "tokens_per_sec": tps,
        "gate": "PASS" if said_ok else "FAIL",
    }


def run_baseline(backend: Backend, task: Task, examples: list[dict]) -> dict[str, Any]:
    rows = []
    for ex in examples:
        user = task.format_prompt(ex)
        chat = backend.format_chat(user)
        sess = backend.start_session(chat)
        gen = sess.continue_decode(task.max_reason_tokens)
        pred = extract_answer(gen.text, task.extract_kind)
        rows.append(
            {
                **ex,
                "pred": pred,
                "correct": pred == ex["gold"],
                "tokens": gen.n_tokens,
                "latency_s": gen.latency_s,
                "prefill_tokens": gen.prefill_tokens,
                "decode_tokens": gen.decode_tokens,
                "prefill_s": gen.prefill_s,
                "decode_s": gen.decode_s,
                "n_generate_calls": gen.n_calls,
                "cache_hit": gen.cache_hit,
                "cache_enabled": sess.cache_enabled,
                "text": gen.text,
            }
        )
    acc = mean(r["correct"] for r in rows)
    total_lat = sum(r["latency_s"] for r in rows)
    out = {
        "step": 1,
        "task": task.name,
        "model_id": backend.model_id,
        "cache_enabled": bool(getattr(backend, "use_cache", False) and rows and rows[0].get("cache_enabled")),
        "n": len(rows),
        "accuracy": acc,
        "exact_match_rate": acc,
        "avg_tokens": mean(r["tokens"] for r in rows),
        "median_tokens": median(r["tokens"] for r in rows),
        "avg_prefill_s": mean(r["prefill_s"] for r in rows),
        "avg_decode_s": mean(r["decode_s"] for r in rows),
        "avg_prefill_tokens": mean(r["prefill_tokens"] for r in rows),
        "avg_decode_tokens": mean(r["decode_tokens"] for r in rows),
        "total_latency_s": total_lat,
        "avg_total_latency_s": mean(r["latency_s"] for r in rows),
        "tokens_per_sec": (sum(r["tokens"] for r in rows) / total_lat) if total_lat else 0.0,
        "gate": "PASS" if acc >= task.min_baseline_acc else "FAIL",
        "min_baseline_acc": task.min_baseline_acc,
        "extract_kind": task.extract_kind,
        "rows": rows,
    }
    return attach_throughput(out)


@dataclass
class ProbeLog:
    probe_idx: int
    reason_tokens: int
    extracted: Any
    probe_latency_s: float
    probe_text: str


def run_probe_and_stop(
    backend: Backend,
    task: Task,
    examples: list[dict],
    k: int,
    stability: int,
) -> dict[str, Any]:
    if k < 1:
        raise ValueError(f"--k must be >= 1, got {k}")
    if stability < 1:
        raise ValueError(f"--stability must be >= 1, got {stability}")

    rows = []
    for ex in examples:
        user = task.format_prompt(ex)
        chat0 = backend.format_chat(user)
        t_all = time.perf_counter()
        sess = backend.start_session(chat0)

        reason = ""
        used = 0
        last = None
        streak = 0
        stopped = False
        stop_pos = None
        probes: list[dict] = []
        call_log: list[dict] = []
        totals = empty_call_totals()

        while used < task.max_reason_tokens:
            chunk = sess.continue_decode(k)
            reason += chunk.text
            used += chunk.n_tokens
            add_result(totals, chunk, "reason")
            call_log.append(chunk.as_log())

            probe = sess.probe(task.probe_suffix, 8)
            add_result(totals, probe, "probe")
            call_log.append(probe.as_log())
            ans = extract_answer(probe.text, task.extract_kind)
            probes.append(
                asdict(
                    ProbeLog(
                        probe_idx=len(probes) + 1,
                        reason_tokens=used,
                        extracted=ans,
                        probe_latency_s=probe.latency_s,
                        probe_text=probe.text,
                    )
                )
            )

            if ans is not None and ans == last:
                streak += 1
            elif ans is not None:
                streak = 1
                last = ans
            else:
                streak = 0
                last = None

            if streak >= stability:
                stopped = True
                stop_pos = used
                break
            if chunk.n_tokens < k:
                break

        wall = time.perf_counter() - t_all
        if stopped:
            final = last
        else:
            final = extract_answer(reason, task.extract_kind)
        correct = final == ex["gold"]
        rows.append(
            {
                **ex,
                "pred": final,
                "correct": correct,
                "false_stable": bool(stopped and not correct),
                "stopped_early": stopped,
                "stop_pos": stop_pos,
                "reason_tokens": used,
                "n_probes": len(probes),
                "reason_latency_s": totals["reason_prefill_s"] + totals["reason_decode_s"],
                "probe_overhead_s": totals["probe_prefill_s"] + totals["probe_decode_s"],
                "total_latency_s": wall,
                "cache_enabled": sess.cache_enabled,
                "reason_text": reason,
                "probes": probes,
                "call_log": call_log,
                **totals,
            }
        )

    out = {
        "step": 2,
        "task": task.name,
        "model_id": backend.model_id,
        "k": k,
        "stability": stability,
        "cache_enabled": bool(rows and rows[0].get("cache_enabled")),
        "n": len(rows),
        "accuracy": mean(r["correct"] for r in rows),
        "false_stability_rate": mean(r["false_stable"] for r in rows),
        "avg_tokens": mean(r["reason_tokens"] for r in rows),
        "avg_reason_tokens": mean(r["reason_tokens"] for r in rows),
        "median_tokens": median(r["reason_tokens"] for r in rows),
        "median_reason_tokens": median(r["reason_tokens"] for r in rows),
        "avg_reason_latency_s": mean(r["reason_latency_s"] for r in rows),
        "avg_probe_overhead_s": mean(r["probe_overhead_s"] for r in rows),
        "avg_reason_prefill_s": mean(r["reason_prefill_s"] for r in rows),
        "avg_reason_decode_s": mean(r["reason_decode_s"] for r in rows),
        "avg_probe_prefill_s": mean(r["probe_prefill_s"] for r in rows),
        "avg_probe_decode_s": mean(r["probe_decode_s"] for r in rows),
        "avg_n_generate_calls": mean(r["n_generate_calls"] for r in rows),
        "avg_total_latency_s": mean(r["total_latency_s"] for r in rows),
        "early_stop_rate": mean(r["stopped_early"] for r in rows),
        "extract_kind": task.extract_kind,
        "rows": rows,
    }
    return attach_throughput(out)


def run_parsestop(backend: Backend, task: Task, examples: list[dict]) -> dict[str, Any]:
    """Stop the same stream when ANSWER: <int> is digit-terminated.

    No extra generate call. Requires task.parse_stop_kind == "int".
    """
    if task.parse_stop_kind != "int":
        raise ValueError(
            f"parsestop is only defined for integer ANSWER tasks, got {task.name}"
        )

    rows = []
    for ex in examples:
        user = task.format_prompt(ex)
        chat0 = backend.format_chat(user)
        t0 = time.perf_counter()
        sess = backend.start_session(chat0)
        acc = ""
        n_tok = 0
        stopped_early = False
        for piece, n_tok, _dt in sess.stream_decode(task.max_reason_tokens):
            acc += piece
            value, terminated = answer_complete_int(acc)
            if value is not None and terminated:
                stopped_early = True
                break
        dt = time.perf_counter() - t0
        value, terminated = answer_complete_int(acc)
        if value is None:
            pred = extract_answer(acc, "int")
        else:
            pred = value
        rows.append(
            {
                **ex,
                "pred": pred,
                "correct": pred == ex["gold"],
                "tokens": n_tok,
                "latency_s": dt,
                "stopped_early": stopped_early,
                "cache_enabled": sess.cache_enabled,
                "text": acc,
            }
        )

    out = {
        "step": 2,
        "mode": "parsestop",
        "task": task.name,
        "model_id": backend.model_id,
        "cache_enabled": bool(rows and rows[0].get("cache_enabled")),
        "n": len(rows),
        "accuracy": mean(r["correct"] for r in rows),
        "avg_tokens": mean(r["tokens"] for r in rows),
        "avg_total_latency_s": mean(r["latency_s"] for r in rows),
        "early_stop_rate": mean(r["stopped_early"] for r in rows),
        "rows": rows,
    }
    return attach_throughput(out)


def check_step1_gate_passed(out_dir: Path) -> dict[str, Any]:
    """Refuse step 2 unless the recorded step-1 gate is PASS.

    File presence is not enough: save() writes even on FAIL.
    """
    for name in (
        "step1_baseline.summary.json",
        "step1_baseline.json",
        "step1_baseline.real_metrics.json",
    ):
        path = out_dir / name
        if path.exists():
            break
    else:
        raise SystemExit(
            f"No step-1 artifact in {out_dir}. Run --step 1 and get GATE=PASS first."
        )
    try:
        recorded = read_json(path)
    except Exception as e:
        raise SystemExit(f"{path} is unreadable: {e}")
    gate = recorded.get("gate")
    acc = recorded.get("exact_match_rate", recorded.get("accuracy", 0.0))
    if gate != "PASS":
        raise SystemExit(
            f"Step 1 is on disk at {path} but gate={gate!r} "
            f"(accuracy={acc!r}). Re-run --step 1 until GATE=PASS."
        )
    return recorded
