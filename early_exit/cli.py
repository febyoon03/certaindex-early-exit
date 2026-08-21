#!/usr/bin/env python3
"""Unified Track 1 CLI.

Examples:
  python -m early_exit.cli --task bitcount_cot --step 0
  python -m early_exit.cli --task add_cot --step 1 --backend mlx --model mlx-community/Qwen2.5-3B-Instruct-4bit
  python -m early_exit.cli --task add_cot --step 2 --k 8 --stability 2
  python -m early_exit.cli --task add_cot --mode parsestop
"""

from __future__ import annotations

import argparse
import sys

from early_exit.backends import (
    build_backend,
    default_model_for,
    fallback_model_for,
)
from early_exit.io_util import print_gate, save
from early_exit.steps import (
    check_step1_gate_passed,
    run_baseline,
    run_parsestop,
    run_probe_and_stop,
    step0,
)
from early_exit.tasks import TASKS, get_task
from early_exit.throughput import compare_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track 1 early-exit harness")
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--step", type=int, choices=[0, 1, 2], default=None)
    p.add_argument("--mode", choices=["probe", "parsestop"], default="probe")
    p.add_argument("--backend", choices=["mlx", "hf"], default="mlx")
    p.add_argument("--model", default=None)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--stability", type=int, default=2)
    p.add_argument("--n", type=int, default=20)
    p.add_argument(
        "--cache",
        dest="cache",
        action="store_true",
        default=True,
        help="Reuse KV cache across chunks; fork it for probes (default).",
    )
    p.add_argument(
        "--no-cache",
        dest="cache",
        action="store_false",
        help="Cold generate every chunk/probe (reproduces the original timings).",
    )
    p.add_argument(
        "--warmup",
        dest="warmup",
        action="store_true",
        default=True,
        help="One discarded generate after load so compile is not in the mean.",
    )
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    p.add_argument(
        "--compare",
        action="store_true",
        help="Print a throughput table from this task's saved runs (no model load).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task = get_task(args.task)

    if args.compare and args.step is None and args.mode == "probe":
        print(compare_dir(task.out_dir), end="")
        return

    if args.mode == "parsestop":
        args.step = 2
    if args.step is None:
        raise SystemExit("pass --step 0|1|2 or --mode parsestop")

    if args.step == 2 and args.mode == "probe":
        if args.k < 1:
            raise SystemExit(f"--k must be >= 1, got {args.k}")
        if args.stability < 1:
            raise SystemExit(f"--stability must be >= 1, got {args.stability}")

    model_id = args.model or default_model_for(args.backend)
    try:
        backend = build_backend(args.backend, model_id, use_cache=args.cache)
    except Exception as e:
        raise SystemExit(
            f"Failed to load backend={args.backend!r} model={model_id!r}: {e!r}\n"
            "Install mlx+mlx-lm or torch+transformers, and use an id that "
            "matches the backend (mlx-community/* for mlx)."
        )
    using_cache = bool(getattr(backend, "use_cache", False) and getattr(backend, "cache_enabled", False))
    print(
        f"loaded {backend.model_id} via {backend.name} task={task.name} "
        f"cache={'on' if using_cache else 'off'}"
    )
    if args.warmup and args.step != 0:
        backend.warmup()
        print("warmup done")

    if args.step == 0:
        result = step0(backend)
        save(task.out_dir, "step0", result)
        print_gate(result)
        if result["gate"] != "PASS":
            raise SystemExit("Step 0 failed. Fix the environment before Step 1.")
        if result["tokens_per_sec"] < 5:
            print("WARNING: generation is slow; Step 1 may be painful.")
        return

    examples = task.make_examples(args.n, task.seed)

    if args.step == 1:
        result = run_baseline(backend, task, examples)
        save(task.out_dir, "step1_baseline", result)
        print_gate(result)
        if result["gate"] != "PASS":
            print(
                "Baseline accuracy too low. "
                f"Retry with --model {fallback_model_for(args.backend)} "
                "or a different --task. Do not start Step 2."
            )
            raise SystemExit(1)
        print("\n" + compare_dir(task.out_dir), end="")
        return

    check_step1_gate_passed(task.out_dir)

    if args.mode == "parsestop":
        result = run_parsestop(backend, task, examples)
        save(task.out_dir, "step2_parsestop", result)
        print_gate(result)
        print("\n" + compare_dir(task.out_dir), end="")
        return

    result = run_probe_and_stop(backend, task, examples, k=args.k, stability=args.stability)
    cache_tag = "cache" if args.cache else "nocache"
    save(task.out_dir, f"step2_k{args.k}_s{args.stability}_{cache_tag}", result)
    print_gate(result)
    print(
        "\nCompare to baseline: "
        "if avg_probe_overhead_s is close to avg_reason_latency_s, "
        "k is too small. Try --k 16 or 32 next."
    )
    print("\n" + compare_dir(task.out_dir), end="")


if __name__ == "__main__":
    main(sys.argv[1:])
