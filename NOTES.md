# Certaindex-style Stopping on a ~28-token MLX Trace: When Does a Stop Rule Actually Save Wall-Clock Time?

## TL;DR

On this stack, cutting reasoning tokens is not the same as cutting time. A stop rule only turns into real wall-clock savings when two things are both true: the stop check itself is cheap, and there's trailing computation left to cut after the answer is known. If either one fails, the win disappears. A model-probe stop signal (Certaindex/Dynasor-style) failed the first condition here. A free parse-based stop passed both.

This is a small, single-request reproduction on a laptop (Apple Silicon, MLX). It's not a claim about Certaindex/Dynasor's own batched-serving results. See Scope below.

## Setup

- mlx-lm on Apple Silicon
- Qwen2.5 Instruct, 4-bit (0.5B / 1.5B / 3B)
- temperature = 0, seed = 42, n = 20
- Single-request generation, no server-side batching

## Step 1: Finding a Task the Model Can Actually Solve

The original plan used a binary-counting task: given a bit string, is the count of `1`s above a threshold? Across several attempts (larger token budget, few-shot prompting, model sizes 0.5B to 3B, shorter sequences), accuracy never passed a 0.70 gate. The 3B model's errors were systematic undercounts, not formatting failures, so this wasn't a prompting problem. This gets recorded as a limitation instead of forced into a result: **small Qwen does not reliably count bit strings under this protocol**, and early-exit was never run on it.

Two-digit addition (`a, b ∈ [10, 40]`, 3B) turned out solvable and became the baseline instead:

| Prompt | Exact acc | Avg tokens | Latency |
|---|---|---|---|
| One-line `ANSWER:` only | 1.00 | 7 | n/a |
| Short CoT, answer last | 1.00 | 28 | 1.623 s |

The one-line format has nothing left to cut, so the short-CoT version became the gated baseline for the stopping experiments below.

## Approach A: Model-probe Early Exit (Certaindex/Dynasor-style)

Generate in chunks of `k` tokens. Probe the current prefix with a dedicated extraction call (forced to output `TRUE`/`FALSE`, no free-form answer). Stop once the extracted answer is stable across 2 consecutive probes.

| Setting | Acc | Tokens | Gen time | Probe time | Total time | False-stable |
|---|---|---|---|---|---|---|
| CoT baseline | 1.00 | 28.0 | 1.623 s | 0 | 1.623 s | n/a |
| k = 8 | 1.00 | 19.6 | 2.060 s | 1.587 s | 3.648 s | 0 |
| k = 16 | 1.00 | 28.0 | 2.206 s | 1.338 s | 3.545 s | 0 |

k = 8 cuts tokens by about 30% with zero wrong early stops. But total wall-clock time is still more than 2x the baseline in both settings. The probe is itself a full generate call on the growing prefix, and that cost swamps whatever the stop actually saved.

## Approach B: A Free Parse-based Stop (No Extra Model Call)

Same CoT stream, no probing. Only accept `ANSWER: <int>` once a following non-digit (or EOS) confirms the number has actually ended, instead of matching on the first digits seen.

| Rule | Prompt | Acc | Tokens | Latency |
|---|---|---|---|---|
| Naive `\d+` match | Answer last | 0.00 | 26 | 1.596 s |
| Boundary-confirmed | Answer last | 1.00 | 28 | 1.627 s |
| No stop (baseline) | CoT + trailing check line | 1.00* | 35.0 | 1.917 s |
| Boundary-confirmed | CoT + trailing check line | 1.00 | 18 | 1.227 s |

\* A leftover TRUE/FALSE extractor from the earlier counting-task script initially reported 0.0 accuracy on this row. The script was checking the generation against the wrong extraction target: arithmetic needs an `ANSWER: <int>` check, not a `TRUE`/`FALSE` one. Rechecked against the correct pattern: an exact match on 20 out of 20.

Naive matching cuts mid-number (`43` becomes `4`), and it's wrong every single time. The boundary rule fixes accuracy, but it only pays off early if the model keeps generating after the answer. With "answer last," there's nothing left after it to cut, so the boundary-confirmed run on that prompt (28 tokens, 1.627s) just runs to completion, matching the plain CoT baseline above.

The prompt with a trailing check line actually gives the stop rule something to cut. Same prompt, stop off vs. on: 35.0 to 18.0 tokens (**-49%**), 1.917s to 1.227s (**-36%**), with no accuracy loss and no extra model call anywhere in the loop. (The 28-token/answer-last row above is a different prompt, and isn't the baseline for this comparison. It's shown for contrast, not as the pre-stop number for the check-line condition.)

## What This Suggests for Certaindex-style Stopping

A stop signal is only useful if it's cheap relative to the generation it's cutting short. On a ~28-token trace, a side-branch probe that is itself a full generate call doesn't clear that bar. The overhead swamps the savings, even with zero wrong stops. A free/regex-based stop does clear that bar, but only when there's trailing text after the answer for it to cut. The counting-task result is the matching negative case: don't run stopping machinery on a task the model can't solve in the first place. There's no stable signal to detect there.

## Follow-up: Does KV-cache Reuse Fix the Probe Overhead?

The probe-overhead result above raises an obvious question: is the overhead inherent to probing itself, or just to re-prefilling the growing prefix on every probe? A follow-up run tests this, but on a different machine than everything above (x86_64 CPU, HF `Qwen/Qwen2.5-0.5B-Instruct`, not the Apple Silicon MLX 3B setup). So treat this as a directional check, not a replacement for the numbers above.

| Path | Wall / ex | Reason tokens | Model calls / ex | Prompt tokens billed |
|---|---|---|---|---|
| Baseline (1 decode) | 2.08 s | 6.0 | 1 | 139 |
| Probe, cache on | 3.28 s | 6.0 | 4 | ~139 + 2x suffix |
| Probe, cache off (cold) | 5.25 s | 7.0 | 4 | 601 |

n = 3 here, not 20. Running the full n at this CPU speed (~3 tok/s) would have taken too long, so the sample is too small to trust the absolute numbers here, only the shape of the effect. (One runtime bug was hit and fixed along the way: this tokenizer's `eos_token_ids` is an int, and an `in` check against it was crashing. Now handled for both int and iterable cases.)

Cache-on is 1.6x faster than cold probing, by skipping about 70% of the redundant prefix re-forwarding. But it's still only 0.63x the speed of the one-shot baseline. The reason: on a 6-token answer, two 8-token probes cost more than the reasoning they're checking. Caching removes the *wasted prefill*, not the *decode-bound* cost of probing itself. Decode throughput (~8 tok/s here) doesn't change either way. The cache just stops you from paying the expensive prefill path multiple times.

This refines the Approach A result rather than overturning it. Probe overhead is still the bottleneck on short traces, but caching should matter more as prompts get longer or probes get more frequent, and that hasn't been measured yet on the actual 3B/MLX setup used above. That run is still pending.

This does **not** claim that early-exit helps unmodified CoT, GSM8K, or any Hao-lab decoder (Dynasor, Jacobi Forcing) as-is. Those were left out on purpose. This is a reproduction on one small task, one model family, one hardware/software stack (MLX, single-request, no batching). The probe-overhead result in particular may not hold once probing gets amortized across concurrent requests, which is the setting Certaindex/Dynasor's own results were actually measured in.

## Reproduction

Code: `early_exit/` (backends, session/cache handling, extractors, step runners), plus `scripts/` (old per-experiment filenames, now thin wrappers), plus `tests/` (control-flow tests against a fake backend, no model download needed). See `REVIEW.md` and `VALIDATION.md` in the repo for the architecture review and the full list of issues found and fixed while consolidating the original 17 one-off scripts into this harness.

```bash
PYTHONPATH=. python -m early_exit.cli --task add_cot --step 0 \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit
PYTHONPATH=. python -m early_exit.cli --task add_cot --step 1
PYTHONPATH=. python -m early_exit.cli --task add_cot --step 2 --k 8 --stability 2
PYTHONPATH=. python -m early_exit.cli --task add_cot --step 2 --k 8 --stability 2 --no-cache
PYTHONPATH=. python -m early_exit.cli --task add_check --mode parsestop
```
