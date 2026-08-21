# Validation review

What was checked, how, what broke, what was changed.

Tests do not load a language model. They use `FakeBackend` so the
control flow, extractors, and gates can be executed here.

```
PYTHONPATH=. python3 tests/test_early_exit.py
```

## Issues found

### I1 — Step-2 gate is file presence (original F1)

**How verified:** `save()` writes `step1_baseline.summary.json` before
`main` inspects `gate`. Original `main` then does
`if not base_path.exists()`. A FAIL baseline still unlocks step 2.

**Correction:** `check_step1_gate_passed` reads JSON and requires
`gate == "PASS"`. Test: FAIL artifact blocks; PASS artifact allows;
missing file blocks.

### I2 — `--k 0` infinite loop (original F2)

**How verified:** loop condition `used < MAX` with `n_tok = max_tokens`
and exit `if n_tok < k`. For `k <= 0`, `used` never increases and
`n_tok < k` is false. v2 test hung the original function for 3s.

**Correction:** validate `k >= 1` and `stability >= 1` in the runner
and in the CLI. Test: `ValueError` immediately.

### I3 — HF default model is an MLX repo (original F3)

**How verified:** docstring `python … --backend hf --step 0` with
`DEFAULT_MODEL = "mlx-community/…-4bit"`.

**Correction:** `default_model_for("hf")` is `Qwen/Qwen2.5-0.5B-Instruct`.

### I4 — Step 0 accepts any non-empty string (original F4)

**How verified:** `gate = "PASS" if text.strip() else "FAIL"` on
`"banana banana"`.

**Correction:** `"ok" in text.strip().lower()`.

### I5 — Load failures are raw tracebacks (original F5)

**Correction:** `build_backend` wrapped in CLI with a SystemExit that
names backend, model, and which packages to install.

### I6 — `TypeError` from stream_generate is swallowed (original F6)

**Correction:** print the exception to stderr before the non-stream
fallback.

### I7 — Global `random.seed` is dead (original F7)

**Correction:** only `random.Random(seed)` inside the example factory.

### I8 — Addition scored with TRUE/FALSE

**How verified:** `extract_tf("ANSWER: 37") is None`. `label` is
`"37"` or the comment says dummy. `accuracy` is therefore ~0 even
when the model is correct. The built-in gate cannot pass on the
real task.

**Correction:** tasks `add_direct` / `add_cot` / `add_check` use
`extract_kind="int"` and `gold = a+b`. Baseline test: all
`ANSWER: {gold}` → accuracy 1.0, GATE=PASS.

### I9 — Addition step 2 KeyError on `seq`

**How verified:** `run_early_exit` in `*_add.py`,
`*_add_cot.py`, `*_add_check_baseline.py` calls
`REASON_PROMPT.format(seq=ex["seq"])`. Examples have `a` and `b`.

**Correction:** `task.format_prompt(ex)` uses the fields the task
owns. Test: `run_probe_and_stop` on `add_cot` does not raise.

### I10 — Parse-stop v1 cuts multi-digit answers

**How verified:** `ANSWER_RE = ANSWER:\s*(-?\d+)` is greedy. After
token `"ANSWER: 3"` the regex matches. v1 breaks. If the next token
is `"7"`, the saved answer is 3.

**Correction:** `answer_complete_int` requires `m.end() < len(acc)`
(a non-digit after the number), matching the v2 comment. Test
streams `… ANSWER: 3` + `7` + `\nCheck` and expects pred 37.

### I11 — Unstable last probe used as the answer

**How verified:** `final = last or extract_tf(reason)`. If the last
probe returned something but `streak < stability` and the loop
exited on EOS, that probe still wins.

**Correction:** `final = last` only when `stopped`; otherwise parse
the reason text. Test: one wrong probe + EOS → pred is not the
wrong probe value.

### I12 — Count-only tasks still gold-label TRUE/FALSE

**How verified:** `*_3b.py` prompt asks for `COUNT: <number>` but
`label` is TRUE/FALSE and scoring uses `extract_tf`.

**Correction:** `bitcount_3b` and `bitcount_len8_countonly` set
`gold` to the integer count and `extract_kind="int"`.

### I13 — Seventeen copies of the same 400 lines

**How verified:** diff the attachments; they differ in `OUT_DIR`,
`REASON_PROMPT`, `SEQ_LEN`, and occasionally `MAX_REASON_TOKENS`.

**Correction:** `TASKS` registry + one CLI. Old filenames in
`scripts/` delegate so existing run notes still work.

### I14 — Step-2 filename collides across `--cache` / `--no-cache` runs

**How verified:** `save(task.out_dir, f"step2_k{args.k}_s{args.stability}",
result)` did not encode `args.cache`. A `--no-cache` run with the same
`--k`/`--stability` silently overwrote the `--cache` run's JSON — same
failure mode as I13 (no run id), now hitting step 2 across cache
variants instead of step 1 across script copies. Caught after a
`--cache` run's `step2_k4_s2.json` was overwritten by a later
`--no-cache` run; the cache-on numbers were only recoverable from that
session's printed summary.

**Correction:** filename includes a cache tag:
`f"step2_k{args.k}_s{args.stability}_{'cache' if args.cache else 'nocache'}"`.

## Issues deferred (not bugs in the algorithm statement)

- No KV-cache reuse (`generate_continue` re-encodes the prefix).
  Documented in backends and REVIEW §9. Fixing it is a backend
  change, not a gate/extract change.
- Stream-event token counts vs tokenizer counts. Same as original.
- n=20, seed=42, no confidence intervals.

## Test map

| Check | Issue |
|---|---|
| extract_tf / extract_int / answer_complete_int | I8, I10 |
| make_examples gold rules | I8, I12 |
| add prompt contains a,b | I9 |
| step0 garbage / OK | I4 |
| add baseline PASS/FAIL | I8 |
| probe streak + false_stable | control flow |
| unstable probe ignored | I11 |
| k=0 / stability=0 raise | I2 |
| add_cot probe no KeyError | I9 |
| parsestop 3+7=37 | I10 |
| parsestop refused for tf task | boundary |
| gate FAIL/PASS/missing | I1 |
| hf default not mlx-community | I3 |

## What “new codes for all code” means here

Re-emitting 17 slightly patched copies would pass a surface review
and recreate I13. The replacement for each original file is:

1. A task entry in `early_exit/tasks.py`
2. Shared runners in `early_exit/steps.py`
3. A `scripts/<old name>` wrapper

Behavior that was *intentionally* different (prompts, seq length,
budget 96 vs 256, parse-stop vs probe) is preserved as task or
`--mode` data, not as a new program.
