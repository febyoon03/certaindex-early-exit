# Validation Review

What was checked, how, what broke, and what got changed.

Tests don't load a real language model. They use `FakeBackend` instead, so the control flow, extractors, and gates can all run without one.

```
PYTHONPATH=. python3 tests/test_early_exit.py
```


## Issues Found

### I1: Step-2 gate is just file presence (original F1)

**How verified:** `save()` writes `step1_baseline.summary.json` before `main` ever checks the `gate` value. The original `main` only checks `if not base_path.exists()`. So a FAIL baseline still unlocks step 2.

**Correction:** `check_step1_gate_passed` reads the JSON and requires `gate == "PASS"`. Test: a FAIL artifact blocks it, a PASS artifact allows it, a missing file blocks it too.

### I2: `--k 0` causes an infinite loop (original F2)

**How verified:** the loop condition is `used < MAX`, with `n_tok = max_tokens`, and it exits `if n_tok < k`. When `k <= 0`, `used` never increases, and `n_tok < k` is never true. The v2 test hung the original function for 3 seconds before I caught this.

**Correction:** validate `k >= 1` and `stability >= 1`, both in the runner and in the CLI. Test: raises `ValueError` right away.

### I3: HF default model is actually an MLX repo (original F3)

**How verified:** the docstring says `python … --backend hf --step 0`, but `DEFAULT_MODEL = "mlx-community/…-4bit"`.

**Correction:** `default_model_for("hf")` is now `Qwen/Qwen2.5-0.5B-Instruct`.

### I4: Step 0 accepts any non-empty string (original F4)

**How verified:** `gate = "PASS" if text.strip() else "FAIL"` passes even on `"banana banana"`.

**Correction:** now checks `"ok" in text.strip().lower()`.

### I5: Load failures show raw Python tracebacks (original F5)

**Correction:** `build_backend` is now wrapped in the CLI with a clean `SystemExit` that names the backend, the model, and which packages need installing.

### I6: `TypeError` from stream_generate gets silently swallowed (original F6)

**Correction:** the exception now gets printed to stderr before falling back to the non-stream path.

### I7: Global `random.seed` does nothing (original F7)

**Correction:** only `random.Random(seed)` inside the example factory actually matters now, so that's the only place setting it.

### I8: Addition scored with a TRUE/FALSE extractor

**How verified:** `extract_tf("ANSWER: 37") is None`. The `label` field is either `"37"` or a comment admitting it's a dummy value. So `accuracy` comes out near 0 even when the model is fully correct. The gate can never pass on the real task, no matter what.

**Correction:** the `add_direct` / `add_cot` / `add_check` tasks now use `extract_kind="int"` and `gold = a+b`. Baseline test: every `ANSWER: {gold}` response now gives accuracy 1.0, GATE=PASS.

### I9: Addition step 2 crashes with a KeyError on `seq`

**How verified:** `run_early_exit` in `*_add.py`, `*_add_cot.py`, and `*_add_check_baseline.py` calls `REASON_PROMPT.format(seq=ex["seq"])`. But these examples only have `a` and `b`, no `seq` field at all.

**Correction:** `task.format_prompt(ex)` now uses whatever fields that specific task actually owns. Test: `run_probe_and_stop` on `add_cot` no longer raises.

### I10: Parse-stop v1 cuts off multi-digit answers early

**How verified:** `ANSWER_RE = ANSWER:\s*(-?\d+)` is greedy, but it fires too soon. After the token `"ANSWER: 3"`, the regex already matches, so v1 stops right there. If the next token would have been `"7"`, the saved answer is wrongly just `3`.

**Correction:** `answer_complete_int` now requires `m.end() < len(acc)`, meaning a non-digit character has to show up after the number first. This matches what the v2 comment already claimed. Test: streams `… ANSWER: 3` then `7` then `\nCheck`, and expects the prediction to correctly be 37.

### I11: An unstable last probe still gets used as the final answer

**How verified:** `final = last or extract_tf(reason)`. So if the last probe returned some answer, but the streak never reached `stability` and the loop only exited because of EOS, that shaky probe answer still wins by default.

**Correction:** `final = last` only when the loop actually `stopped` cleanly. Otherwise, it parses the reasoning text directly instead. Test: one wrong probe followed by EOS, and the prediction correctly avoids that wrong probe's value.

### I12: Count-only tasks were still scored as TRUE/FALSE

**How verified:** the `*_3b.py` prompt asks the model for `COUNT: <number>`, but the `label` field is TRUE/FALSE, and scoring runs through `extract_tf` anyway. These two things don't match at all.

**Correction:** `bitcount_3b` and `bitcount_len8_countonly` now set `gold` to the actual integer count, with `extract_kind="int"`.

### I13: Seventeen near-identical copies of the same 400 lines

**How verified:** diffing the files shows they only differ in `OUT_DIR`, `REASON_PROMPT`, `SEQ_LEN`, and occasionally `MAX_REASON_TOKENS`. Everything else is duplicated code.

**Correction:** replaced with a `TASKS` registry plus one single CLI. The old filenames now live in `scripts/` and just delegate to the new code, so old run notes still work unchanged.

### I14: Step-2 filenames collide between `--cache` and `--no-cache` runs

**How verified:** `save(task.out_dir, f"step2_k{args.k}_s{args.stability}", result)` never encoded whether `args.cache` was on or off. So a `--no-cache` run with the same `--k`/`--stability` would silently overwrite the `--cache` run's JSON. Same root problem as I13 (no run id), just showing up at step 2 across cache variants instead of at step 1 across script copies. This one was caught the hard way: a `--cache` run's `step2_k4_s2.json` got silently overwritten by a later `--no-cache` run. The cache-on numbers were only recoverable from that earlier session's printed summary, not from disk.

**Correction:** the filename now includes a cache tag: `f"step2_k{args.k}_s{args.stability}_{'cache' if args.cache else 'nocache'}"`.

## Issues Deferred (not bugs in the algorithm itself)

- No KV-cache reuse (`generate_continue` re-encodes the whole prefix every time). This is documented in the backends and in REVIEW section 9. Fixing it is a backend-level change, not a gate or extractor fix.
- Stream-event token counts don't match tokenizer counts exactly. Same as the original behavior.
- n=20, seed=42, no confidence intervals reported.

## Test Map

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

## What "New Code for All Code" Means Here

Just re-emitting 17 slightly patched copies would pass a surface-level review, but it would just recreate I13 all over again. The actual replacement for each original file is:

1. A task entry in `early_exit/tasks.py`
2. Shared runners in `early_exit/steps.py`
3. A `scripts/<old name>` wrapper

Behavior that was *deliberately* different between the old scripts (prompts, sequence length, a budget of 96 vs. 256, parse-stop vs. probe) is kept, but as task data or a `--mode` flag, not as yet another separate program.
