# Architecture Review, Track 1 Early-Exit Scripts

This is not a product with auth, billing, or campaigns. But the review framework still works. Just map those words to what this repo actually has: **gates**, **scoring**, **generation backends**, and **experiment variants**.

## 1. One-line Summary

Seventeen Python files run the same three-step LM experiment. Smoke test, then baseline CoT, then probe-and-stop or parse-stop. Two toy tasks: bit counting, two-digit addition. Almost all of the "architecture" was just copy-paste. One file, copied, with a different prompt string and `OUT_DIR`.

## 2. Responsibility Boundaries

**Who should own what**

| Piece | One-sentence job | Where it actually lived |
|---|---|---|
| Backend | Turn a string + max_tokens into text, token count, latency | Duplicated class pair in every 17k-line script |
| Task | Dataset, prompt, gold label, extractor, output dir | Scattered constants, plus a few broken `format(seq=)` calls |
| Extractor | Parse a model string into a comparable answer | `extract_tf` reused on addition, so gold never matched |
| Gate | Decide whether the next step may run | File existence, not the recorded `gate` value |
| Step runner | Baseline / probe / parse-stop control flow | Copied; addition step 2 still asked for `ex["seq"]` |
| Persistence | Write full JSON + text-stripped summary | `save` / `drop_texts` copied everywhere |

**Why it was this way:** habit. Each new experiment meant copying the last script and tweaking the prompt. That's not a real constraint. It's just a habit.

**Blast radius of a change:** one bug fix in the probe-loop termination had to be repeated in every file. v2 fixed seven issues, but only in one copy. The other fifteen kept the bugs.

**Auth analog:** there's no user auth here. The closest thing is "may step 2 run?" But even that check was smeared everywhere: the docstring said GATE=PASS, `main` checked `Path.exists()` instead, `save` wrote the file before the accuracy check even ran, and addition used a dummy `label` with a TRUE/FALSE parser that didn't fit.

## 3. Actual Execution Flow

### Success path (intended: bit-count, original `main`)

1. `python … --step 0`
2. `build_backend` loads the mlx or hf model.
3. `step0` generates from `"Reply with exactly: OK"`, max 16 tokens.
4. Gate check: original just checks `bool(text.strip())`. v2 / revised checks `"ok" in text.lower()` instead.
5. `save("step0", …)`. Print GATE. Exit 0 if PASS.

6. `python … --step 1`
7. `make_examples(n, 42)` builds 20 binary strings, or 20 add pairs.
8. For each example: format the prompt, call `backend.generate`, run `extract_*`, compare to gold.
9. Accuracy vs. `MIN_BASELINE_ACC` (0.70) becomes the `gate` value.
10. `save("step1_baseline")` runs first. **Then** `SystemExit` fires if it's a FAIL.

11. `python … --step 2 --k 8 --stability 2`
12. Original: continues if the summary file just exists on disk. Revised: actually checks the JSON's `gate=="PASS"`.
13. For each example, loop:
    - format the chat prefix once
    - generate `k` tokens onto `reason`
    - probe: generate 8 tokens on `prefix + PROBE_SUFFIX`, then discard those probe tokens
    - a streak of identical extracted answers, reaching `stability`, stops the loop
    - or `n_tok < k` (EOS) stops it
    - or `used >= MAX_REASON_TOKENS` stops it
14. Final answer: original falls back to `last or extract(reason)`, even if the streak never actually reached stability. Revised only uses `last` when the loop truly `stopped`.
15. Write `step2_k{k}_s{stability}.json`.

### Failure path (addition scripts, original)

1. Step 1 generates `ANSWER: 37`.
2. `extract_tf` looks for TRUE/FALSE, finds neither, so `pred=None`.
3. `label` is the string `"37"`. So `correct` is False on every single row.
4. `gate=FAIL`. But the JSON is already written to disk by then.
5. Step 2 then either:
   - KeyErrors on `ex["seq"]` inside `run_early_exit`, or
   - just gets skipped, since a later dedicated probe script doesn't share extractors with it anyway.

Mapped to code: `step0_2_early_exit_add.py`'s `REASON_PROMPT` (addition), `extract_tf` (TRUE/FALSE), `run_baseline` comparing `pred == ex["label"]`, `run_early_exit`'s `REASON_PROMPT.format(seq=ex["seq"])`.

### Parse-stop success (v2 script)

1. `stream_generate` runs token by token.
2. After each new piece of text, `ANSWER_RE.search(acc)` checks for a match.
3. v1 stops the instant it finds a match. That can cut `37` down to just `3`.
4. v2 / revised wait one more step: `m.end() < len(acc)` must hold, meaning a non-digit character has already arrived.

## 4. Auth / State / Communication

- **Auth:** none. The gate is the only thing acting like "permission" here.
- **State:** per-example dicts, held in memory, then dumped to JSON. No transactions. No shared mutable state, besides the model weights themselves.
- **Communication:** just direct function calls, all inside one process. The backend is the only real I/O boundary (the model's generate call).
- **Trade-off:** this is fine for a local experiment. It breaks down if this ever grows into a multi-task sweep runner with shared caches. In that case, the backend should own a KV cache handle. Right now, no script does that.

There's no circular import today, simply because everything sits in one file. But that's not a win: no modules means no boundary to actually violate in the first place.

## 5. Tight Coupling and Break Points

- Prompt, dataset, extractor, gate, and OUT_DIR are all baked into the same file. So changing how addition gets scored meant writing a whole *new script* (`add_probe.py`), instead of just a new extractor.
- `MlxBackend` and `HfBackend` are copy-pasted. Dropping mlx tomorrow means editing every single file, not one backend module.
- Step 2, in most of the addition copies, still tries to format `{seq}`. Those scripts just can't run at all.
- `generate_continue` re-encodes the entire prefix on every chunk, and every probe. Fixing that needs a real cache API. This is flagged here, not actually fixed, since renaming functions wouldn't help.

If the next ask is "add a new task type," the current answer is still "copy the file again." That's the expiry date on "fine for now."

## 6. Why This Abstraction (and the Replacement)

**Original abstraction:** none. One script equals one experiment.

**v2 abstraction:** same script, seven local bugfixes, a new OUT_DIR. This doesn't earn a new architecture. It just earns the fixes it made.

**Revised abstraction:**

- `tasks.py`: experiment variants, as plain data
- `backends.py`: mlx / hf / a test double
- `extract.py`: tf vs. int, plus parse-stop completeness
- `steps.py`: step0 / baseline / probe / parse-stop / gate
- `cli.py`: one single entry point
- `scripts/`: old script names just delegate here now

**Given up:** opening one 400-line file and seeing the whole experiment at a glance. **Gained:** one place to fix the loop, real scoring for addition, and tests that don't need to load a model at all.

Could this be simpler? Yes. A single file with a `TASKS` dict would've worked too. A full package is justified here only because parse-stop, probe-and-stop, and two separate extractors had already drifted apart.

## 7. Explanation vs. Runtime Mismatches

| Claim | Runtime |
|---|---|
| "Do not run step N+1 until GATE=PASS" | Original step 2 only checks that the summary file exists |
| "label is dummy; real scoring done separately" | The same script still computes `accuracy` and a gate off that dummy label |
| Step 0 "smoke test" | Any non-empty string counts as a pass |
| `--backend hf --step 0` | The default model is actually an mlx 4-bit repo |
| Addition early-exit | `format(seq=ex["seq"])` just raises an error |
| Parse-stop "the instant ANSWER: \<int\> appears" (v1) | Actually stops on the first digit token of a multi-digit answer |
| Token counts | Stream events get counted as tokens; the non-stream fallback instead uses `len(text.split())` |
| v2 docstring "CORRECTED" | True for only that one file. The addition copies still keep every bug |

## 8. Blind Spots

**Edges**

- `--k 0` / `--stability 0`: `n_tok < k` never fires, `used` stays stuck at 0, and it loops forever.
- Empty generation: extract just returns None, and that resets the streak.
- Negative or huge `--n`: no check at all. It just runs.
- Duplicate step 1 runs just overwrite the old JSON. There's no run id to tell them apart.
- The probe suffix meant for TRUE/FALSE tasks gets reused on addition, in the copied scripts.

**Races**

- None across processes. Inside one decode: there's no cache, so chunks are just independent re-decodes of a growing prefix. Since temp=0, the text comes out about the same each time anyway.

**Resources**

- The model stays loaded for the whole process lifetime. That's fine.
- Full traces, including probe text, get written twice: once in full, once in the summary.
- The hf model never gets a `close()` call.

**Dead / leftover**

- `COUNT_THRESHOLD` sits in the addition copies, completely unused there.
- `extract_tf` sits in the addition copies too, and gets used incorrectly.
- `random.seed(SEED)` runs inside `main`, while `make_examples` quietly uses its own separate `Random` instance.
- `isinstance(backend, MlxBackend)` gets used instead of just checking `backend.format_chat`.

**Parse-stop v1 vs. v2:** the v2 comment is right. The v1 one isn't.

## 9. Must Fix Now / Can Defer

**Must fix (already done in the revised code)**

- Score addition with a real integer extractor.
- Format addition prompts with `a` and `b`, properly.
- Actually enforce a recorded `gate=="PASS"` before step 2 can run.
- Reject `k < 1` and `stability < 1` outright.
- Step 0 now requires the word "ok" to actually appear.
- Give each backend its own proper default model id.
- Let the TypeError from `stream_generate` surface, instead of hiding it.
- Parse-stop now waits for a full digit run to end before stopping.
- Stop treating an unstable last probe as if it were the final answer.
- Stop copying the file entirely. Register a task instead.

**Defer (acceptable for a small, local n=20 study)**

- Reusing the KV-cache between a chunk and its probe. This gets risky as n, sequence length, or model size grow, since probe overhead would then start to dominate for reasons that have nothing to do with the algorithm itself.
- Token-count consistency between stream events and the tokenizer. Fine, as long as it's only used for relative comparisons on one backend.
- Run ids, or handling concurrent writers. This is single-user, single-process right now.
- Statistical significance or confidence intervals. n=20 is a smoke study, not a paper's results table.
- Streaming parse-stop for hf's `past_key_values`. The mlx path is the one the comments are actually optimized for.

## 10. Explainability Test

Another engineer can get the *idea* in five minutes: generate a chain of thought, periodically ask "so what's the answer?", and stop once that answer repeats.

They can't get the *repo* in five minutes, though. They'd have to diff all 17 files just to notice that three of them can't even run step 2, and that the addition accuracy number is really just a parser bug in disguise.

Revised layout: `cli.py` calls into `steps.py`, which calls `backends.py` and `extract.py`, with the task itself picked from `tasks.py`. That's the five-minute version of this repo.

## Fast Judgment (Original Tree)

1. Can you state each module's job in one sentence? No. Each file is really "the whole experiment," all at once.
2. Does auth belong here? Not really applicable. But the gate check definitely shouldn't be buried inside a save-then-check-if-file-exists pattern.
3. Tightly coupled? Yes.
4. Can someone else follow it quickly? Only one file at a time, and several of those files are actively misleading.
5. Could you remove mlx tomorrow? Not without touching every single script.
6. Confident-sounding comments, but odd actual behavior? Yes, see section 7 above.

## Refactor Order Followed

1. Removed the dead `random.seed`, unused thresholds, and the wrong extractors.
2. Fixed the naming: `gold` is now clearly the scored field, and `extract_kind` is now explicit.
3. Split everything into separate backend / task / extract / steps modules.
4. Made tasks fully data-driven. The old scripts became simple aliases.
5. Unified everything into one probe loop and one parse-stop loop.
6. Gave the tests the same FakeBackend contract to work against.
7. Wrote this document, plus a separate VALIDATION.md.
