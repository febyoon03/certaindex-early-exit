# Architecture review — Track 1 early-exit scripts

This is not a product with auth, billing, or campaigns. The review
framework still applies if those words are mapped to what this repo
actually has: **gates**, **scoring**, **generation backends**, and
**experiment variants**.

## 1. One-line summary

Seventeen Python files run the same three-step LM experiment
(smoke → baseline CoT → probe-and-stop or parse-stop) on two toy
tasks (bit counting, two-digit addition). Almost all of the
“architecture” was copy-paste of one file with a different prompt
string and `OUT_DIR`.

## 2. Responsibility boundaries

**Who should own what**

| Piece | One-sentence job | Where it actually lived |
|---|---|---|
| Backend | Turn a string + max_tokens into text, token count, latency | Duplicated class pair in every 17k-line script |
| Task | Dataset, prompt, gold label, extractor, output dir | Scattered constants + a few broken `format(seq=)` calls |
| Extractor | Parse a model string into a comparable answer | `extract_tf` reused on addition, so gold never matched |
| Gate | Decide whether the next step may run | File existence, not the recorded `gate` value |
| Step runner | Baseline / probe / parse-stop control flow | Copied; addition step 2 still asked for `ex["seq"]` |
| Persistence | Write full JSON + text-stripped summary | `save` / `drop_texts` copied everywhere |

**Why it was this way:** habit. Each new experiment was “copy the last
script and tweak the prompt.” That is not a domain constraint.

**Blast radius of a change:** a bug fix in probe-loop termination had
to be repeated in every file. v2 fixed seven issues in one copy and
left the other fifteen alone.

**Auth analog:** there is no user auth. The only access-control is
“may step 2 run?” That check was smeared: docstring said GATE=PASS,
`main` checked `Path.exists()`, `save` wrote the file before the
accuracy check, addition used a dummy `label` and a TRUE/FALSE
parser.

## 3. Actual execution flow

### Success path (intended: bit-count, original `main`)

1. `python … --step 0`
2. `build_backend` loads mlx or hf model.
3. `step0` generates from `"Reply with exactly: OK"`, max 16 tokens.
4. Gate: original = `bool(text.strip())`. v2 / revised = `"ok" in text.lower()`.
5. `save("step0", …)`. Print GATE. Exit 0 if PASS.

6. `python … --step 1`
7. `make_examples(n, 42)` builds 20 binary strings (or 20 add pairs).
8. For each example: `REASON_PROMPT.format(...)` → `backend.generate` → `extract_*` → compare to gold.
9. Accuracy vs `MIN_BASELINE_ACC` (0.70) becomes `gate`.
10. `save("step1_baseline")` **then** `SystemExit` if FAIL.

11. `python … --step 2 --k 8 --stability 2`
12. Original: if summary file exists, continue. Revised: JSON `gate=="PASS"`.
13. For each example:
    - format chat prefix once
    - loop: generate `k` tokens onto `reason`; probe by generating 8 tokens on `prefix + PROBE_SUFFIX`; discard probe tokens
    - streak of identical extracted answers ≥ `stability` → stop
    - or `n_tok < k` (EOS) → stop
    - or `used >= MAX_REASON_TOKENS` → stop
14. Final answer: original `last or extract(reason)` even if streak never reached stability. Revised: only use `last` when `stopped`.
15. Write `step2_k{k}_s{stability}.json`.

### Failure path (addition scripts, original)

1. Step 1 generates `ANSWER: 37`.
2. `extract_tf` finds no TRUE/FALSE → `pred=None`.
3. `label` is `"37"` (string). `correct` is False for every row.
4. `gate=FAIL`, but the JSON is already on disk.
5. Step 2 either:
   - KeyErrors on `ex["seq"]` in `run_early_exit`, or
   - is skipped by a later dedicated probe script that does not share extractors.

Mapped to code: `step0_2_early_exit_add.py` `REASON_PROMPT` (addition),
`extract_tf` (TRUE/FALSE), `run_baseline` compare `pred == ex["label"]`,
`run_early_exit` `REASON_PROMPT.format(seq=ex["seq"])`.

### Parse-stop success (v2 script)

1. `stream_generate` token by token.
2. After each piece, `ANSWER_RE.search(acc)`.
3. v1 stops on first match → can cut `37` to `3`.
4. v2 / revised require `m.end() < len(acc)` so a non-digit has arrived.

## 4. Auth / state / communication

- **Auth:** none. Gate is the only “permission.”
- **State:** per-example dicts in memory, then JSON on disk. No
  transactions, no shared mutable model state besides the loaded
  weights.
- **Communication:** direct function calls inside one process.
  Backend is the only I/O boundary (model generate).
- **Trade-off:** correct for a local experiment. Wrong if this grows
  into a multi-task sweep runner with shared caches — then the
  backend should own a KV cache handle, which no script does.

There is no circular import today because everything is one file.
The cost is the opposite problem: no modules, so no boundary to
violate.

## 5. Tight coupling and break points

- Prompt, dataset, extractor, gate, and OUT_DIR are compiled into
  the same file. Changing addition scoring required a *new script*
  (`add_probe.py`) rather than a new extractor.
- `MlxBackend` / `HfBackend` are copy-pasted. Removing mlx tomorrow
  means editing every file, not one backend module.
- Step 2 in most addition copies still formats `{seq}` — they cannot
  be run at all.
- `generate_continue` re-encodes the full prefix every chunk and
  every probe. Removing that cost requires a cache API; flagged,
  not “fixed” by renaming functions.

If the next requirement is “new task type,” the original answer is
“copy the file again.” That is the expiry date of “fine for now.”

## 6. Why this abstraction (and the replacement)

**Original abstraction:** none. One script = one experiment.

**v2 abstraction:** same script, seven local bugfixes, new OUT_DIR.
Does not earn a new architecture; it does earn the fixes.

**Revised abstraction:**

- `tasks.py` — experiment variants as data
- `backends.py` — mlx / hf / test double
- `extract.py` — tf vs int, parse-stop completeness
- `steps.py` — step0 / baseline / probe / parse-stop / gate
- `cli.py` — one entry
- `scripts/` — old names delegate here

**Given up:** opening one 400-line file and seeing the whole
experiment. **Gained:** one place to fix the loop, real scoring for
addition, tests that do not load a model.

Could it be simpler? Yes: a single file with a `TASKS` dict would
also have worked. A package is justified because parse-stop,
probe-and-stop, and two extractors already diverged.

## 7. Explanation vs runtime mismatches

| Claim | Runtime |
|---|---|
| “Do not run step N+1 until GATE=PASS” | Original step 2 only checks that the summary file exists |
| “label is dummy; real scoring done separately” | The same script still computes `accuracy` and a gate from the dummy |
| Step 0 “smoke test” | Any non-empty string PASSed |
| `--backend hf --step 0` | Default model is an mlx 4-bit repo |
| Addition early-exit | `format(seq=ex["seq"])` raises |
| Parse-stop “the instant ANSWER: \<int\> appears” (v1) | Stops on the first digit token of a multi-digit answer |
| Token counts | Stream events counted as tokens; non-stream fallback uses `len(text.split())` |
| v2 docstring “CORRECTED” | Only that one file; addition copies keep the bugs |

## 8. Blind spots

**Edges**

- `--k 0` / `--stability 0`: `n_tok < k` never fires, `used` stays 0, infinite loop.
- Empty generation: extract returns None; treated as reset streak.
- Negative / huge `--n`: no check; will just run.
- Duplicate step 1 runs overwrite JSON; no run id.
- Probe suffix for TRUE/FALSE used on addition in the copied scripts.

**Races**

- None across processes. Inside one decode: no cache, so chunks are
  independent re-decodes of growing prefixes (temp=0, so
  approximately the same text).

**Resources**

- Model stays loaded for the process lifetime (fine).
- Full traces including probe text written twice (full + summary).
- No close() on hf model.

**Dead / leftover**

- `COUNT_THRESHOLD` in addition copies, unused for addition.
- `extract_tf` in addition copies, used incorrectly.
- `random.seed(SEED)` in `main` while `make_examples` uses its own `Random`.
- `isinstance(backend, MlxBackend)` instead of `backend.format_chat`.

**Parse-stop v1 vs v2:** v2 comment is correct; v1 is not.

## 9. Must fix now / can defer

**Must fix (done in revised code)**

- Score addition with an integer extractor.
- Format addition prompts with `a` and `b`.
- Enforce recorded `gate=="PASS"` before step 2.
- Reject `k < 1` and `stability < 1`.
- Step 0 requires “ok”.
- Backend-specific default model ids.
- Surface TypeError from `stream_generate`.
- Parse-stop waits for a digit run to terminate.
- Do not treat an unstable last probe as the answer.
- Stop copying the file; register a task.

**Defer (acceptable for a local n=20 study)**

- KV-cache reuse between chunk and probe. Dangerous when n, seq
  length, or model size grow; then probe overhead dominates for
  reasons unrelated to the algorithm.
- Token-count consistency (stream events vs tokenizer). Fine for
  relative comparisons on one backend.
- Run ids / concurrent writers. Single user, one process.
- Statistical significance, confidence intervals. n=20 is a
  smoke study, not a paper table.
- hf `past_key_values` streaming parse-stop. mlx path is the
  one the comments optimize for.

## 10. Explainability test

Another engineer can understand the *idea* in five minutes: generate
CoT, periodically ask “so what is the answer?”, stop when it repeats.

They cannot understand the *repo* in five minutes. They have to
diff 17 files to see that three of them cannot run step 2 and that
addition accuracy is a parser bug.

Revised layout: `cli.py` → `steps.py` → `backends.py` / `extract.py`,
task picked from `tasks.py`. That is the five-minute version.

## Fast judgment (original tree)

1. Module responsibility in one sentence? No — each file is “the whole experiment.”
2. Auth belong here? N/A; the gate does not belong in `save()`-then-exist.
3. Tightly coupled? Yes.
4. Can someone else follow it quickly? Only one file at a time, and several lie.
5. Remove mlx tomorrow? Not without touching every script.
6. Smart comments, odd runtime? Yes (see §7).

## Refactor order followed

1. Dead `random.seed`, unused thresholds, wrong extractors — removed.
2. Names: `gold` is the scored field; `extract_kind` is explicit.
3. Split backend / task / extract / steps.
4. Tasks are data; scripts are aliases.
5. One probe loop, one parse-stop loop.
6. Same FakeBackend contract for tests.
7. This document plus VALIDATION.md.
