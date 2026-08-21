# Track 1 early-exit (revised)

Reproducing Certaindex/Dynasor-style confidence-based early stopping on a small local model (MLX, Apple Silicon) — and finding that "fewer tokens" and "less wall-clock time" are not the same claim. A model-probe stop signal cut tokens ~30% with zero wrong early stops, but *lost* on wall-clock time because the probe itself is a full extra generate call. A free, parse-based stop rule (no extra model call) cut tokens 49% and wall-clock 36% with no accuracy loss — but only on prompts that leave something after the answer to cut. Full write-up: [`NOTES.md`](NOTES.md).

One harness instead of 17 near-copies. Tasks differ in prompt, gold label,
and extractor. Generation, gating, and I/O are shared.

## Run

```bash
cd track1_revised
python -m venv .venv && source .venv/bin/activate
pip install -U mlx mlx-lm          # Apple Silicon
# or: pip install -U transformers torch accelerate

PYTHONPATH=. python -m early_exit.cli --task add_cot --step 0 \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit
PYTHONPATH=. python -m early_exit.cli --task add_cot --step 1
PYTHONPATH=. python -m early_exit.cli --task add_cot --step 2 --k 8 --stability 2
PYTHONPATH=. python -m early_exit.cli --task add_cot --mode parsestop

# KV-cache on (default): chunks continue the same cache; probes fork and discard it.
# --no-cache reproduces the original cold-prefill-every-call timings.
PYTHONPATH=. python -m early_exit.cli --task add_cot --step 2 --k 8 --no-cache --no-warmup

# Table from saved summaries (no model load):
PYTHONPATH=. python -m early_exit.cli --task add_cot --compare
```

Old filenames still work via `scripts/`.

## Tests (no model download)

```bash
PYTHONPATH=. python3 tests/test_early_exit.py
```

## Tasks

| --task | Replaces | Gold |
|---|---|---|
| bitcount_cot | step0_2_early_exit.py | TRUE/FALSE count>10, len 20 |
| bitcount_cot_budget256 | *_budget256.py | same, 256-token cap |
| bitcount_fewshot | *_fewshot.py | few-shot TRUE/FALSE |
| bitcount_1p5b_fewshot | *_1p5b.py | same, separate out dir |
| bitcount_len8 | *_len8.py | TRUE/FALSE count>4, len 8 |
| bitcount_len8_countonly | *_len8_countonly.py | integer COUNT |
| bitcount_3b | *_3b.py | integer COUNT |
| bitcount_3b_fs | *_3b_fs.py | integer COUNT, other shots |
| add_direct | *_add.py | a+b |
| add_cot | *_add_cot.py, *_add_probe.py, parsestop | a+b |
| add_check | *_add_check_* | a+b |

See `REVIEW.md` and `VALIDATION.md` for the architecture review and
the before/after verification. See `NOTES.md` for the Track 1
results write-up (counting-task limitation, LM-probe vs
boundary-confirmed parse-stop, and the KV-cache follow-up).

## Throughput comparison

`reason tok/s` = generated answer tokens / wall. `prefix billed` is how
many prompt tokens were forwarded (cold probe recounts the prefix every
chunk). `vs base` / `vs cold` are example-rate ratios.

CPU run, `Qwen/Qwen2.5-0.5B-Instruct`, `add_cot`, n = 3, k = 4, s = 2
(not the 3B MLX laptop numbers):

| run | cache | tok | wall/ex | ex/s | reason tok/s | prefix billed (n=3) | vs base | vs cold |
|---|---|---|---|---|---|---|---|---|
| baseline | on | 6.0 | 2.08 s | 0.481 | 2.88 | 417 | 1.00× | 2.53× |
| probe k=4 s=2 | on | 6.0 | 3.28 s | 0.305 | 1.83 | one 139-token prefill + suffixes | 0.63× | 1.60× |
| probe k=4 s=2 | off | 7.0 | 5.25 s | 0.190 | 1.33 | 1803 | 0.40× | 1.00× |
| parsestop | on | 6.0 | 2.09 s | 0.478 | 2.87 | — | 0.99× | 2.51× |

`--compare` reprints this from `*.summary.json`. Cache-on probe is not on
disk (same filename as cold); numbers above are from that session’s printout.

Cache deleted repeated prefix work (1803 billed tokens vs 417). It did
not beat one-shot decode: two 8-token probes still cost more than a
6-token answer. Parse-stop matches baseline because the answer is last.
