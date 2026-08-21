#!/usr/bin/env python3
"""Validation harness for the revised Track 1 package.

No mlx/torch. FakeBackend only.
Run from track1_revised/:
  PYTHONPATH=. python3 tests/test_early_exit.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from early_exit.backends import Backend, default_model_for
from early_exit.extract import answer_complete_int, extract_answer, extract_int, extract_tf
from early_exit.io_util import drop_texts, save
from early_exit.stats import mean, median
from early_exit.steps import (
    check_step1_gate_passed,
    run_baseline,
    run_parsestop,
    run_probe_and_stop,
    step0,
)
from early_exit.tasks import TASKS, get_task


class FakeBackend(Backend):
    name = "fake"
    model_id = "fake-model"
    load_s = 0.0

    def __init__(self, generate_script=None, continue_script=None, stream_pieces=None):
        self._generate_script = list(generate_script or [])
        self._continue_script = list(continue_script or [])
        self._stream_pieces = list(stream_pieces or [])

    def format_chat(self, user_text: str) -> str:
        return "<chat>" + user_text

    def generate(self, prompt, max_tokens):
        text = self._generate_script.pop(0) if self._generate_script else ""
        return text, max(1, len(text.split())), 0.001

    def generate_continue(self, prefix_prompt, max_tokens):
        if self._continue_script:
            text = self._continue_script.pop(0)
            return text, max_tokens, 0.001
        return self.generate(prefix_prompt, max_tokens)

    def stream_continue(self, prefix_prompt, max_tokens):
        if self._stream_pieces:
            acc_n = 0
            for piece in self._stream_pieces:
                acc_n += 1
                yield piece, acc_n, 0.001
            return
        text, n_tok, dt = self.generate_continue(prefix_prompt, max_tokens)
        yield text, n_tok, dt


results = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    extra = f" — {detail}" if detail and status == "FAIL" else ""
    print(f"[{status}] {name}{extra}")


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

check("extract_tf: FINAL wins", extract_tf("blah true blah FINAL: FALSE") == "FALSE")
check("extract_tf: last bare token", extract_tf("TRUE then FALSE") == "FALSE")
check("extract_tf: none", extract_tf("no answer") is None)
check("extract_tf: empty", extract_tf("") is None)
check("extract_int: ANSWER preferred", extract_int("14 + 23 = 37\nANSWER: 37") == 37)
check("extract_int: COUNT", extract_int("COUNT: 5") == 5)
check("extract_answer kind dispatch", extract_answer("FINAL: TRUE", "tf") == "TRUE")
check(
    "answer_complete_int: mid-digit is incomplete",
    answer_complete_int("ANSWER: 3") == (3, False),
)
check(
    "answer_complete_int: non-digit after number terminates",
    answer_complete_int("ANSWER: 37\n") == (37, True),
)

# ---------------------------------------------------------------------------
# dataset / tasks
# ---------------------------------------------------------------------------

t = get_task("bitcount_cot")
ex_a = t.make_examples(20, 42)
ex_b = t.make_examples(20, 42)
check("make_examples deterministic", ex_a == ex_b)
check(
    "bitcount gold matches count>10",
    all((e["count"] > 10) == (e["gold"] == "TRUE") for e in ex_a),
)
add = get_task("add_cot").make_examples(20, 42)
check(
    "add examples gold is a+b",
    all(e["gold"] == e["a"] + e["b"] for e in add),
)
check(
    "add prompt uses a,b not seq",
    "What is" in get_task("add_cot").format_prompt(add[0])
    and str(add[0]["a"]) in get_task("add_cot").format_prompt(add[0]),
)
countonly = get_task("bitcount_3b").make_examples(5, 42)
check(
    "countonly gold is the integer count",
    all(isinstance(e["gold"], int) and e["gold"] == e["count"] for e in countonly),
)

# ---------------------------------------------------------------------------
# stats / drop_texts
# ---------------------------------------------------------------------------

check("mean basic", mean([1, 2, 3]) == 2.0)
check("mean empty", mean([]) == 0.0)
check("median odd", median([3, 1, 2]) == 2.0)
check("median even", median([1, 2, 3, 4]) == 2.5)
dropped = drop_texts(
    {
        "rows": [
            {
                "id": 0,
                "text": "long",
                "reason_text": "r",
                "probes": [{"extracted": "TRUE"}],
                "n_probes": 1,
            }
        ]
    }
)
check(
    "drop_texts strips traces",
    "text" not in dropped["rows"][0]
    and "reason_text" not in dropped["rows"][0]
    and dropped["rows"][0]["probe_answers"] == ["TRUE"],
)

# ---------------------------------------------------------------------------
# F4 step0
# ---------------------------------------------------------------------------

class Garbage(Backend):
    name = "fake"
    model_id = "fake"

    def generate(self, prompt, max_tokens):
        return "banana banana", 2, 0.01


class Ok(Backend):
    name = "fake"
    model_id = "fake"

    def generate(self, prompt, max_tokens):
        return "OK", 1, 0.01


check("step0 FAIL on garbage", step0(Garbage())["gate"] == "FAIL")
check("step0 PASS on OK", step0(Ok())["gate"] == "PASS")

# ---------------------------------------------------------------------------
# baseline real scoring (addition no longer uses extract_tf)
# ---------------------------------------------------------------------------

examples4 = get_task("add_cot").make_examples(4, 1)
correct = [f"working\nANSWER: {e['gold']}" for e in examples4]
res_pass = run_baseline(FakeBackend(generate_script=list(correct)), get_task("add_cot"), examples4)
check(
    "add baseline all-correct -> acc=1 gate=PASS",
    res_pass["accuracy"] == 1.0 and res_pass["gate"] == "PASS",
)
wrong = [f"ANSWER: {e['gold'] + 1}" for e in examples4]
res_fail = run_baseline(FakeBackend(generate_script=list(wrong)), get_task("add_cot"), examples4)
check(
    "add baseline all-wrong -> acc=0 gate=FAIL",
    res_fail["accuracy"] == 0.0 and res_fail["gate"] == "FAIL",
)

# The original addition scripts would have scored extract_tf("ANSWER: 37") as None
# against label "37" and always failed.
check(
    "original mismatch: extract_tf cannot score addition",
    extract_tf("ANSWER: 37") is None,
)

tf_ex = get_task("bitcount_cot").make_examples(4, 1)
tf_ok = [f"FINAL: {e['gold']}" for e in tf_ex]
res_tf = run_baseline(FakeBackend(generate_script=list(tf_ok)), get_task("bitcount_cot"), tf_ex)
check("bitcount baseline uses tf extract", res_tf["accuracy"] == 1.0)

# ---------------------------------------------------------------------------
# probe-and-stop
# ---------------------------------------------------------------------------

one = get_task("bitcount_cot").make_examples(1, 7)
label = one[0]["gold"]
other = "FALSE" if label == "TRUE" else "TRUE"
fb_stop = FakeBackend(
    continue_script=[
        "chunk1",
        f"FINAL: {other}",
        "chunk2",
        f"FINAL: {label}",
        "chunk3",
        f"FINAL: {other}",
        "chunk4",
        f"FINAL: {label}",
        "chunk5",
        f"FINAL: {label}",
    ]
)
res_early = run_probe_and_stop(fb_stop, get_task("bitcount_cot"), one, k=8, stability=2)
row = res_early["rows"][0]
check(
    "probe stops on streak with correct answer",
    row["stopped_early"] is True and row["pred"] == label and row["correct"] is True,
    f"row={ {k: row[k] for k in ('pred','correct','stopped_early','false_stable')} }",
)
check("false_stable False when correct", row["false_stable"] is False)

fb_fs = FakeBackend(
    continue_script=["chunk1", f"FINAL: {other}", "chunk2", f"FINAL: {other}"]
)
row_fs = run_probe_and_stop(fb_fs, get_task("bitcount_cot"), one, k=8, stability=2)["rows"][0]
check(
    "false_stable True when stable wrong",
    row_fs["stopped_early"] and row_fs["false_stable"] and not row_fs["correct"],
)

# Unstable last probe must not become the answer (original used `last or extract`).
# After one probe of OTHER and then EOS (n_tok < k simulated by max_tokens=k still
# returning k — we force EOS by returning n_tok via k? FakeBackend returns max_tokens.
# To simulate EOS we need n_tok < k. Patch one call.)

class EosAfterOne(FakeBackend):
    def generate_continue(self, prefix_prompt, max_tokens):
        if self._continue_script:
            text = self._continue_script.pop(0)
        else:
            text = ""
        # First reason chunk is short (EOS). Probe still uses continue_script.
        return text, 1 if "FINAL" not in text else max_tokens, 0.001


fb_unstable = EosAfterOne(continue_script=["I am still counting", f"FINAL: {other}"])
row_u = run_probe_and_stop(fb_unstable, get_task("bitcount_cot"), one, k=8, stability=2)["rows"][0]
check(
    "unstable last probe is not used as the decision",
    row_u["stopped_early"] is False and row_u["pred"] != other,
    f"pred={row_u['pred']!r} reason={row_u['reason_text']!r}",
)

try:
    run_probe_and_stop(FakeBackend(), get_task("bitcount_cot"), one, k=0, stability=2)
    k0_raised = False
except ValueError:
    k0_raised = True
check("k=0 raises ValueError", k0_raised)

try:
    run_probe_and_stop(FakeBackend(), get_task("bitcount_cot"), one, k=8, stability=0)
    s0_raised = False
except ValueError:
    s0_raised = True
check("stability=0 raises ValueError", s0_raised)

# add_cot step2 must not KeyError on seq
add_one = get_task("add_cot").make_examples(1, 42)
fb_add = FakeBackend(continue_script=["1+2=", "ANSWER: 99", "more", "ANSWER: 99"])
try:
    res_add = run_probe_and_stop(fb_add, get_task("add_cot"), add_one, k=8, stability=2)
    add_step2_ok = True
    add_pred_int = isinstance(res_add["rows"][0]["pred"], int)
except KeyError as e:
    add_step2_ok = False
    add_pred_int = False
    print("KeyError", e)
check("add_cot probe-and-stop does not KeyError on seq", add_step2_ok)
check("add_cot probe extracts int", add_pred_int)

# ---------------------------------------------------------------------------
# parsestop digit boundary
# ---------------------------------------------------------------------------

pieces = ["14 + 23 = 3", "7", "\nANSWER: 3", "7", "\nCheck"]
fb_ps = FakeBackend(stream_pieces=pieces)
ps = run_parsestop(fb_ps, get_task("add_cot"), add_one)
check("parsestop does not cut mid-number", ps["rows"][0]["pred"] == 37)
check("parsestop stopped after terminator", ps["rows"][0]["stopped_early"] is True)

try:
    run_parsestop(FakeBackend(stream_pieces=["x"]), get_task("bitcount_cot"), one)
    ps_blocked = False
except ValueError:
    ps_blocked = True
check("parsestop refused for tf tasks", ps_blocked)

# ---------------------------------------------------------------------------
# F1 gate value, not file presence
# ---------------------------------------------------------------------------

tmpdir = Path(tempfile.mkdtemp(prefix="gate_"))
try:
    failing = get_task("add_cot").make_examples(4, 1)
    wrong = [f"ANSWER: {e['gold'] + 1}" for e in failing]
    baseline = run_baseline(FakeBackend(generate_script=list(wrong)), get_task("add_cot"), failing)
    assert baseline["gate"] == "FAIL"
    save(tmpdir, "step1_baseline", baseline)
    check("FAIL baseline still writes a file", (tmpdir / "step1_baseline.summary.json").exists())
    blocked = False
    try:
        check_step1_gate_passed(tmpdir)
    except SystemExit:
        blocked = True
    check("step2 blocked when recorded gate=FAIL", blocked)

    ok_texts = [f"ANSWER: {e['gold']}" for e in failing]
    ok_base = run_baseline(FakeBackend(generate_script=list(ok_texts)), get_task("add_cot"), failing)
    save(tmpdir, "step1_baseline", ok_base)
    allowed = True
    try:
        check_step1_gate_passed(tmpdir)
    except SystemExit:
        allowed = False
    check("step2 allowed when recorded gate=PASS", allowed)

    missing_dir = tmpdir / "empty"
    missing_dir.mkdir()
    missing_blocked = False
    try:
        check_step1_gate_passed(missing_dir)
    except SystemExit:
        missing_blocked = True
    check("step2 blocked when no step1 file", missing_blocked)
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

# ---------------------------------------------------------------------------
# F3 backend defaults
# ---------------------------------------------------------------------------

check(
    "hf default is not an mlx-community id",
    default_model_for("mlx") != default_model_for("hf")
    and "mlx-community" not in default_model_for("hf"),
)
check("all documented tasks are registered", len(TASKS) >= 11)

# ---------------------------------------------------------------------------
# session accounting + probe fork
# ---------------------------------------------------------------------------

from early_exit.session import DecodeSession, StatelessSession
from early_exit.timing import GenerateResult

row_calls = run_probe_and_stop(
    FakeBackend(
        continue_script=["chunk1", f"FINAL: {label}", "chunk2", f"FINAL: {label}"]
    ),
    get_task("bitcount_cot"),
    one,
    k=8,
    stability=2,
)["rows"][0]
check("probe row records n_generate_calls", row_calls["n_generate_calls"] >= 4)
check(
    "probe row splits reason vs probe time",
    "reason_prefill_s" in row_calls and "probe_decode_s" in row_calls,
)
check("cold fake session is not cache_enabled", row_calls["cache_enabled"] is False)
check(
    "call_log has one entry per generate",
    len(row_calls["call_log"]) == row_calls["n_generate_calls"],
)

sess = StatelessSession(FakeBackend(continue_script=["AAA", "PROBE", "BBB"]), "PREFIX")
first = sess.continue_decode(8)
probe = sess.probe("\nSUFFIX", 8)
second = sess.continue_decode(8)
check("stateless continue appends to prefix", first.text == "AAA" and second.text == "BBB")
check("stateless probe does not steal the next continue", probe.text == "PROBE")


class RecordingCacheSession(DecodeSession):
    cache_enabled = True

    def __init__(self):
        self.main = ["ROOT"]
        self.probes: list[str] = []

    def continue_decode(self, max_new: int) -> GenerateResult:
        self.main.append("DEC")
        return GenerateResult(
            text="ANSWER: 12\n",
            n_tokens=4,
            latency_s=0.01,
            prefill_tokens=0,
            decode_tokens=4,
            decode_s=0.01,
            cache_hit=True,
            kind="decode",
        )

    def probe(self, suffix: str, max_new: int) -> GenerateResult:
        self.probes.append(suffix)
        return GenerateResult(
            text=" 12",
            n_tokens=1,
            latency_s=0.002,
            prefill_tokens=len(suffix),
            decode_tokens=1,
            prefill_s=0.001,
            decode_s=0.001,
            cache_hit=True,
            kind="probe",
        )


class CachedFake(FakeBackend):
    use_cache = True
    cache_enabled = True

    def start_session(self, prompt: str):
        if not hasattr(self, "sessions"):
            self.sessions = []
        sess = RecordingCacheSession()
        self.sessions.append(sess)
        return sess


cfb = CachedFake()
cached_run = run_probe_and_stop(cfb, get_task("add_cot"), add_one, k=4, stability=2)
crow = cached_run["rows"][0]
check("cached path sets cache_enabled on the result", cached_run["cache_enabled"] is True)
check(
    "probe fork does not append onto the main token list",
    cfb.sessions[0].main == ["ROOT", "DEC", "DEC"] and len(cfb.sessions[0].probes) == 2,
)
check("cached probe tokens are suffix-only prefill", crow["probe_prefill_tokens"] > 0)

base_row = run_baseline(
    FakeBackend(generate_script=[f"ANSWER: {add_one[0]['gold']}"]),
    get_task("add_cot"),
    add_one,
)["rows"][0]
check("baseline logs prefill/decode fields", "prefill_s" in base_row and "n_generate_calls" in base_row)

from early_exit.throughput import attach_throughput, format_table, row_from_result

synth = attach_throughput(
    {
        "step": 1,
        "cache_enabled": True,
        "n": 2,
        "accuracy": 1.0,
        "avg_tokens": 6.0,
        "rows": [
            {"tokens": 6, "latency_s": 2.0, "prefill_tokens": 100, "prefill_s": 1.0, "decode_tokens": 6, "decode_s": 1.0},
            {"tokens": 6, "latency_s": 2.0, "prefill_tokens": 100, "prefill_s": 1.0, "decode_tokens": 6, "decode_s": 1.0},
        ],
    }
)
check("examples/s is n / wall", abs(synth["throughput"]["examples_per_s"] - 0.5) < 1e-9)
check("reason tok/s is tokens / wall", abs(synth["reason_tokens_per_s"] - 3.0) < 1e-9)
check("decode tok/s uses decode seconds only", abs(synth["throughput"]["decode_tokens_per_s"] - 6.0) < 1e-9)

cold = row_from_result(
    {
        "step": 2,
        "k": 4,
        "stability": 2,
        "cache_enabled": False,
        "n": 1,
        "accuracy": 1.0,
        "avg_reason_tokens": 7.0,
        "rows": [
            {
                "reason_tokens": 7,
                "total_latency_s": 5.0,
                "reason_prefill_tokens": 282,
                "probe_prefill_tokens": 319,
                "reason_prefill_s": 2.5,
                "probe_prefill_s": 2.5,
                "reason_decode_s": 0.0,
                "probe_decode_s": 0.0,
            }
        ],
    }
)
warm = row_from_result(synth)
table = format_table([warm, cold], baseline=warm)
check("comparison table has vs-base column", "vs base" in table and "1.00x" in table)
check("cold probe bills the repeated prefix", cold["prefill_billed"] == 601)

# ---------------------------------------------------------------------------
n_pass = sum(1 for _, s, _ in results if s == "PASS")
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
print(f"\n{n_pass} passed, {n_fail} failed, {len(results)} total")
if n_fail:
    print("\nFAILED CHECKS:")
    for name, status, detail in results:
        if status == "FAIL":
            print(f"  - {name}: {detail}")
    sys.exit(1)
sys.exit(0)
