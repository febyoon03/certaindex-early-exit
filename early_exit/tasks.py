"""Experiment variants as data, not as 17 copied scripts.

Each task owns: prompt, example factory, gold label, extract kind, output
directory, and the probe suffix that must match the answer format.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

Example = dict[str, Any]
ExampleFactory = Callable[[int, int], list[Example]]


BITCOUNT_COT_PROMPT = """Count the number of 1s in the binary sequence one by one.
Then decide whether the count is greater than {threshold}.
Write your counting steps first. Do not give the final answer until the end.
When you finish, write exactly one of:
FINAL: TRUE
FINAL: FALSE

Sequence: {seq}
"""

BITCOUNT_FEWSHOT_PROMPT = """Count the 1s in the sequence. Do not list bits one by one.
Reply with exactly two lines:
COUNT: <number>
FINAL: TRUE
or
COUNT: <number>
FINAL: FALSE
TRUE means the count is greater than {threshold}.

Example 1
Sequence: {ex1_seq}
COUNT: {ex1_count}
FINAL: {ex1_label}

Example 2
Sequence: {ex2_seq}
COUNT: {ex2_count}
FINAL: {ex2_label}

Example 3
Sequence: {ex3_seq}
COUNT: {ex3_count}
FINAL: {ex3_label}

Now solve this one.
Sequence: {seq}
"""

BITCOUNT_COUNTONLY_PROMPT = """Count the 1s in the sequence. Do not list bits one by one.
Reply with exactly one line:
COUNT: <number>

Example 1
Sequence: {ex1_seq}
COUNT: {ex1_count}

Example 2
Sequence: {ex2_seq}
COUNT: {ex2_count}

Example 3
Sequence: {ex3_seq}
COUNT: {ex3_count}

Now solve this one.
Sequence: {seq}
"""

ADD_DIRECT_PROMPT = """Add the two numbers. Reply with exactly one line:
ANSWER: <number>

Example 1
What is 14 + 23?
ANSWER: 37

Example 2
What is 50 + 19?
ANSWER: 69

Example 3
What is 8 + 7?
ANSWER: 15

Now solve this one.
What is {a} + {b}?
"""

ADD_COT_PROMPT = """Add the two numbers. Write a short calculation, then the answer.
Reply with the working, then exactly one last line:
ANSWER: <number>

Example 1
What is 14 + 23?
14 + 23 = 37
ANSWER: 37

Example 2
What is 50 + 19?
50 + 19 = 69
ANSWER: 69

Now solve this one.
What is {a} + {b}?
"""

ADD_CHECK_PROMPT = """Add the two numbers. Write the working, then the answer, then a one-line check.
Reply with exactly three lines in this order:
<a> + <b> = <sum>
ANSWER: <sum>
Check: <sum> - <b> = <a>. OK.

Example 1
What is 14 + 23?
14 + 23 = 37
ANSWER: 37
Check: 37 - 23 = 14. OK.

Example 2
What is 50 + 19?
50 + 19 = 69
ANSWER: 69
Check: 69 - 19 = 50. OK.

Now solve this one.
What is {a} + {b}?
"""

TF_PROBE_SUFFIX = (
    "\nOh, I suddenly got the answer to the whole problem. "
    "Final Answer: "
)
INT_PROBE_SUFFIX = (
    "\nOh, I suddenly got the answer to the whole problem.\nANSWER:"
)


def _bitcount_examples(n: int, seed: int, seq_len: int, threshold: int) -> list[Example]:
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        bits = [rng.choice("01") for _ in range(seq_len)]
        seq = "".join(bits)
        count = seq.count("1")
        examples.append(
            {
                "id": i,
                "seq": seq,
                "count": count,
                "label": "TRUE" if count > threshold else "FALSE",
                "gold": "TRUE" if count > threshold else "FALSE",
            }
        )
    return examples


def _countonly_examples(n: int, seed: int, seq_len: int, threshold: int) -> list[Example]:
    rows = _bitcount_examples(n, seed, seq_len, threshold)
    for row in rows:
        row["gold"] = row["count"]
        row["label"] = row["count"]
    return rows


def _add_examples(n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        a = rng.randint(10, 40)
        b = rng.randint(10, 40)
        answer = a + b
        examples.append({"id": i, "a": a, "b": b, "answer": answer, "gold": answer, "label": answer})
    return examples


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    out_dir: Path
    extract_kind: str  # "tf" | "int"
    max_reason_tokens: int
    min_baseline_acc: float
    seed: int
    make_examples: ExampleFactory
    format_prompt: Callable[[Example], str]
    probe_suffix: str
    parse_stop_kind: str | None  # "int" or None


def _fs_fields(seq_len: int, threshold: int, shots: list[tuple[str, int]]) -> dict[str, Any]:
    out: dict[str, Any] = {"threshold": threshold}
    for i, (seq, count) in enumerate(shots, start=1):
        out[f"ex{i}_seq"] = seq
        out[f"ex{i}_count"] = count
        out[f"ex{i}_label"] = "TRUE" if count > threshold else "FALSE"
    return out


def _bitcount_task(
    name: str,
    out_dir: str,
    seq_len: int,
    threshold: int,
    prompt: str,
    shots: list[tuple[str, int]] | None,
    extract_kind: str,
    count_as_gold: bool,
    max_reason_tokens: int = 96,
) -> Task:
    extra = _fs_fields(seq_len, threshold, shots or [])

    def make(n: int, seed: int) -> list[Example]:
        if count_as_gold:
            return _countonly_examples(n, seed, seq_len, threshold)
        return _bitcount_examples(n, seed, seq_len, threshold)

    def format_prompt(ex: Example) -> str:
        return prompt.format(seq=ex["seq"], **extra)

    return Task(
        name=name,
        description=f"bit-count seq_len={seq_len} threshold={threshold} extract={extract_kind}",
        out_dir=Path(out_dir),
        extract_kind=extract_kind,
        max_reason_tokens=max_reason_tokens,
        min_baseline_acc=0.70,
        seed=42,
        make_examples=make,
        format_prompt=format_prompt,
        probe_suffix=TF_PROBE_SUFFIX if extract_kind == "tf" else INT_PROBE_SUFFIX,
        parse_stop_kind="int" if extract_kind == "int" else None,
    )


def _add_task(name: str, out_dir: str, prompt: str) -> Task:
    def format_prompt(ex: Example) -> str:
        return prompt.format(a=ex["a"], b=ex["b"])

    return Task(
        name=name,
        description="two-digit addition, integer exact match",
        out_dir=Path(out_dir),
        extract_kind="int",
        max_reason_tokens=96,
        min_baseline_acc=0.70,
        seed=42,
        make_examples=_add_examples,
        format_prompt=format_prompt,
        probe_suffix=INT_PROBE_SUFFIX,
        parse_stop_kind="int",
    )


TASKS: dict[str, Task] = {
    "bitcount_cot": _bitcount_task(
        "bitcount_cot",
        "runs/track1/bitcount_cot",
        seq_len=20,
        threshold=10,
        prompt=BITCOUNT_COT_PROMPT,
        shots=None,
        extract_kind="tf",
        count_as_gold=False,
    ),
    "bitcount_cot_budget256": _bitcount_task(
        "bitcount_cot_budget256",
        "runs/track1/bitcount_cot_budget256",
        seq_len=20,
        threshold=10,
        prompt=BITCOUNT_COT_PROMPT,
        shots=None,
        extract_kind="tf",
        count_as_gold=False,
        max_reason_tokens=256,
    ),
    "bitcount_fewshot": _bitcount_task(
        "bitcount_fewshot",
        "runs/track1/step1_fewshot",
        seq_len=20,
        threshold=10,
        prompt=BITCOUNT_FEWSHOT_PROMPT,
        shots=[
            ("11100000000000000000", 3),
            ("11111111111000000000", 11),
            ("01010101010101010101", 10),
        ],
        extract_kind="tf",
        count_as_gold=False,
    ),
    "bitcount_1p5b_fewshot": _bitcount_task(
        "bitcount_1p5b_fewshot",
        "runs/track1/step1_1p5b",
        seq_len=20,
        threshold=10,
        prompt=BITCOUNT_FEWSHOT_PROMPT,
        shots=[
            ("11100000000000000000", 3),
            ("11111111111000000000", 11),
            ("01010101010101010101", 10),
        ],
        extract_kind="tf",
        count_as_gold=False,
    ),
    "bitcount_len8": _bitcount_task(
        "bitcount_len8",
        "runs/track1/step1_1p5b_len8",
        seq_len=8,
        threshold=4,
        prompt=BITCOUNT_FEWSHOT_PROMPT,
        shots=[
            ("11100000", 3),
            ("11111000", 5),
            ("01010101", 4),
        ],
        extract_kind="tf",
        count_as_gold=False,
    ),
    "bitcount_len8_countonly": _bitcount_task(
        "bitcount_len8_countonly",
        "runs/track1/step1_1p5b_len8_countonly",
        seq_len=8,
        threshold=4,
        prompt=BITCOUNT_COUNTONLY_PROMPT,
        shots=[
            ("11100000", 3),
            ("11111000", 5),
            ("01010101", 4),
        ],
        extract_kind="int",
        count_as_gold=True,
    ),
    "bitcount_3b": _bitcount_task(
        "bitcount_3b",
        "runs/track1/step1_3b_count",
        seq_len=8,
        threshold=4,
        prompt=BITCOUNT_COUNTONLY_PROMPT,
        shots=[
            ("11100000", 3),
            ("11111000", 5),
            ("01010101", 4),
        ],
        extract_kind="int",
        count_as_gold=True,
    ),
    "bitcount_3b_fs": _bitcount_task(
        "bitcount_3b_fs",
        "runs/track1/step1_3b_count_fs",
        seq_len=8,
        threshold=4,
        prompt=BITCOUNT_COUNTONLY_PROMPT,
        shots=[
            ("10000000", 1),
            ("11111100", 6),
            ("11111000", 5),
        ],
        extract_kind="int",
        count_as_gold=True,
    ),
    "add_direct": _add_task("add_direct", "runs/track1/step1_3b_add", ADD_DIRECT_PROMPT),
    "add_cot": _add_task("add_cot", "runs/track1/step1_3b_add_cot", ADD_COT_PROMPT),
    "add_check": _add_task(
        "add_check", "runs/track1/step1_3b_add_check", ADD_CHECK_PROMPT
    ),
}

# Old script filename → task name (compatibility wrappers use this).
SCRIPT_TO_TASK = {
    "step0_2_early_exit.py": "bitcount_cot",
    "step0_2_early_exit_v2.py": "bitcount_cot",
    "step0_2_early_exit_budget256.py": "bitcount_cot_budget256",
    "step0_2_early_exit_fewshot.py": "bitcount_fewshot",
    "step0_2_early_exit_1p5b.py": "bitcount_1p5b_fewshot",
    "step0_2_early_exit_len8.py": "bitcount_len8",
    "step0_2_early_exit_len8_countonly.py": "bitcount_len8_countonly",
    "step0_2_early_exit_3b.py": "bitcount_3b",
    "step0_2_early_exit_3b_fs.py": "bitcount_3b_fs",
    "step0_2_early_exit_add.py": "add_direct",
    "step0_2_early_exit_add_cot.py": "add_cot",
    "step0_2_early_exit_add_check_baseline.py": "add_check",
    "step0_2_early_exit_add_probe.py": "add_cot",
    "step0_2_early_exit_add_parsestop.py": "add_cot",
    "step0_2_early_exit_add_parsestop_v2.py": "add_cot",
    "step0_2_early_exit_add_check_parsestop.py": "add_check",
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        known = ", ".join(sorted(TASKS))
        raise SystemExit(f"unknown task {name!r}. known: {known}")
    return TASKS[name]
