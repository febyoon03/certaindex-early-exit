"""Thin wrappers so old script names still launch the unified CLI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from early_exit.cli import main
from early_exit.tasks import SCRIPT_TO_TASK


def run_named(script_filename: str) -> None:
    task = SCRIPT_TO_TASK[script_filename]
    argv = list(sys.argv[1:])
    if "--task" not in argv:
        argv = ["--task", task, *argv]
    # parse-stop scripts default to that mode if --step is absent
    if "parsestop" in script_filename and "--mode" not in argv and "--step" not in argv:
        argv = ["--mode", "parsestop", *argv]
    if "--step" not in argv and "--mode" not in argv:
        argv = ["--step", "0", *argv]
    main(argv)
