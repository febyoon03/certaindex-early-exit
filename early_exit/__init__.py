"""Track 1 early-exit experiment harness.

Responsibility: define tasks, run generation steps, extract answers, and
enforce sequential gates. Model I/O lives in backends; scoring lives in
extract; experiment variants live in tasks.
"""

from early_exit.extract import extract_answer, extract_tf, extract_int
from early_exit.tasks import TASKS, get_task

__all__ = ["TASKS", "get_task", "extract_answer", "extract_tf", "extract_int"]
