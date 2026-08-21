#!/usr/bin/env python3
"""Compatibility entry for step0_2_early_exit_v2.py. Delegates to the unified harness."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _compat import run_named
if __name__ == "__main__":
    run_named('step0_2_early_exit_v2.py')
