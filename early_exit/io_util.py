from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def drop_texts(obj: dict) -> dict:
    out = dict(obj)
    rows = []
    for r in obj.get("rows", []):
        rr = {
            k: v
            for k, v in r.items()
            if k not in {"text", "reason_text", "probes"}
        }
        if "probes" in r:
            rr["n_probes"] = r.get("n_probes")
            rr["probe_answers"] = [p.get("extracted") for p in r["probes"]]
        rows.append(rr)
    out["rows"] = rows
    return out


def save(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    full = out_dir / f"{name}.json"
    summary = out_dir / f"{name}.summary.json"
    full.write_text(json.dumps(payload, indent=2))
    summary.write_text(json.dumps(drop_texts(payload), indent=2))
    return summary


def print_gate(payload: dict) -> None:
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    print(f"\nGATE={payload.get('gate', 'n/a')}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
