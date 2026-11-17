"""Read/write per-method prediction files: predictions/<method>_test.jsonl."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["doc_id"]] = r
    return out
