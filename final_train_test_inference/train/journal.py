"""Per-model journal files."""

from __future__ import annotations

import time
from pathlib import Path

from final_train.constants import RESULTS


def log_path(model_name: str) -> Path:
    return RESULTS / model_name / "journal.log"


def log(msg: str, model_name: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{model_name}] {msg}"
    print(line, flush=True)
    path = log_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
