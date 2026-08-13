"""Journal logging for model_tuning."""

from __future__ import annotations

import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "journal.log"


def log(msg: str, tag: str = "tuning") -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{tag}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
