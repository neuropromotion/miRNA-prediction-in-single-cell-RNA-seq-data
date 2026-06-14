"""Journal logging for stage03 model screen."""

from __future__ import annotations

import time

from shared.paths import STAGE03

LOG_PATH = STAGE03 / "journal.log"


def log(msg: str, tag: str = "screen") -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{tag}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
