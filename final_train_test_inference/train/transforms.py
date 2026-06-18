"""log2(x+1) transform for expression matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log2p1(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(dtype=np.float64)
    out = np.log2(np.clip(arr, 0.0, None) + 1.0)
    return pd.DataFrame(out, index=df.index, columns=df.columns)
