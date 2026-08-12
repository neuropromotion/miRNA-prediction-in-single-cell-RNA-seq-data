"""TabR / RealTabR via pytabkit (official-style defaults, sklearn API)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import logging
import torch

for _n in ("lightning.pytorch.utilities.seed","pytorch_lightning.utilities.seed","lightning_fabric.utilities.seed"):
    logging.getLogger(_n).setLevel(logging.ERROR)

Variant = Literal["tabr", "realtabr"]


class _GpuIndexFlatConfig:
    def __init__(self) -> None:
        self.device = 0


class _CpuFlatL2Bridge:
    """faiss-cpu stand-in for faiss.GpuIndexFlatL2 used by pytabkit TabR on CUDA."""

    def __init__(self, d: int) -> None:
        import faiss

        self._index = faiss.IndexFlatL2(int(d))

    def reset(self) -> None:
        self._index.reset()

    def add(self, x) -> None:  # noqa: ANN001
        if torch.is_tensor(x):
            x = x.detach().float().cpu().contiguous().numpy()
        self._index.add(np.asarray(x, dtype=np.float32))

    def search(self, x, k: int):  # noqa: ANN001
        device = x.device if torch.is_tensor(x) else torch.device("cpu")
        if torch.is_tensor(x):
            xn = x.detach().float().cpu().contiguous().numpy()
        else:
            xn = np.asarray(x, dtype=np.float32)
        distances, idx = self._index.search(xn, int(k))
        return (
            torch.as_tensor(distances, device=device),
            torch.as_tensor(idx, device=device),
        )


def _ensure_faiss_cuda_bridge() -> None:
    """pytabkit TabR uses faiss.GpuIndexFlat* on CUDA; faiss-cpu lacks those."""
    import faiss

    if hasattr(faiss, "GpuIndexFlatConfig") and hasattr(faiss, "GpuIndexFlatL2"):
        # Still override if the attrs exist but are unusable (some broken installs).
        try:
            faiss.GpuIndexFlatConfig()
            return
        except Exception:
            pass

    faiss.GpuIndexFlatConfig = _GpuIndexFlatConfig  # type: ignore[attr-defined]
    faiss.StandardGpuResources = lambda: None  # type: ignore[attr-defined]
    faiss.GpuIndexFlatL2 = lambda _res, d, _cfg=None: _CpuFlatL2Bridge(d)  # type: ignore[attr-defined]


def _clear_search_indexes(obj: Any) -> None:
    """Drop faiss indexes so the fitted estimator is picklable."""
    seen: set[int] = set()

    def walk(x: Any) -> None:
        if x is None:
            return
        xid = id(x)
        if xid in seen:
            return
        seen.add(xid)
        if hasattr(x, "search_index"):
            try:
                setattr(x, "search_index", None)
            except Exception:
                pass
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)
            return
        d = getattr(x, "__dict__", None)
        if isinstance(d, dict):
            for v in d.values():
                walk(v)

    walk(obj)


def _make_model(variant: Variant, device: str, **kwargs):
    from pytabkit import RealTabR_D_Regressor, TabR_S_D_Regressor

    _ensure_faiss_cuda_bridge()
    common = dict(
        device=device,
        n_cv=1,
        n_refit=0,
        val_metric_name="r2",
        verbosity=0,
        random_state=42,
        n_threads=8,
        val_fraction=0.0,
    )
    common.update(kwargs)
    if variant == "realtabr":
        return RealTabR_D_Regressor(**common)
    return TabR_S_D_Regressor(**common)


def train_tabr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_dir: Path,
    device: str = "cuda",
    variant: Variant = "tabr",
) -> tuple[Any, dict[str, Any]]:
    """Fit TabR and return (model, info). Model is kept in memory for immediate eval."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model = _make_model(variant, device=device)
    model.fit(
        np.asarray(x_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.float64),
        X_val=np.asarray(x_val, dtype=np.float32),
        y_val=np.asarray(y_val, dtype=np.float64),
    )
    (model_dir / "meta.json").write_text(
        json.dumps({"variant": variant, "device": device, "arch": f"TabR:{variant}"}, indent=2),
        encoding="utf-8",
    )
    return model, {"variant": variant}


def save_tabr(model: Any, model_dir: Path) -> None:
    """Pickle after clearing faiss indexes (recreated lazily on predict)."""
    _ensure_faiss_cuda_bridge()
    _clear_search_indexes(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")


def load_tabr(model_dir: Path):
    _ensure_faiss_cuda_bridge()
    return joblib.load(model_dir / "model.joblib")


def predict_tabr(model, x: np.ndarray) -> np.ndarray:
    _ensure_faiss_cuda_bridge()
    return np.asarray(model.predict(np.asarray(x, dtype=np.float32)), dtype=np.float64).reshape(-1)
