#!/usr/bin/env python3
"""TabPFN-3 auth / download smoke test."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

token = os.environ.get("TABPFN_TOKEN", "").strip()
hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
print(f"TABPFN_TOKEN present: {bool(token)}")
print(f"HF_TOKEN present: {bool(hf)}")

try:
    from tabpfn import browser_auth

    api_url = browser_auth.settings.tabpfn.auth_api_url
    if token and hasattr(browser_auth, "verify_token"):
        print(f"verify_token: {browser_auth.verify_token(token, api_url)}")
except Exception as exc:
    print(f"verify_token error: {type(exc).__name__}: {exc}")

model_path = None
try:
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="Prior-Labs/tabpfn_3",
        filename="tabpfn-v3-regressor-v3_default.ckpt",
        token=hf or None,
    )
    print(f"hf download OK: {bool(model_path)}")
except Exception as exc:
    print(f"hf download FAIL: {type(exc).__name__}: {exc}")

try:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    x = np.random.randn(200, 10).astype(np.float32)
    y = np.random.randn(200).astype(np.float32)
    kwargs = {"device": "cuda"}
    if model_path:
        kwargs["model_path"] = model_path
    model = TabPFNRegressor.create_default_for_version(ModelVersion.V3, **kwargs)
    model.fit(x, y)
    pred = model.predict(x[:5])
    print(f"smoke fit/predict OK shape={pred.shape}")
except Exception as exc:
    print(f"smoke FAIL: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)
