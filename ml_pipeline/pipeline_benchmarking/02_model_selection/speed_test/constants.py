"""Constants for stage03 speed benchmarks."""

from __future__ import annotations

from shared.paths import SEED, STAGE03

RESULTS = STAGE03 / "results"

SPEED_N_TARGETS = 5
OPTUNA_TRIALS_SPEED = 5
EARLY_STOPPING_ROUNDS = 30

# Fixed pilot targets used in the original speed_benchmark run
SPEED_TARGETS = (
    "hsa-mir-125a-5p",
    "hsa-mir-301b-5p",
    "hsa-mir-411-5p",
    "hsa-mir-487a-3p",
    "hsa-mir-99b-3p",
)

CANDIDATE_MODELS = ("lassonet", "gandalf")

XGB_DEFAULT = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": SEED,
}

# 7 models kept for screen (TabPFN-3 tested separately in tabpfn3_speed_benchmark.py)
SPEED_MODELS = (
    "xgb_default",
    "xgb_optuna",
    "dcnv2",
    "realmlp",
    "resnet",
    "tabm",
    "tabnet",
)
