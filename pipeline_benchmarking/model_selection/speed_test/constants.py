"""Constants for stage03 speed benchmarks."""

from __future__ import annotations

from shared.paths import SEED, STAGE03

RESULTS = STAGE03 / "results"

SPEED_N_TARGETS = 5
OPTUNA_TRIALS_SPEED = 5
EARLY_STOPPING_ROUNDS = 30

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
