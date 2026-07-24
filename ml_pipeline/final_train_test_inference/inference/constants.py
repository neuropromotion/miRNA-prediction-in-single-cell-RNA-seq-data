"""Paths for stack inference (final_train_test_inference/inference/)."""

from __future__ import annotations

from pathlib import Path

# Directory layout (this file lives in .../inference/constants.py):
#   INFERENCE_DIR  -> final_train_test_inference/inference/
#   FTTI_ROOT      -> final_train_test_inference/
#   ML_PIPELINE    -> ml_pipeline root
INFERENCE_DIR = Path(__file__).resolve().parent
FTTI_ROOT = INFERENCE_DIR.parent
ML_PIPELINE = FTTI_ROOT.parent

# Back-compat aliases (old FINAL_VERSION naming) — prefer names above
TOTAL_INFERENCE = INFERENCE_DIR
FINAL = FTTI_ROOT
FINAL_VERSION = FTTI_ROOT

ENSEMBLE_ID = "catboost_tabm_resnet_stack"

# Pretrained weights + base models (download separately; see data/README.md)
MODELS_ROOT = FTTI_ROOT / "models"
WEIGHTS_DIR = MODELS_ROOT / "ensemble" / ENSEMBLE_ID / "weights"

# KNN reference for K1 imputation (from training splits)
K1_REF_PATH = ML_PIPELINE / "data" / "splits" / "sc_k1" / "X_train.parquet"

# Batch inference I/O (scRNA datasets — NOT in git; see data/README.md)
INFERENCE_INPUT_DIR = ML_PIPELINE / "data" / "inference_inputs"
INFERENCE_OUTPUT_DIR = ML_PIPELINE / "data" / "inference_outputs"

CONFIG_PATH = INFERENCE_DIR / "target_config.json"
GENE_LENGTHS_PATH = INFERENCE_DIR / "df_gene_mapping.parquet"
MRNA_NAMES_PATH = INFERENCE_DIR / "mRNA_names.json"
GENE_MAPPING_PATH = INFERENCE_DIR / "ensembl_gene_mapping.csv"
GENE_MAPPING_CANDIDATES = (GENE_MAPPING_PATH,)

STACK_MODELS = ("catboost_optuna", "tabm", "resnet")
MANIFEST_PATH = CONFIG_PATH


def resolve_gene_mapping_path(explicit: Path | str | None = None) -> str:
    if explicit is not None:
        return str(explicit)
    for path in GENE_MAPPING_CANDIDATES:
        if path.is_file():
            return str(path)
    return str(GENE_MAPPING_CANDIDATES[0])
