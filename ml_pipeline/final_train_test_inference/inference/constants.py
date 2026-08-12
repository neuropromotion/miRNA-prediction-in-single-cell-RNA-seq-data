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

ENSEMBLE_ID = "tabpack_dcnv2_tabm_stack"

# Prefer packaged ../models (publish via train/publish_models.sh); else train/results.
_PACKAGED = FTTI_ROOT / "models"
_TRAIN_RESULTS = FTTI_ROOT / "train" / "results"
_packaged_weights = _PACKAGED / "ensemble" / ENSEMBLE_ID / "weights"
MODELS_ROOT = (
    _PACKAGED
    if _packaged_weights.is_dir() and any(_packaged_weights.glob("*.json"))
    else _TRAIN_RESULTS
)
WEIGHTS_DIR = MODELS_ROOT / "ensemble" / ENSEMBLE_ID / "weights"

# KNN reference for K1 imputation (from training splits)
K1_REF_PATH = ML_PIPELINE / "data" / "splits" / "sc_k1" / "X_train.parquet"

# Batch inference I/O (scRNA datasets — NOT in git; see data/README.md)
INFERENCE_INPUT_DIR = ML_PIPELINE / "data" / "inference_inputs"
INFERENCE_OUTPUT_DIR = ML_PIPELINE / "data" / "inference_outputs"

CONFIG_PATH = INFERENCE_DIR / "prediction_config.json"
GENE_LENGTHS_PATH = INFERENCE_DIR / "df_gene_mapping.parquet"
MRNA_NAMES_PATH = INFERENCE_DIR / "mRNA_names.json"
GENE_MAPPING_PATH = INFERENCE_DIR / "ensembl_gene_mapping.csv"
GENE_MAPPING_CANDIDATES = (GENE_MAPPING_PATH,)

STACK_MODELS = ("tabpack", "dcnv2", "tabm")
MANIFEST_PATH = CONFIG_PATH


def resolve_gene_mapping_path(explicit: Path | str | None = None) -> str:
    if explicit is not None:
        return str(explicit)
    for path in GENE_MAPPING_CANDIDATES:
        if path.is_file():
            return str(path)
    return str(GENE_MAPPING_CANDIDATES[0])


def parse_prediction_config(config: dict) -> tuple[list[str], dict[str, list[str]], dict[str, dict]]:
    """
    Normalize prediction_config.json into (eligible_mirs, cohorts, target_info).

    Supports:
      - v2 Optimal_K format: cohorts is a list of names; per-cohort maps under
        config["K1"], … with per-miRNA ``features`` / metrics.
      - legacy: cohorts is dict name→mirnas; optional config["targets"] with ``genes``.
    """
    eligible = list(config["eligible_mirs"])
    raw_cohorts = config["cohorts"]

    if isinstance(raw_cohorts, dict) and "targets" in config:
        cohorts = {str(k): list(v) for k, v in raw_cohorts.items()}
        target_info: dict[str, dict] = {}
        for mir, info in config["targets"].items():
            genes = list(info.get("genes") or info.get("features") or [])
            target_info[str(mir)] = {
                **dict(info),
                "assigned_cohort": str(info.get("assigned_cohort") or ""),
                "genes": genes,
                "features": genes,
            }
        return eligible, cohorts, target_info

    if isinstance(raw_cohorts, dict):
        cohort_names = list(raw_cohorts.keys())
        # May already be name→mirna list, or need building from config[K] blocks.
        sample_val = next(iter(raw_cohorts.values()), None)
        if isinstance(sample_val, list):
            cohorts = {str(k): list(v) for k, v in raw_cohorts.items()}
        else:
            cohort_names = [str(k) for k in raw_cohorts]
            cohorts = {}
    else:
        cohort_names = [str(c) for c in raw_cohorts]
        cohorts = {}

    target_info = {}
    for cname in cohort_names:
        block = config.get(cname) or {}
        if not isinstance(block, dict):
            continue
        mirs: list[str] = []
        for mir, info in block.items():
            info = dict(info or {})
            genes = list(info.get("features") or info.get("genes") or [])
            mirs.append(str(mir))
            target_info[str(mir)] = {
                **info,
                "assigned_cohort": cname,
                "genes": genes,
                "features": genes,
            }
        cohorts[cname] = mirs

    return eligible, cohorts, target_info
