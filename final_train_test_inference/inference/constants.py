from __future__ import annotations

from pathlib import Path

# local dirs
TOTAL_INFERENCE = Path(__file__).resolve().parent
FINAL_VERSION = TOTAL_INFERENCE.parent
WORKSPACE = FINAL_VERSION.parent

# paths on my workspace
MODELS_ROOT = FINAL_VERSION / "final_train" / "results" # model weights (available on kaggle)
WEIGHTS_DIR = MODELS_ROOT / "ensemble" / ENSEMBLE_ID / "weights"
K1_REF_PATH = FINAL_VERSION / "final_train" / "splits" / "sc_k1" / "X_train.parquet" # Imputing ref - singlecell train dataset on K1 level (available on kaggle)
ENSEMBLE_ID = "catboost_tabm_resnet_stack"
CONFIG_PATH = "target_config.json"
GENE_LENGTHS_PATH = "df_gene_mapping.parquet"
MRNA_NAMES_PATH = "mRNA_names.json"
GENE_MAPPING_PATH = "ensembl_gene_mapping.csv"
STACK_MODELS = ("catboost_optuna", "tabm", "resnet")


def resolve_gene_mapping_path(explicit: Path | str | None = None) -> str:
    if explicit is not None:
        return str(explicit)
    for path in GENE_MAPPING_CANDIDATES:
        if path.is_file():
            return str(path)
    return str(GENE_MAPPING_CANDIDATES[0])
