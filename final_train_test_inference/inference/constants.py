from __future__ import annotations

from pathlib import Path

TOTAL_INFERENCE = Path(__file__).resolve().parent
FINAL_VERSION = TOTAL_INFERENCE.parent
WORKSPACE = FINAL_VERSION.parent

# paths on my workspace
CONFIG_PATH = FINAL_VERSION / "FINAL_CONFIG_AND_FIGURES" / "target_config.json"
MODELS_ROOT = FINAL_VERSION / "final_train" / "results"
ENSEMBLE_ID = "catboost_tabm_resnet_stack"
WEIGHTS_DIR = MODELS_ROOT / "ensemble" / ENSEMBLE_ID / "weights"
K1_REF_PATH = FINAL_VERSION / "final_train" / "splits" / "sc_k1" / "X_train.parquet"
GENE_LENGTHS_PATH = TOTAL_INFERENCE / "df_gene_mapping.parquet"
MRNA_NAMES_PATH = TOTAL_INFERENCE / "mRNA_names.json"
GENE_MAPPING_PATH = TOTAL_INFERENCE / "ensembl_gene_mapping.csv"
GENE_MAPPING_CANDIDATES = (
    GENE_MAPPING_PATH,
    WORKSPACE / "inference" / "ensembl_gene_mapping.csv",
    Path("/mnt/jack-5/amismailov/ensembl_gene_mapping.csv"),
)

STACK_MODELS = ("catboost_optuna", "tabm", "resnet")


def resolve_gene_mapping_path(explicit: Path | str | None = None) -> str:
    if explicit is not None:
        return str(explicit)
    for path in GENE_MAPPING_CANDIDATES:
        if path.is_file():
            return str(path)
    return str(GENE_MAPPING_CANDIDATES[0])
