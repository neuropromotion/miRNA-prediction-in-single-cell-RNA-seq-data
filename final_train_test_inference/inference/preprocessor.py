import json
import os
import sys
import contextlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .constants import K1_REF_PATH, MANIFEST_PATH, resolve_gene_mapping_path
    from .stack_predictor import StackPredictor
except ImportError:
    _fv = Path(__file__).resolve().parent.parent
    if str(_fv) not in sys.path:
        sys.path.insert(0, str(_fv))
    from TOTAL_INFERENCE.constants import K1_REF_PATH, MANIFEST_PATH, resolve_gene_mapping_path
    from TOTAL_INFERENCE.stack_predictor import StackPredictor

ALLOWED_PSEUDOBULK_K = frozenset({2, 3, 4, 5, 10})
DEFAULT_KNN_REF_PATH = str(K1_REF_PATH)

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def _auto_orient_cells_by_genes(df, gene_prefix=("ENSG",)):
    cols_ok = any(str(c).startswith(gene_prefix) for c in df.columns[:50])
    rows_ok = any(str(r).startswith(gene_prefix) for r in df.index[:50])
    if cols_ok and not rows_ok:
        return df
    if rows_ok and not cols_ok:
        return df.T
    return df


def _knn_impute_zeros_cpu(X_df, zero_mask, neighbor_indices, donor_df):
    X = X_df.values.astype(np.float32).copy()
    mask = zero_mask.values
    donor = donor_df.values.astype(np.float32)
    _, n_features = X.shape
    for j in range(n_features):
        missing_idx = np.where(mask[:, j])[0]
        if len(missing_idx) == 0:
            continue
        neigh_vals = donor[neighbor_indices[missing_idx], j]
        neigh_vals = np.where(neigh_vals == 0, np.nan, neigh_vals)
        with np.errstate(all="ignore"):
            imputed = np.nanmean(neigh_vals, axis=1)
        imputed = np.where(np.isnan(imputed), 0.0, imputed)
        X[missing_idx, j] = imputed
    return pd.DataFrame(X, index=X_df.index, columns=X_df.columns)


def _run_knn_imputer(X_ref, X_query, n_neighbors=5):
    try:
        from sklearn.neighbors import NearestNeighbors
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for KNN imputation (NearestNeighbors)."
        ) from exc

    common = sorted(set(X_ref.columns) & set(X_query.columns))
    Xr = X_ref[common].copy()
    Xq = X_query[common].copy()
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(Xr.values.astype(np.float32))
    _, ind_ref = nn.kneighbors(Xr.values.astype(np.float32))
    _, ind_q = nn.kneighbors(Xq.values.astype(np.float32))
    Xr_filled = _knn_impute_zeros_cpu(Xr, Xr == 0, ind_ref, Xr)
    Xq_filled = _knn_impute_zeros_cpu(Xq, Xq == 0, ind_q, Xr_filled)
    return Xr_filled, Xq_filled


def align_and_knn_impute(X_query, required_cols, X_ref_knn, n_neighbors=5):
    out = X_query.copy()
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        miss_df = pd.DataFrame(0.0, index=out.index, columns=missing, dtype=np.float32)
        out = pd.concat([out, miss_df], axis=1)
    Xq = out[list(required_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    Xref = X_ref_knn.copy()
    missing_ref = [c for c in required_cols if c not in Xref.columns]
    if missing_ref:
        miss_df = pd.DataFrame(0.0, index=Xref.index, columns=missing_ref, dtype=np.float32)
        Xref = pd.concat([Xref, miss_df], axis=1)
    Xref = Xref[list(required_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    _, Xq_imp = _run_knn_imputer(Xref, Xq, n_neighbors=n_neighbors)
    return Xq_imp


class SingleCell:
    """
    Raw single-cell / pseudobulk preprocessing + final_train stack inference.

    Pipeline: counts → TPM/log2 → (KNN impute for K1) → CatBoost+TabM+ResNet stack.
  """

    def __init__(
        self,
        path_length="df_gene_mapping.parquet",
        path_mrna="mRNA_names.json",
        manifest_path=None,
        device="cuda",
        catboost_task="CPU",
        preload_models=False,
        log=True,
    ):
        base = Path(__file__).resolve().parent
        self._base_dir = base
        self.log = log
        self._device = device
        self._catboost_task = catboost_task
        self._manifest_path = Path(manifest_path or MANIFEST_PATH)

        path_length = Path(path_length)
        path_mrna = Path(path_mrna)
        if not path_length.is_absolute():
            path_length = base / path_length
        if not path_mrna.is_absolute():
            path_mrna = base / path_mrna

        self.gene_lengths = pd.read_parquet(path_length)
        self._predictor: StackPredictor | None = None

        with open(path_mrna, "r", encoding="utf-8") as f:
            self.standard_mrna = json.load(f)
        self.standard_mrna_set = set(self.standard_mrna)

        self.gene_lengths = self.gene_lengths[
            self.gene_lengths["gene_id"].isin(self.standard_mrna_set)
        ].copy()

        if "gene_id" not in self.gene_lengths.columns or "gene_length_kb" not in self.gene_lengths.columns:
            raise ValueError(
                "Gene length table must contain 'gene_id' and 'gene_length_kb' columns."
            )

        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        self._cohorts: dict[str, list[str]] = manifest["cohorts"]
        self._available_mirnas: list[str] = list(manifest["eligible_mirs"])

        self._knn_ref = None
        self._knn_ref_path = None

        if preload_models:
            self.load_models()

    @property
    def available_mirnas(self) -> list[str]:
        """Eligible miRNAs with trained stack models (168 by default)."""
        return list(self._available_mirnas)

    @property
    def cohorts(self) -> dict[str, list[str]]:
        """miRNA lists per inference cohort (K1 … K10)."""
        return {k: list(v) for k, v in self._cohorts.items()}

    def mirnas_for_cohort(self, cohort: str) -> list[str]:
        if cohort not in self._cohorts:
            raise KeyError(f"Unknown cohort {cohort!r}. Expected one of {list(self._cohorts)}.")
        return list(self._cohorts[cohort])

    def mirnas_for_pseudobulk_k(self, K: int) -> list[str]:
        self._validate_pseudobulk_k(K)
        return self.mirnas_for_cohort(f"K{K}")

    def _get_predictor(self) -> StackPredictor:
        if self._predictor is None:
            self.load_models()
        return self._predictor

    def load_models(self) -> StackPredictor:
        """Load (or return cached) CatBoost+TabM+ResNet stack predictor."""
        if self._predictor is not None:
            return self._predictor
        print("Loading final_train stack models...")
        self._predictor = StackPredictor(
            manifest_path=self._manifest_path,
            device=self._device,
            catboost_task=self._catboost_task,
            preload_all=False,
        )
        print(f"✔ Stack ready: {len(self._predictor.available_mirnas)} eligible miRNAs")
        return self._predictor

    def _detect_gene_axis(self, data):
        index_hits = sum(idx in self.standard_mrna_set for idx in data.index)
        col_hits = sum(col in self.standard_mrna_set for col in data.columns)
        if index_hits == 0 and col_hits == 0:
            # Fallback when input uses gene symbols instead of ENSG IDs.
            return "columns" if len(data.columns) >= len(data.index) else "index"
        if index_hits >= col_hits:
            return "index"
        return "columns"

    @staticmethod
    def _looks_like_ens_id(value):
        return isinstance(value, str) and value.startswith("ENSG")

    def _needs_symbol_to_ens_mapping(self, genes):
        gene_tokens = [gene for gene in genes if isinstance(gene, str) and gene]
        if not gene_tokens:
            return False
        if any(self._looks_like_ens_id(gene) for gene in gene_tokens):
            return False
        return True

    def standardize_mrna(self, data):
        df = data.copy(deep=True)
        gene_axis = self._detect_gene_axis(df)

        # Convert to genes x samples matrix.
        if gene_axis == "columns":
            gene_cols = [col for col in df.columns if col in self.standard_mrna_set]
            df = df[gene_cols].T
        else:
            gene_rows = [idx for idx in df.index if idx in self.standard_mrna_set]
            df = df.loc[gene_rows]

        # Keep only numeric values (important when metadata columns exist in input).
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        df = df.loc[~df.index.duplicated(keep='first')]

        missing_genes = list(self.standard_mrna_set - set(df.index))
        if missing_genes:
            missing_df = pd.DataFrame(0.0, index=missing_genes, columns=df.columns)
            df = pd.concat([df, missing_df], axis=0)

        # Strict fixed order for all datasets before normalization/inference.
        df = df.reindex(self.standard_mrna, axis=0, fill_value=0.0)
        return df

    def prepare_input(self, data, mapping_path=None):
        df = data.copy(deep=True)
        gene_axis = self._detect_gene_axis(df)

        # Normalize orientation first: genes x samples for consistent downstream logic.
        if gene_axis == "columns":
            df = df.T

        mapping_path = resolve_gene_mapping_path(mapping_path)
        if self._needs_symbol_to_ens_mapping(df.index):
            print("Detected gene symbols. Mapping to ENSG IDs...")
            df = self.replace_genes_names(df, mapping_path=mapping_path)

        return self.standardize_mrna(df)

    def TPM(self, data, log=True, enforce_mrna_standard=True):
        df = data.copy(deep=True)
        if enforce_mrna_standard:
            df = self.standardize_mrna(df)
        else:
            df = df.loc[~df.index.duplicated(keep='first')]
        
        share = sorted(list(set(df.index) & set(self.gene_lengths['gene_id'])))
        percent = len(share)*100/len(df.index)
        print(f"✔ Found length for {len(share)}/{len(df.index)} genes ({percent:.2f}%)")
        
        if len(share) == 0:
            raise ValueError("❌ Нет общих генов между данными и gene_lengths!")
        
        df_filtered = df.loc[share].copy()  
        gene_lengths_filtered = self.gene_lengths[
            self.gene_lengths['gene_id'].isin(share)
        ].set_index('gene_id')['gene_length_kb']

        rpk = df_filtered.div(gene_lengths_filtered, axis=0)
        
        library_sizes = rpk.sum(axis=0)
        tpm = rpk.div(library_sizes, axis=1) * 1e6
        
        tpm = tpm.fillna(0.0) 
        
        if log:
            return np.log2(tpm + 1)
        
        return tpm
 
    
    def replace_genes_names(
        self,
        data,
        mapping_path=None,
    ):
        mapping_path = resolve_gene_mapping_path(mapping_path)
        print("Loading HGNC → ENSG mapping...")
    
        mapping = pd.read_csv(mapping_path)
        mapping = mapping.dropna(subset=["feature_name", "feature_id"])
        mapping = mapping.drop_duplicates(subset=["feature_name"])
    
        # dict: HGNC → ENSG
        ens_map = dict(zip(mapping["feature_name"], mapping["feature_id"]))
    
        df = data.copy(deep=True)
    
        # оригинальные имена генов
        #df["original_symbol"] = df.index
    
        total_genes = df.shape[0]
        print(f"Replacing {total_genes} genes...")
    
        # маппинг index → ENSG
        df["ensembl_id"] = df.index.map(ens_map)
    
        # сколько найдено
        found = df["ensembl_id"].notna().sum()
        percent = found / total_genes * 100
    
        # удалить ненайденные
        df_clean = df.dropna(subset=["ensembl_id"]).copy()
    
        # убрать дубликаты ENSG
        df_clean = df_clean[~df_clean["ensembl_id"].duplicated()]
    
        # ENSG → индекс
        df_clean.index = df_clean.pop("ensembl_id")
        df_clean = df_clean.sort_index()
    
        print(f"✔ Found ENSG for {found}/{total_genes} genes ({percent:.2f}%)")
        print(f"✔ After removing duplicates: {len(df_clean)} unique ENSG")
    
        return df_clean
    
    def predict(
        self,
        data_tpm,
        mirnas=None,
        show_missing_report=False,
    ):
        """
        Stack inference on log2(TPM+1) matrix (cells × ENSG genes).

        Parameters
        ----------
        data_tpm
            cells × genes DataFrame.
        mirnas
            Subset of available_mirnas; default = all eligible.
        """
        if mirnas is None:
            mirnas = self._available_mirnas
        else:
            mirnas = self._validate_mirna_targets(mirnas)

        df = data_tpm.copy(deep=True).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if show_missing_report:
            missing = set(self.standard_mrna) - set(df.columns)
            if missing:
                print(f"Note: {len(missing)} standard genes absent from input (filled with 0).")

        df = df.reindex(columns=self.standard_mrna, fill_value=0.0)
        print(f"Stack prediction for {len(mirnas)} miRNAs...")
        predictor = self._get_predictor()
        return predictor.predict_many(df, mirnas)

    def predict_single_cell(
        self,
        data,
        mirnas=None,
        mapping_path=None,
        show_missing_report=False,
    ):
        """Single-cell inference (no KNN impute). Default: K1 cohort miRNAs."""
        if mirnas is None:
            mirnas = self.mirnas_for_cohort("K1")
        standardized = self.prepare_input(data, mapping_path=mapping_path)
        data_tpm = self.TPM(standardized, enforce_mrna_standard=False)
        return self.predict(data_tpm.T, mirnas=mirnas, show_missing_report=show_missing_report)

    def load_knn_reference(self, path=None):
        path = Path(path or DEFAULT_KNN_REF_PATH)
        if self._knn_ref is not None and self._knn_ref_path == path:
            return self._knn_ref

        if not path.is_file():
            raise FileNotFoundError(f"KNN reference not found: {path}")

        print(f"Loading KNN reference from {path}...")
        if path.suffix == ".parquet":
            ref = pd.read_parquet(path)
        else:
            ref = pd.read_csv(path, index_col=0)

        ref = _auto_orient_cells_by_genes(ref)
        ref = ref.reindex(columns=self.standard_mrna, fill_value=0.0)
        ref = ref.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        print(f"✔ KNN reference ready: {ref.shape[0]} cells × {ref.shape[1]} genes")

        self._knn_ref = ref
        self._knn_ref_path = path
        return ref

    def knn_impute_log_tpm(self, data_tpm, knn_ref_path=None, knn_k=5):
        """
        Impute zero entries in log2(TPM+1) matrix.

        data_tpm: genes × cells (output of TPM()).
        Returns genes × cells with zeros replaced by KNN donor means.
        """
        ref = self.load_knn_reference(path=knn_ref_path)
        k = min(knn_k, max(1, ref.shape[0] - 1))

        X_query = data_tpm.T.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_imputed = align_and_knn_impute(
            X_query=X_query,
            required_cols=self.standard_mrna,
            X_ref_knn=ref,
            n_neighbors=k,
        )
        return X_imputed.T.reindex(index=data_tpm.index, columns=data_tpm.columns)

    def predict_single_cell_knn_imputed(
        self,
        data,
        mirnas=None,
        mapping_path=None,
        knn_ref_path=None,
        knn_k=5,
        show_missing_report=False,
    ):
        """
        Single-cell inference with KNN imputation (recommended for K1 cohort).

        Pipeline: prepare_input → TPM/log2 → KNN impute → stack predict.
        Default miRNAs: K1 cohort from manifest.
        """
        if mirnas is None:
            mirnas = self.mirnas_for_cohort("K1")
        standardized = self.prepare_input(data, mapping_path=mapping_path)
        data_tpm = self.TPM(standardized, enforce_mrna_standard=False)
        data_tpm_imputed = self.knn_impute_log_tpm(
            data_tpm,
            knn_ref_path=knn_ref_path,
            knn_k=knn_k,
        )
        return self.predict(
            data_tpm_imputed.T,
            mirnas=mirnas,
            show_missing_report=show_missing_report,
        )

    def prepare_single_cell(self, data, mapping_path=None):
        standardized = self.prepare_input(data, mapping_path=mapping_path)
        data_tpm = self.TPM(standardized, enforce_mrna_standard=False)
        return data_tpm.T

    @staticmethod
    def _validate_pseudobulk_k(K):
        if K in ALLOWED_PSEUDOBULK_K:
            return
        allowed = ", ".join(str(k) for k in sorted(ALLOWED_PSEUDOBULK_K))
        raise ValueError(
            f"K must be one of {{{allowed}}}. For single-cell level use predict_single_cell(). "
            f"Got K={K}."
        )

    def _validate_mirna_targets(self, mirnas):
        requested = list(mirnas)
        if not requested:
            raise ValueError("mirnas must be a non-empty list of miRNA names.")
        unknown = sorted(set(requested) - set(self._available_mirnas))
        if unknown:
            raise ValueError(f"Unknown miRNAs (not in available_mirnas): {unknown}")
        return requested

    @staticmethod
    def _load_raw_input(data) -> pd.DataFrame:
        """Load raw counts from path (.csv / .parquet) or return a DataFrame copy."""
        if isinstance(data, pd.DataFrame):
            df = data.copy(deep=True)
        elif isinstance(data, (str, Path)):
            path = Path(data)
            if not path.is_file():
                raise FileNotFoundError(f"Input file not found: {path}")
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                df = pd.read_parquet(path)
            elif suffix == ".csv":
                df = pd.read_csv(path)
            else:
                raise ValueError(
                    f"Unsupported input format {suffix!r}. Expected .csv or .parquet."
                )
        else:
            raise TypeError(
                "data must be a pandas DataFrame or a path to .csv / .parquet"
            )

        if "barcode" in df.columns:
            df = df.set_index("barcode", drop=True)
        return df

    def predict_all(
        self,
        data,
        mapping_path=None,
        knn_ref_path=None,
        knn_k=5,
        celltype_col="CellType",
        n_hvg=2000,
        n_pca=30,
        show_missing_report=False,
    ) -> pd.DataFrame:
        """
        Full inference for all eligible miRNAs from raw counts.

        Routing (from target_config.json):
        - **K1** cohort → single-cell + KNN impute
        - **K2, K3, K4, K5, K10** cohorts → KNN pseudobulk (within CellType)

        Parameters
        ----------
        data
            Raw counts: pandas DataFrame, or path to ``.csv`` / ``.parquet``.
            Expected layout: cells × genes (ENSG columns) + optional ``CellType``.
            If a ``barcode`` column is present, it becomes the row index.

        Returns
        -------
        DataFrame
            cells × all eligible miRNAs (log2 scale), columns in manifest order.
        """
        raw = self._load_raw_input(data)
        n_cells = raw.shape[0]
        print(f"Full inference: {n_cells} cells, {len(self._available_mirnas)} eligible miRNAs")

        parts: list[pd.DataFrame] = []

        k1_mirnas = self.mirnas_for_cohort("K1")
        if k1_mirnas:
            print(f"  K1 single-cell + KNN impute: {len(k1_mirnas)} miRNAs")
            k1_pred = self.predict_single_cell_knn_imputed(
                raw,
                mirnas=k1_mirnas,
                mapping_path=mapping_path,
                knn_ref_path=knn_ref_path,
                knn_k=knn_k,
                show_missing_report=show_missing_report,
            )
            parts.append(k1_pred)

        for k in sorted(ALLOWED_PSEUDOBULK_K):
            cohort = f"K{k}"
            pb_mirnas = self.mirnas_for_cohort(cohort)
            if not pb_mirnas:
                continue
            print(f"  {cohort} KNN pseudobulk (K={k}): {len(pb_mirnas)} miRNAs")
            pb_pred = self.predict_knn_pseudobulk(
                raw,
                K=k,
                mirnas=pb_mirnas,
                celltype_col=celltype_col,
                mapping_path=mapping_path,
                n_hvg=n_hvg,
                n_pca=n_pca,
                show_missing_report=show_missing_report,
            )
            parts.append(pb_pred)

        if not parts:
            raise RuntimeError("No cohort predictions produced; check manifest.")

        combined = pd.concat(parts, axis=1)
        combined = combined.reindex(columns=self._available_mirnas)
        print(f"✔ Done: {combined.shape[0]} cells × {combined.shape[1]} miRNAs")
        return combined

    def _split_expression_and_celltype(self, data, celltype_col="CellType"):
        df = data.copy(deep=True)
        if celltype_col in df.columns:
            celltypes = df[celltype_col].copy()
            expression = df.drop(columns=[celltype_col])
            barcodes = expression.index.astype(str)
            celltypes.index = barcodes
            return expression, celltypes

        if celltype_col in df.index:
            celltype_value = df.loc[celltype_col]
            expression = df.drop(index=[celltype_col])
            barcodes = expression.columns.astype(str)
            celltypes = pd.Series(celltype_value.values, index=barcodes, name=celltype_col)
            return expression, celltypes

        raise ValueError(
            f"Column '{celltype_col}' not found. Expected '{celltype_col}' as a metadata column "
            "(typical format: barcodes x genes + CellType)."
        )

    @staticmethod
    def _log1p_cpm_for_knn(counts_gc):
        lib = counts_gc.sum(axis=0)
        lib = np.where(lib == 0, 1.0, lib)
        scaled = counts_gc / lib * 1e4
        return np.log1p(scaled)

    def _knn_neighbor_indices(self, counts_gc, K, n_hvg=2000, n_pca=30):
        try:
            from sklearn.decomposition import PCA
            from sklearn.neighbors import NearestNeighbors
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "scikit-learn is required for KNN pseudobulk (PCA + NearestNeighbors)."
            ) from exc

        n_cells = counts_gc.shape[1]
        if n_cells < K:
            raise ValueError(f"Need at least K={K} cells, got {n_cells}.")

        x = self._log1p_cpm_for_knn(counts_gc)
        variances = np.var(x, axis=1)
        n_hvg_eff = min(n_hvg, x.shape[0])
        top_idx = np.argpartition(variances, -n_hvg_eff)[-n_hvg_eff:]
        x_hvg = x[top_idx, :]

        n_components = max(1, min(n_pca, x_hvg.shape[0], n_cells))
        emb = PCA(n_components=n_components, random_state=0).fit_transform(x_hvg.T)

        nn = NearestNeighbors(n_neighbors=K, metric="euclidean")
        nn.fit(emb)
        _, indices = nn.kneighbors(emb, return_distance=True)
        return indices.astype(np.intp)

    @staticmethod
    def _sum_pseudobulk_counts(counts_gc, neighbor_indices):
        n_genes, n_cells = counts_gc.shape
        out = np.zeros((n_genes, n_cells), dtype=np.float64)
        for anchor in range(n_cells):
            out[:, anchor] = counts_gc[:, neighbor_indices[anchor]].sum(axis=1)
        return out

    def predict_knn_pseudobulk(
        self,
        data,
        K,
        mirnas=None,
        celltype_col="CellType",
        mapping_path=None,
        n_hvg=2000,
        n_pca=30,
        show_missing_report=False,
    ):
        """
        Per-cell KNN pseudobulk predictions within each CellType.

        For every anchor cell: pseudobulk = anchor + (K-1) nearest neighbors
        (Euclidean KNN on PCA of HVG log1p-CPM, built inside CellType only).
        Raw counts are summed, then TPM/log2 and stack prediction.
        Default miRNAs: cohort K{K} from manifest (no KNN impute on pseudobulk).
        """
        self._validate_pseudobulk_k(K)
        cohort = f"K{K}"
        if mirnas is None:
            mirnas = self.mirnas_for_cohort(cohort)
        else:
            mirnas = self._validate_mirna_targets(mirnas)
            wrong = sorted(set(mirnas) - set(self._cohorts[cohort]))
            if wrong:
                raise ValueError(
                    f"miRNAs {wrong[:5]} are not assigned to cohort {cohort}. "
                    f"Use mirnas_for_pseudobulk_k({K}) or check manifest."
                )

        expression, celltypes = self._split_expression_and_celltype(
            data, celltype_col=celltype_col
        )
        standardized = self.prepare_input(expression, mapping_path=mapping_path)
        barcodes = standardized.columns.astype(str)
        celltypes = celltypes.reindex(barcodes)

        missing_ct = celltypes.isna().sum()
        if missing_ct:
            warnings.warn(
                f"{missing_ct} barcodes have no {celltype_col}; their predictions will be NaN.",
                stacklevel=2,
            )

        preds = pd.DataFrame(index=barcodes, columns=mirnas, dtype=np.float64)
        preds[:] = np.nan

        ct_series = celltypes.dropna()
        for cell_type in ct_series.unique():
            type_barcodes = [
                bc for bc in ct_series.index[ct_series == cell_type]
                if bc in standardized.columns
            ]
            if not type_barcodes:
                continue

            n_cells = len(type_barcodes)
            if n_cells < K:
                warnings.warn(
                    f"CellType '{cell_type}' has {n_cells} cells < K={K}. "
                    "Skipping prediction (NaN).",
                    stacklevel=2,
                )
                continue

            counts = standardized[type_barcodes].values
            neighbor_idx = self._knn_neighbor_indices(
                counts, K=K, n_hvg=n_hvg, n_pca=n_pca
            )
            pseudobulk_counts = self._sum_pseudobulk_counts(counts, neighbor_idx)
            pseudobulk_df = pd.DataFrame(
                pseudobulk_counts,
                index=standardized.index,
                columns=type_barcodes,
            )

            pseudobulk_tpm = self.TPM(pseudobulk_df, enforce_mrna_standard=False)
            type_preds = self.predict(
                pseudobulk_tpm.T,
                mirnas=mirnas,
                show_missing_report=show_missing_report,
            )
            preds.loc[type_barcodes, mirnas] = type_preds.reindex(
                columns=mirnas
            ).values

        return preds

    def build_cluster_counts(
        self,
        raw_data,
        clusters_df,
        barcode_col="barcode",
        cluster_col="cluster",
        mapping_path=None
    ):
        standardized = self.prepare_input(raw_data, mapping_path=mapping_path)
        cluster_counts = pd.DataFrame(index=standardized.index)

        for cluster in clusters_df[cluster_col].unique():
            barcodes = clusters_df.loc[clusters_df[cluster_col] == cluster, barcode_col].tolist()
            available_barcodes = [bc for bc in barcodes if bc in standardized.columns]
            if not available_barcodes:
                continue
            cluster_counts[cluster] = standardized[available_barcodes].sum(axis=1)

        return cluster_counts

    def predict_pseudobulk(
        self,
        raw_data,
        clusters_df,
        barcode_col="barcode",
        cluster_col="cluster",
        mapping_path=None,
        expand_to_cells=True,
        show_missing_report=False
    ):
        cluster_counts = self.build_cluster_counts(
            raw_data=raw_data,
            clusters_df=clusters_df,
            barcode_col=barcode_col,
            cluster_col=cluster_col,
            mapping_path=mapping_path,
        )

        cluster_tpm = self.TPM(cluster_counts, enforce_mrna_standard=False)
        cluster_preds = self.predict(cluster_tpm.T, show_missing_report=show_missing_report)

        if not expand_to_cells:
            return cluster_preds

        expanded = []
        for cluster in clusters_df[cluster_col].unique():
            if cluster not in cluster_preds.index:
                continue
            cluster_series = cluster_preds.loc[cluster]
            barcodes = clusters_df.loc[clusters_df[cluster_col] == cluster, barcode_col].tolist()
            if not barcodes:
                continue
            expanded.append(pd.DataFrame({bc: cluster_series for bc in barcodes}))

        if not expanded:
            return pd.DataFrame(columns=cluster_preds.columns)
        return pd.concat(expanded, axis=1).T

    def run_workflow(
        self,
        path_data,
        path_ss=None,
        path_clusters=None,
        path_bulk=None,
        mapping_path=None,
        barcode_col="barcode",
        cluster_col="cluster",
        show_missing_report=False
    ):
        raw_data = pd.read_csv(path_data, index_col=0)
        ss_preds = self.predict_single_cell_knn_imputed(
            raw_data,
            mapping_path=mapping_path,
            show_missing_report=show_missing_report,
        )

        if path_ss is not None:
            ss_preds.to_csv(path_ss)

        pb_preds = None
        if path_clusters is not None:
            clusters_df = pd.read_csv(path_clusters)
            pb_preds = self.predict_pseudobulk(
                raw_data=raw_data,
                clusters_df=clusters_df,
                barcode_col=barcode_col,
                cluster_col=cluster_col,
                mapping_path=mapping_path,
                expand_to_cells=True,
                show_missing_report=show_missing_report,
            )
            if path_bulk is not None:
                pb_preds.to_csv(path_bulk)

        return ss_preds, pb_preds

   
