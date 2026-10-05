"""miRNA stack inference for single-cell RNA-seq.

Public entry point: ``miRPredictor`` with ``predict_csv`` / ``predict`` (AnnData),
plus ``get_mirs`` / ``get_features`` / ``get_metrics``.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from constants import (
        K1_REF_PATH,
        CONFIG_PATH,
        ROBUSTNESS_CONFIG_PATH,
        resolve_gene_mapping_path,
        parse_prediction_config,
    )
    from stack_predictor import StackPredictor
except ImportError:  # pragma: no cover
    _inf = Path(__file__).resolve().parent
    if str(_inf) not in sys.path:
        sys.path.insert(0, str(_inf))
    from constants import (
        K1_REF_PATH,
        CONFIG_PATH,
        ROBUSTNESS_CONFIG_PATH,
        resolve_gene_mapping_path,
        parse_prediction_config,
    )
    from stack_predictor import StackPredictor

ALLOWED_PSEUDOBULK_K = frozenset({2, 3, 4, 5, 10})
DEFAULT_KNN_REF_PATH = str(K1_REF_PATH)

_CELLTYPE_CANDIDATES = ("CellType", "celltype", "cell_type", "Celltype")
_CLUSTER_FALLBACK = ("clusters", "cluster", "leiden", "louvain")
_PCA_CANDIDATES = ("X_pca_harmony", "X_pca")

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        if iterable is None:
            class _Dummy:
                def update(self, n=1):
                    pass

                def close(self):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            return _Dummy()
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
        counts = np.sum(np.isfinite(neigh_vals), axis=1)
        sums = np.nansum(neigh_vals, axis=1)
        imputed = np.zeros(len(missing_idx), dtype=np.float32)
        np.divide(sums, counts, out=imputed, where=counts > 0)
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


def _looks_like_integer_counts(matrix, *, max_check: int = 50_000) -> bool:
    """True if matrix looks like non-negative integer raw counts."""
    try:
        import scipy.sparse as sp
    except ImportError:  # pragma: no cover
        sp = None

    if sp is not None and sp.issparse(matrix):
        data = np.asarray(matrix.data, dtype=np.float64)
        if data.size == 0:
            # All zeros still count as integer counts.
            return True
        if data.size > max_check:
            rng = np.random.default_rng(0)
            data = data[rng.choice(data.size, size=max_check, replace=False)]
    else:
        arr = np.asarray(matrix)
        if arr.size == 0:
            return True
        flat = arr.ravel()
        if flat.size > max_check:
            rng = np.random.default_rng(0)
            flat = flat[rng.choice(flat.size, size=max_check, replace=False)]
        data = flat.astype(np.float64, copy=False)

    if np.any(~np.isfinite(data)) or np.any(data < -1e-8):
        return False
    return bool(np.allclose(data, np.round(data), atol=1e-6))


def _to_dense_float_df(matrix, index, columns) -> pd.DataFrame:
    try:
        import scipy.sparse as sp
    except ImportError:  # pragma: no cover
        sp = None
    if sp is not None and sp.issparse(matrix):
        matrix = matrix.toarray()
    return pd.DataFrame(np.asarray(matrix, dtype=np.float64), index=index, columns=columns)


class miRPredictor:
    """End-to-end miRNA prediction from raw scRNA-seq counts (CSV or AnnData)."""

    def __init__(
        self,
        path_length="df_gene_mapping.parquet",
        path_mrna="mRNA_names.json",
        config_path=None,
        robustness_config_path=None,
        device="cuda",
        catboost_task="CPU",  # unused; kept for API compatibility
        preload_models=False,
        log=True,
    ):
        base = Path(__file__).resolve().parent
        self._base_dir = base
        self.log = log
        self._device = device
        self._catboost_task = catboost_task
        self._config_path = Path(config_path or CONFIG_PATH)
        self._robustness_config_path = Path(robustness_config_path or ROBUSTNESS_CONFIG_PATH)

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

        config = json.loads(self._config_path.read_text(encoding="utf-8"))
        eligible, cohorts, target_info = parse_prediction_config(config)
        self._cohorts: dict[str, list[str]] = cohorts
        self._available_mirnas: list[str] = eligible
        self._target_info: dict[str, dict] = target_info
        self._robustness_meta, self._robustness_targets = self._load_robustness_config(
            self._robustness_config_path
        )
        self._robustness_version = int(self._robustness_meta.get("version") or 1)
        self._minimal_features = {
            mir: list(info.get("minimal_features") or [])
            for mir, info in self._robustness_targets.items()
            if info.get("minimal_features")
        }
        self._sc_features = self._load_sc_features(self._base_dir.parent / "sc_features.json")

        self._knn_ref = None
        self._knn_ref_path = None

        if preload_models:
            self.load_models()

    # ------------------------------------------------------------------
    # Public metadata API
    # ------------------------------------------------------------------

    def get_mirs(self) -> list[str]:
        """Eligible miRNAs from ``prediction_config.json``."""
        return list(self._available_mirnas)

    def get_features(self, mirnas: list[str] | None = None) -> dict[str, list[str]]:
        """Model feature panels (ENSG) from ``prediction_config``."""
        targets = self._resolve_mirna_subset(mirnas)
        return {m: list(self._target_info[m]["features"]) for m in targets}

    def get_metrics(self, mirnas: list[str] | None = None) -> dict[str, dict]:
        """Held-out metrics: bulk/sc R² and MSE per miRNA."""
        targets = self._resolve_mirna_subset(mirnas)
        keys = ("test_bulk_r2", "test_sc_r2", "test_bulk_mse", "test_sc_mse")
        out: dict[str, dict] = {}
        for m in targets:
            info = self._target_info[m]
            out[m] = {k: info.get(k) for k in keys}
        return out

    def feature_diagnostics(
        self,
        adata,
        mirnas: list[str] | None = None,
        mapping_path=None,
        *,
        feature_set: str = "sc",
    ) -> pd.DataFrame:
        """Per-miRNA share of **SC** features absent in ``adata``.

        By default the panel is ``sc_features.json`` intersected with the
        trained model panel (bulk-only genes are excluded). With
        ``feature_set='minimal'``, uses legacy ``minimal_features`` gene lists
        (robustness config v1 only); for v3 configs this is the same as ``sc``.

        A feature counts as *absent* when it is missing from the measured count
        matrix or is present but all-zero across every cell (after ENSG mapping).

        Returns a table with ``n_sc_features``, ``n_missing``, ``n_all_zero``,
        ``n_absent``, and ``pct_absent`` (0–100). When robustness v3 limits are
        available, also ``max_absent_frac``, ``absent_frac``, and
        ``passes_robustness``.
        """
        targets = self._resolve_mirna_subset(mirnas)
        measured = self._measured_ensg_from_adata(adata, mapping_path=mapping_path)
        rows: list[dict] = []
        for mir in targets:
            if feature_set == "minimal":
                panel = list(self._minimal_features.get(mir) or [])
            elif feature_set in ("sc", "model"):
                panel = self._sc_panel_for_mir(mir)
            else:
                raise ValueError(
                    f"feature_set must be 'sc' or 'minimal', got {feature_set!r}"
                )
            stats = self._absent_feature_stats(measured, panel)
            stats["n_sc_features"] = stats.pop("n_features")
            row = {"mirna": mir, **stats}
            limit = self._max_absent_frac_for_mir(mir)
            if limit is not None and stats["n_sc_features"]:
                absent_frac = stats["pct_absent"] / 100.0
                row["max_absent_frac"] = limit
                row["absent_frac"] = absent_frac
                row["passes_robustness"] = absent_frac <= limit
            rows.append(row)
        return pd.DataFrame(rows).set_index("mirna", drop=False)

    @property
    def available_mirnas(self) -> list[str]:
        return self.get_mirs()

    @property
    def cohorts(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._cohorts.items()}

    @property
    def target_info(self) -> dict[str, dict]:
        return {k: dict(v) for k, v in self._target_info.items()}

    def mirnas_for_cohort(self, cohort: str) -> list[str]:
        if cohort not in self._cohorts:
            raise KeyError(f"Unknown cohort {cohort!r}. Expected one of {list(self._cohorts)}.")
        return list(self._cohorts[cohort])

    def mirnas_for_pseudobulk_k(self, K: int) -> list[str]:
        self._validate_pseudobulk_k(K)
        return self.mirnas_for_cohort(f"K{K}")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_predictor(self) -> StackPredictor:
        if self._predictor is None:
            self.load_models()
        return self._predictor

    def load_models(self) -> StackPredictor:
        if self._predictor is not None:
            return self._predictor
        self._predictor = StackPredictor(
            config_path=self._config_path,
            device=self._device,
            catboost_task=self._catboost_task,
            preload_all=False,
        )
        return self._predictor

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict_csv(
        self,
        data,
        mirnas: list[str] | None = None,
        mapping_path=None,
        knn_ref_path=None,
        knn_k=5,
        celltype_col: str | None = None,
        n_hvg=2000,
        n_pca=30,
        show_missing_report=False,
    ) -> pd.DataFrame:
        """Full inference from a raw-counts CSV / parquet / DataFrame."""
        raw = self._load_raw_input(data)
        expression, celltypes = self._split_expression_and_celltype(
            raw, celltype_col=celltype_col
        )
        return self._run_full_inference(
            expression=expression,
            celltypes=celltypes,
            mirnas=mirnas,
            mapping_path=mapping_path,
            knn_ref_path=knn_ref_path,
            knn_k=knn_k,
            n_hvg=n_hvg,
            n_pca=n_pca,
            pca_coords=None,
            show_missing_report=show_missing_report,
        )

    def predict(
        self,
        adata,
        mirnas: list[str] | None = None,
        mapping_path=None,
        knn_ref_path=None,
        knn_k=5,
        show_missing_report=False,
        inplace: bool = False,
    ):
        """Full inference from AnnData; returns AnnData with miRNA columns in ``.obs``.

        Adds one ``obs`` column per successfully predicted miRNA (log2 scale).
        Also writes ``obsm['X_mirna']`` (cells × predicted miRNAs) and
        ``uns['mirna_prediction']`` metadata (predicted / skipped lists).
        """
        expression, celltypes, pca_coords = self._extract_from_adata(adata)
        pred = self._run_full_inference(
            expression=expression,
            celltypes=celltypes,
            mirnas=mirnas,
            mapping_path=mapping_path,
            knn_ref_path=knn_ref_path,
            knn_k=knn_k,
            n_hvg=2000,
            n_pca=30,
            pca_coords=pca_coords,
            show_missing_report=show_missing_report,
        )
        return self._attach_predictions_to_adata(adata, pred, inplace=inplace)

    # ------------------------------------------------------------------
    # Core inference orchestration
    # ------------------------------------------------------------------

    def _run_full_inference(
        self,
        *,
        expression: pd.DataFrame,
        celltypes: pd.Series,
        mirnas: list[str] | None,
        mapping_path,
        knn_ref_path,
        knn_k: int,
        n_hvg: int,
        n_pca: int,
        pca_coords: pd.DataFrame | None,
        show_missing_report: bool,
    ) -> pd.DataFrame:
        requested = self._resolve_mirna_subset(mirnas)
        n_cells, n_genes = expression.shape
        print(
            f"Full inference: {n_cells} cells, {n_genes} genes, "
            f"{len(requested)} eligible miRNAs"
        )

        # Quiet third-party noise (NVML, sklearn version, empty-slice, …).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            standardized = self._prepare_input(expression, mapping_path=mapping_path)
            # Robustness gate on ENSG IDs from the *measured* matrix (before zero-fill).
            measured_ensg = self._expression_to_ensg_counts(
                expression, mapping_path=mapping_path
            )
            runnable, skipped, skip_detail = self._filter_by_robustness(
                measured_ensg, requested
            )

            n_req = len(requested)
            n_run = len(runnable)
            n_skip = len(skipped)
            skip_reason = (
                "exceed SC feature dropout tolerance (robustness_config)"
                if self._robustness_uses_fraction_gate()
                else "require minimal robustness features missing from the input data"
            )
            print(f"Predicting {n_run}/{n_req} miRNAs; {n_skip}/{n_req} {skip_reason}")

            if not runnable:
                empty = pd.DataFrame(
                    index=standardized.columns.astype(str), columns=requested, dtype=float
                )
                empty.attrs["skipped_mirnas"] = skipped
                empty.attrs["skip_detail"] = skip_detail
                empty.attrs["predicted_mirnas"] = []
                return empty

            barcodes = standardized.columns.astype(str)
            celltypes = celltypes.reindex(barcodes)
            preds = pd.DataFrame(index=barcodes, columns=requested, dtype=float)
            preds[:] = np.nan

            by_cohort: dict[str, list[str]] = {}
            for mir in runnable:
                cohort = str(self._target_info[mir]["assigned_cohort"])
                by_cohort.setdefault(cohort, []).append(mir)

            pbar = tqdm(total=len(runnable), desc="Predicting miRNAs", unit="mir")
            try:
                k1 = by_cohort.get("K1") or []
                if k1:
                    k1_mat = self._prepare_k1_imputed_cells_x_genes(
                        standardized,
                        knn_ref_path=knn_ref_path,
                        knn_k=knn_k,
                    )
                    k1_pred = self._predict_stack(
                        k1_mat,
                        mirnas=k1,
                        show_missing_report=show_missing_report,
                        on_mirna=lambda _m: pbar.update(1),
                    )
                    preds.loc[k1_pred.index, k1] = k1_pred[k1].to_numpy()

                for k in sorted(ALLOWED_PSEUDOBULK_K):
                    cohort = f"K{k}"
                    pb = by_cohort.get(cohort) or []
                    if not pb:
                        continue
                    pb_mat = self._prepare_knn_pseudobulk_cells_x_genes(
                        standardized=standardized,
                        celltypes=celltypes,
                        K=k,
                        n_hvg=n_hvg,
                        n_pca=n_pca,
                        pca_coords=pca_coords,
                    )
                    pb_pred = self._predict_stack(
                        pb_mat,
                        mirnas=pb,
                        show_missing_report=show_missing_report,
                        on_mirna=lambda _m: pbar.update(1),
                    )
                    preds.loc[pb_pred.index, pb] = pb_pred[pb].to_numpy()
            finally:
                pbar.close()

            preds.attrs["predicted_mirnas"] = list(runnable)
            preds.attrs["skipped_mirnas"] = list(skipped)
            preds.attrs["skip_detail"] = skip_detail
            return preds

    @staticmethod
    def _normalize_ensg(gene_id: str) -> str:
        """Strip optional Ensembl version suffix (ENSG....N)."""
        g = str(gene_id)
        if g.startswith("ENSG") and "." in g:
            return g.split(".", 1)[0]
        return g

    def _expression_to_ensg_counts(
        self, expression: pd.DataFrame, mapping_path=None
    ) -> pd.DataFrame:
        """cells×genes → genes×cells with ENSG index; no zero-fill of missing genes."""
        df = expression.copy(deep=True)
        # drop non-expression metadata if still present
        for col in ("barcode", "Barcode", "cell_id"):
            if col in df.columns:
                df = df.drop(columns=[col])
        gene_axis = self._detect_gene_axis(df)
        if gene_axis == "columns":
            df = df.T
        mapping_path = resolve_gene_mapping_path(mapping_path)
        if self._needs_symbol_to_ens_mapping(df.index):
            df = self._replace_genes_names(df, mapping_path=mapping_path)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        df = df.loc[~df.index.duplicated(keep="first")]
        df.index = [self._normalize_ensg(g) for g in df.index]
        # if normalize created dupes, sum
        if pd.Index(df.index).duplicated().any():
            df = df.groupby(level=0).sum()
        return df

    @staticmethod
    def _format_skip_warning(skip_detail: dict[str, dict], skipped: list[str], *, max_mirs: int = 8) -> str:
        lines = []
        for mir in skipped[:max_mirs]:
            info = skip_detail.get(mir) or {}
            reason = info.get("reason")
            if reason in ("exceeds_max_absent_frac", "pending_robustness", "no_robustness_limit"):
                pct = info.get("pct_absent")
                max_af = info.get("max_absent_frac")
                if pct is not None and max_af is not None:
                    lines.append(
                        f"  - {mir}: absent {pct:.1f}% of SC panel "
                        f"(limit {100.0 * max_af:.1f}% absent)"
                    )
                else:
                    lines.append(f"  - {mir}: {reason}")
                continue
            missing = info.get("missing") or []
            allzero = info.get("all_zero") or []
            bits = []
            if missing:
                bits.append(f"missing {len(missing)} ENSG (e.g. {missing[:3]})")
            if allzero:
                bits.append(f"all-zero {len(allzero)} ENSG (e.g. {allzero[:3]})")
            if reason:
                bits.append(str(reason))
            lines.append(f"  - {mir}: " + ("; ".join(bits) if bits else "robustness gate"))
        if len(skipped) > max_mirs:
            lines.append(f"  … and {len(skipped) - max_mirs} more")
        return "\n".join(lines)

    def _attach_predictions_to_adata(self, adata, pred: pd.DataFrame, *, inplace: bool = False):
        try:
            import anndata as ad  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "anndata is required for miRPredictor.predict(). "
                "Install it or use predict_csv() instead."
            ) from exc

        out = adata if inplace else adata.copy()
        obs_index = pd.Index(out.obs_names.astype(str))
        aligned = pred.reindex(obs_index)
        predicted = [c for c in aligned.columns if aligned[c].notna().any()]
 
        if predicted:
            out.obs[predicted] = aligned[predicted].to_numpy()
            out.obsm["X_mirna"] = aligned[predicted].to_numpy(dtype=np.float32)
            out.uns["mirna_prediction"] = {
                "predicted_mirnas": predicted,
                "skipped_mirnas": list(pred.attrs.get("skipped_mirnas") or []),
                "columns": predicted,
            }
        else:
            out.uns["mirna_prediction"] = {
                "predicted_mirnas": [],
                "skipped_mirnas": list(pred.attrs.get("skipped_mirnas") or []),
                "columns": [],
            }
        return out

    def _prepare_k1_imputed_cells_x_genes(
        self,
        standardized_genes_x_cells: pd.DataFrame,
        *,
        knn_ref_path=None,
        knn_k: int = 5,
    ) -> pd.DataFrame:
        data_tpm = self._tpm(standardized_genes_x_cells, enforce_mrna_standard=False)
        data_tpm_imputed = self._knn_impute_log_tpm(
            data_tpm, knn_ref_path=knn_ref_path, knn_k=knn_k
        )
        return data_tpm_imputed.T

    def _prepare_knn_pseudobulk_cells_x_genes(
        self,
        *,
        standardized: pd.DataFrame,
        celltypes: pd.Series,
        K: int,
        n_hvg: int,
        n_pca: int,
        pca_coords: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Return cells × genes log2(TPM+1) of within-cluster KNN pseudobulks."""
        barcodes = standardized.columns.astype(str)
        out_counts = pd.DataFrame(
            np.nan,
            index=standardized.index,
            columns=barcodes,
            dtype=np.float64,
        )

        ct_series = celltypes.dropna()
        # Barcodes without a cluster label stay NaN in the output (no warning spam).

        for cell_type in ct_series.unique():
            type_barcodes = [
                bc for bc in ct_series.index[ct_series == cell_type].astype(str)
                if bc in standardized.columns
            ]
            if not type_barcodes:
                continue
            n_cells = len(type_barcodes)
            if n_cells < K:
                continue

            counts = standardized[type_barcodes].values
            if pca_coords is not None:
                emb = pca_coords.reindex(type_barcodes)
                if emb.isna().any().any():
                    raise ValueError(
                        f"PCA coordinates missing for some cells in '{cell_type}'."
                    )
                neighbor_idx = self._knn_neighbor_indices_from_embedding(
                    emb.to_numpy(dtype=np.float32), K=K
                )
            else:
                neighbor_idx = self._knn_neighbor_indices(
                    counts, K=K, n_hvg=n_hvg, n_pca=n_pca
                )
            pb = self._sum_pseudobulk_counts(counts, neighbor_idx)
            out_counts.loc[:, type_barcodes] = pb

        # TPM only on columns that received pseudobulk counts
        valid_cols = [c for c in out_counts.columns if out_counts[c].notna().any()]
        if not valid_cols:
            return pd.DataFrame(
                np.nan,
                index=barcodes,
                columns=self.standard_mrna,
                dtype=float,
            )
        tpm = self._tpm(out_counts[valid_cols].fillna(0.0), enforce_mrna_standard=False)
        cells_x_genes = tpm.T
        # Reindex to all barcodes (NaN rows for skipped)
        return cells_x_genes.reindex(barcodes)

    # ------------------------------------------------------------------
    # Robustness gating
    # ------------------------------------------------------------------

    @staticmethod
    def _load_sc_features(path: Path) -> dict[str, list[str]]:
        if not path.is_file():
            raise FileNotFoundError(
                f"sc_features.json not found: {path}. "
                "Expected final_train_test_inference/sc_features.json."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): list(v) for k, v in raw.items()}

    def _sc_panel_for_mir(self, mir: str) -> list[str]:
        """SC genes from ``sc_features.json`` that are in this miRNA's model panel."""
        model_feats = list(self._target_info[mir]["features"])
        sc_set = set(self._sc_features.get(mir) or [])
        if not sc_set:
            raise KeyError(f"{mir!r} missing from sc_features.json")
        return [g for g in model_feats if g in sc_set]

    @staticmethod
    def _load_robustness_config(path: Path) -> tuple[dict, dict[str, dict]]:
        if not path.is_file():
            raise FileNotFoundError(
                f"robustness_config.json not found: {path}. "
                "Expected ../robustness/results/robustness_config.json "
                "or inference/robustness_config.json."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        targets = {str(k): dict(v or {}) for k, v in (raw.get("targets") or {}).items()}
        meta = {k: v for k, v in raw.items() if k != "targets"}
        return meta, targets

    def _robustness_uses_fraction_gate(self) -> bool:
        if self._robustness_version >= 3:
            return True
        return any(
            info.get("max_absent_frac") is not None
            for info in self._robustness_targets.values()
        )

    def _max_absent_frac_for_mir(self, mir: str) -> float | None:
        info = self._robustness_targets.get(mir) or {}
        if info.get("status") == "pending":
            return None
        val = info.get("max_absent_frac")
        if val is None:
            return None
        return float(val)

    def _measured_ensg_from_adata(self, adata, mapping_path=None) -> pd.DataFrame:
        """Raw counts from AnnData → genes×cells ENSG matrix (no zero-fill)."""
        counts_matrix, gene_labels, source = self._resolve_adata_counts(adata)
        obs_names = np.asarray(adata.obs_names).astype(str)
        if source == "raw":
            var = adata.raw.var
        else:
            var = adata.var
        if "gene_ids" in var.columns:
            gene_labels = var["gene_ids"].astype(str).tolist()
        expression = _to_dense_float_df(
            counts_matrix, index=obs_names, columns=gene_labels
        )
        if expression.columns.duplicated().any():
            expression = expression.T.groupby(level=0).sum().T
        return self._expression_to_ensg_counts(expression, mapping_path=mapping_path)

    @staticmethod
    def _absent_feature_stats(
        measured_ensg_genes_x_cells: pd.DataFrame,
        features: list[str],
    ) -> dict:
        """Count missing / all-zero genes in a feature panel vs measured matrix."""
        mat = measured_ensg_genes_x_cells
        panel = [miRPredictor._normalize_ensg(g) for g in features]
        n_features = len(panel)
        if n_features == 0:
            return {
                "n_features": 0,
                "n_missing": 0,
                "n_all_zero": 0,
                "n_present_nonzero": 0,
                "n_absent": 0,
                "pct_absent": float("nan"),
            }
        present = set(map(miRPredictor._normalize_ensg, mat.index.astype(str)))
        nz_series = (mat != 0).any(axis=1)
        nonzero = {
            miRPredictor._normalize_ensg(g)
            for g, flag in nz_series.items()
            if bool(flag)
        }
        missing = [g for g in panel if g not in present]
        all_zero = [g for g in panel if g in present and g not in nonzero]
        n_missing = len(missing)
        n_all_zero = len(all_zero)
        n_absent = n_missing + n_all_zero
        n_present_nonzero = n_features - n_absent
        pct_absent = 100.0 * n_absent / n_features
        return {
            "n_features": n_features,
            "n_missing": n_missing,
            "n_all_zero": n_all_zero,
            "n_present_nonzero": n_present_nonzero,
            "n_absent": n_absent,
            "pct_absent": pct_absent,
        }

    def _filter_by_robustness(
        self,
        measured_ensg_genes_x_cells: pd.DataFrame,
        mirnas: list[str],
    ) -> tuple[list[str], list[str], dict[str, dict]]:
        """Gate miRNAs using robustness config (v3 fraction or legacy gene list).

        ``measured_ensg_genes_x_cells`` must be genes × cells with ENSG index from the
        input assay (symbol→ENSG mapped if needed), *without* zero-filling absent genes.

        v3+: keep miRNA when absent fraction of its SC feature panel is
        ``<= max_absent_frac`` from ``robustness_config.json``.
        Legacy v1: every gene in ``minimal_features`` must be present and non-zero.
        """
        if self._robustness_uses_fraction_gate():
            return self._filter_by_robustness_fraction(measured_ensg_genes_x_cells, mirnas)
        return self._filter_by_robustness_minimal_list(measured_ensg_genes_x_cells, mirnas)

    def _filter_by_robustness_fraction(
        self,
        measured_ensg_genes_x_cells: pd.DataFrame,
        mirnas: list[str],
    ) -> tuple[list[str], list[str], dict[str, dict]]:
        mat = measured_ensg_genes_x_cells
        ok: list[str] = []
        skipped: list[str] = []
        detail: dict[str, dict] = {}
        for mir in mirnas:
            info = self._robustness_targets.get(mir) or {}
            if info.get("status") == "pending":
                skipped.append(mir)
                detail[mir] = {"reason": "pending_robustness"}
                continue
            max_af = self._max_absent_frac_for_mir(mir)
            if max_af is None:
                skipped.append(mir)
                detail[mir] = {"reason": "no_robustness_limit"}
                continue
            panel = self._sc_panel_for_mir(mir)
            stats = self._absent_feature_stats(mat, panel)
            n_feat = stats["n_features"]
            if n_feat == 0:
                skipped.append(mir)
                detail[mir] = {
                    "reason": "empty_sc_panel",
                    "max_absent_frac": max_af,
                    "pct_absent": float("nan"),
                }
                continue
            absent_frac = stats["n_absent"] / n_feat
            pct_absent = stats["pct_absent"]
            if absent_frac > max_af:
                skipped.append(mir)
                detail[mir] = {
                    "reason": "exceeds_max_absent_frac",
                    "max_absent_frac": max_af,
                    "absent_frac": absent_frac,
                    "pct_absent": pct_absent,
                    "n_absent": stats["n_absent"],
                    "n_sc_features": n_feat,
                    "n_missing": stats["n_missing"],
                    "n_all_zero": stats["n_all_zero"],
                }
            else:
                ok.append(mir)
        return ok, skipped, detail

    def _filter_by_robustness_minimal_list(
        self,
        measured_ensg_genes_x_cells: pd.DataFrame,
        mirnas: list[str],
    ) -> tuple[list[str], list[str], dict[str, dict]]:
        mat = measured_ensg_genes_x_cells
        present = set(map(self._normalize_ensg, mat.index.astype(str)))
        nz_series = (mat != 0).any(axis=1)
        nonzero = {self._normalize_ensg(g) for g, flag in nz_series.items() if bool(flag)}

        ok: list[str] = []
        skipped: list[str] = []
        detail: dict[str, dict] = {}
        for mir in mirnas:
            minimal_raw = self._minimal_features.get(mir) or []
            minimal = [self._normalize_ensg(g) for g in minimal_raw]
            if not minimal:
                skipped.append(mir)
                detail[mir] = {"missing": [], "all_zero": [], "reason": "no_minimal_features"}
                continue
            missing = [g for g in minimal if g not in present]
            all_zero = [g for g in minimal if g in present and g not in nonzero]
            if missing or all_zero:
                skipped.append(mir)
                detail[mir] = {"missing": missing, "all_zero": all_zero}
            else:
                ok.append(mir)
        return ok, skipped, detail

    def _resolve_mirna_subset(self, mirnas: list[str] | None) -> list[str]:
        if mirnas is None:
            return list(self._available_mirnas)
        return self._validate_mirna_targets(mirnas)

    # ------------------------------------------------------------------
    # Private preprocessing (TPM / prepare_input / mapping)
    # ------------------------------------------------------------------

    def _detect_gene_axis(self, data):
        genes = pd.read_csv(resolve_gene_mapping_path())
        gene_ens = set(genes["feature_id"].tolist())
        gene_names = set(genes["feature_name"].tolist())
        all_gene_names = gene_ens | gene_names
        index_hits = len(set(data.index) & all_gene_names)
        col_hits = len(set(data.columns) & all_gene_names)
        if index_hits == 0 and col_hits == 0:
            raise ValueError("Genes were not found neither in columns nor in index")
        return "index" if index_hits > col_hits else "columns"

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

    def _standardize_mrna(self, data):
        df = data.copy(deep=True)
        gene_axis = self._detect_gene_axis(df)
        if gene_axis == "columns":
            gene_cols = [col for col in df.columns if col in self.standard_mrna_set]
            df = df[gene_cols].T
        else:
            gene_rows = [idx for idx in df.index if idx in self.standard_mrna_set]
            df = df.loc[gene_rows]
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        df = df.loc[~df.index.duplicated(keep="first")]
        missing_genes = list(self.standard_mrna_set - set(df.index))
        if missing_genes:
            missing_df = pd.DataFrame(0.0, index=missing_genes, columns=df.columns)
            df = pd.concat([df, missing_df], axis=0)
        return df.reindex(self.standard_mrna, axis=0, fill_value=0.0)

    def _prepare_input(self, data, mapping_path=None):
        df = data.copy(deep=True)
        gene_axis = self._detect_gene_axis(df)
        if gene_axis == "columns":
            df = df.T
        mapping_path = resolve_gene_mapping_path(mapping_path)
        if self._needs_symbol_to_ens_mapping(df.index):
            df = self._replace_genes_names(df, mapping_path=mapping_path)
        return self._standardize_mrna(df)

    def _tpm(self, data, log=True, enforce_mrna_standard=True):
        df = data.copy(deep=True)
        if enforce_mrna_standard:
            df = self._standardize_mrna(df)
        else:
            df = df.loc[~df.index.duplicated(keep="first")]

        share = sorted(list(set(df.index) & set(self.gene_lengths["gene_id"])))
        if len(share) == 0:
            raise ValueError("Could not find gene lengths for any genes in the matrix.")

        df_filtered = df.loc[share].copy()
        gene_lengths_filtered = self.gene_lengths[
            self.gene_lengths["gene_id"].isin(share)
        ].set_index("gene_id")["gene_length_kb"]

        rpk = df_filtered.div(gene_lengths_filtered, axis=0)
        library_sizes = rpk.sum(axis=0)
        tpm = rpk.div(library_sizes, axis=1) * 1e6
        tpm = tpm.fillna(0.0)
        if log:
            return np.log2(tpm + 1)
        return tpm

    def _replace_genes_names(self, data, mapping_path=None):
        mapping_path = resolve_gene_mapping_path(mapping_path)
        mapping = pd.read_csv(mapping_path)
        mapping = mapping.dropna(subset=["feature_name", "feature_id"])
        mapping = mapping.drop_duplicates(subset=["feature_name"])
        ens_map = dict(zip(mapping["feature_name"], mapping["feature_id"]))
        df = data.copy(deep=True)
        df["ensembl_id"] = df.index.map(ens_map)
        df_clean = df.dropna(subset=["ensembl_id"]).copy()
        df_clean = df_clean[~df_clean["ensembl_id"].duplicated()]
        df_clean.index = df_clean.pop("ensembl_id")
        return df_clean.sort_index()

    def _predict_stack(
        self,
        data_tpm_cells_x_genes: pd.DataFrame,
        mirnas=None,
        show_missing_report=False,
        on_mirna=None,
    ) -> pd.DataFrame:
        """Stack inference on log2(TPM+1) matrix (cells × ENSG)."""
        if mirnas is None:
            mirnas = self._available_mirnas
        else:
            mirnas = self._validate_mirna_targets(mirnas)

        df = data_tpm_cells_x_genes.copy(deep=True)
        # Drop all-NaN rows (cells skipped by pseudobulk) before numeric fill
        valid_mask = df.notna().any(axis=1)
        df_valid = df.loc[valid_mask].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out = pd.DataFrame(np.nan, index=df.index, columns=list(mirnas), dtype=float)
        if df_valid.shape[0] == 0:
            if on_mirna is not None:
                for m in mirnas:
                    on_mirna(m)
            return out
        if show_missing_report:
            missing = set(self.standard_mrna) - set(df_valid.columns)
            if missing:
                print(f"Note: {len(missing)} standard genes absent from input (filled with 0).")
        df_valid = df_valid.reindex(columns=self.standard_mrna, fill_value=0.0)
        predictor = self._get_predictor()
        pred_valid = predictor.predict_many(df_valid, mirnas, on_mirna=on_mirna)
        out.loc[pred_valid.index, pred_valid.columns] = pred_valid.to_numpy()
        return out

    # ------------------------------------------------------------------
    # KNN impute (K1)
    # ------------------------------------------------------------------

    def load_knn_reference(self, path=None):
        path = Path(path or DEFAULT_KNN_REF_PATH)
        if self._knn_ref is not None and self._knn_ref_path == path:
            return self._knn_ref
        if not path.is_file():
            raise FileNotFoundError(f"KNN reference not found: {path}")
        if path.suffix == ".parquet":
            ref = pd.read_parquet(path)
        else:
            ref = pd.read_csv(path, index_col=0)
        ref = _auto_orient_cells_by_genes(ref)
        ref = ref.reindex(columns=self.standard_mrna, fill_value=0.0)
        ref = ref.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self._knn_ref = ref
        self._knn_ref_path = path
        return ref

    def _knn_impute_log_tpm(self, data_tpm, knn_ref_path=None, knn_k=5):
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

    # ------------------------------------------------------------------
    # Input loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_pseudobulk_k(K):
        if K in ALLOWED_PSEUDOBULK_K:
            return
        allowed = ", ".join(str(k) for k in sorted(ALLOWED_PSEUDOBULK_K))
        raise ValueError(f"K must be one of {{{allowed}}}. Got K={K}.")

    def _validate_mirna_targets(self, mirnas):
        requested = list(mirnas)
        if not requested:
            raise ValueError("mirnas must be a non-empty list of miRNA names.")
        unknown = sorted(set(requested) - set(self._available_mirnas))
        if unknown:
            raise ValueError(f"Unknown miRNAs (not in eligible list): {unknown}")
        return requested

    @staticmethod
    def _load_raw_input(data) -> pd.DataFrame:
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
            raise TypeError("data must be a pandas DataFrame or a path to .csv / .parquet")
        if "barcode" in df.columns:
            df = df.set_index("barcode", drop=True)
        return df

    def _split_expression_and_celltype(
        self,
        data: pd.DataFrame,
        celltype_col: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        df = data.copy(deep=True)
        col = celltype_col or self._find_celltype_column(df.columns)
        if col is None:
            raise ValueError(
                "No CellType / cell_type / clusters column found. "
                "CSV input must include a cell-type or cluster annotation for "
                "KNN pseudobulk cohorts (K2–K10)."
            )
        if col in df.columns:
            celltypes = df[col].copy()
            expression = df.drop(columns=[col])
            barcodes = expression.index.astype(str)
            celltypes.index = barcodes
            celltypes.name = col
            return expression, celltypes
        raise ValueError(f"Column '{col}' not found in input.")

    @staticmethod
    def _find_celltype_column(columns) -> str | None:
        cols = list(columns)
        for name in _CELLTYPE_CANDIDATES:
            if name in cols:
                return name
        for name in _CLUSTER_FALLBACK:
            if name in cols:
                return name
        # case-insensitive
        lower = {str(c).lower(): c for c in cols}
        for name in [n.lower() for n in _CELLTYPE_CANDIDATES + _CLUSTER_FALLBACK]:
            if name in lower:
                return lower[name]
        return None

    def _extract_from_adata(self, adata):
        """Return (cells×genes counts DF, celltype Series, PCA DataFrame)."""
        counts_matrix, gene_labels, source = self._resolve_adata_counts(adata)
        obs_names = np.asarray(adata.obs_names).astype(str)

        # Prefer ENSG from var['gene_ids'] when labels are not already ENSG
        if source == "raw":
            var = adata.raw.var
        else:
            var = adata.var
        if "gene_ids" in var.columns:
            gene_ids = var["gene_ids"].astype(str).tolist()
            gene_labels = gene_ids

        expression = _to_dense_float_df(counts_matrix, index=obs_names, columns=gene_labels)
        if expression.columns.duplicated().any():
            expression = expression.T.groupby(level=0).sum().T

        celltypes = self._resolve_adata_clusters(adata)
        pca_coords = self._resolve_adata_pca(adata)
        return expression, celltypes, pca_coords

    def _resolve_adata_counts(self, adata):
        candidates = []
        # User-specified search order: X → raw → layers['counts']
        candidates.append(("adata.X", adata.X, "X"))
        if getattr(adata, "raw", None) is not None and adata.raw is not None:
            candidates.append(("adata.raw.X", adata.raw.X, "raw"))
        layers = getattr(adata, "layers", None) or {}
        if "counts" in layers:
            candidates.append(("adata.layers['counts']", layers["counts"], "layers"))

        for label, matrix, source in candidates:
            if matrix is None:
                continue
            if _looks_like_integer_counts(matrix):
                return matrix, self._gene_labels_for_source(adata, source), source

        raise ValueError(
            "Raw counts not found: none of adata.X, adata.raw.X, or "
            "adata.layers['counts'] contain non-negative integer count data."
        )

    @staticmethod
    def _gene_labels_for_source(adata, source: str):
        if source == "raw":
            return np.asarray(adata.raw.var_names).astype(str)
        return np.asarray(adata.var_names).astype(str)

    def _resolve_adata_clusters(self, adata) -> pd.Series:
        cols = list(adata.obs.columns)
        for name in _CELLTYPE_CANDIDATES:
            if name in cols:
                s = adata.obs[name].copy()
                s.index = np.asarray(adata.obs_names).astype(str)
                s.name = name
                return s
        lower = {str(c).lower(): c for c in cols}
        for name in [n.lower() for n in _CELLTYPE_CANDIDATES]:
            if name in lower:
                key = lower[name]
                s = adata.obs[key].copy()
                s.index = np.asarray(adata.obs_names).astype(str)
                s.name = key
                return s
        for name in _CLUSTER_FALLBACK:
            if name in cols:
                s = adata.obs[name].copy()
                s.index = np.asarray(adata.obs_names).astype(str)
                s.name = name
                return s
        raise ValueError(
            "No cell-type / cluster annotation found in adata.obs. "
            "Expected one of CellType / celltype / cell_type, or clusters. "
            "Please run clustering (and preferably cell-type annotation) first."
        )

    def _resolve_adata_pca(self, adata) -> pd.DataFrame:
        obsm = getattr(adata, "obsm", {})
        for key in _PCA_CANDIDATES:
            if key in obsm:
                arr = np.asarray(obsm[key], dtype=np.float32)
                return pd.DataFrame(
                    arr,
                    index=np.asarray(adata.obs_names).astype(str),
                    columns=[f"PC{i+1}" for i in range(arr.shape[1])],
                )
        raise ValueError(
            "PCA coordinates not found in adata.obsm. "
            "Expected 'X_pca_harmony' (preferred for integrated data) or 'X_pca'."
        )

    # ------------------------------------------------------------------
    # KNN pseudobulk helpers
    # ------------------------------------------------------------------

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
        return self._knn_neighbor_indices_from_embedding(emb, K=K)

    @staticmethod
    def _knn_neighbor_indices_from_embedding(emb: np.ndarray, K: int) -> np.ndarray:
        try:
            from sklearn.neighbors import NearestNeighbors
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "scikit-learn is required for KNN pseudobulk (NearestNeighbors)."
            ) from exc
        n_cells = emb.shape[0]
        if n_cells < K:
            raise ValueError(f"Need at least K={K} cells, got {n_cells}.")
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
