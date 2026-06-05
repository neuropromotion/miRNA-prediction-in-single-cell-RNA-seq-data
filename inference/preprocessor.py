import json
import os
import sys
import contextlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED_PSEUDOBULK_K = frozenset({2, 3, 4, 5, 10})
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
            
class SingleCell():
    def __init__(
        self,
        path_length='df_gene_mapping.parquet',
        path_features='features.json',
        path_mrna='mRNA_names.json',
        path_models='models',
        log=True
    ):
        self.gene_lengths = pd.read_parquet(path_length)
        self.log = log
        self.path_models = Path(path_models)
        self.models = None

        with open(path_features, "r", encoding="utf-8") as f:
            features_payload = json.load(f)

        # New features format:
        # {"features": {mir: [...]}, "stages": {"stage_1": [...], ...}, ...}
        self.features_by_mir = features_payload["features"]
        raw_stages = features_payload["stages"]
        self.stage_order = sorted(raw_stages.keys(), key=lambda s: int(s.split("_")[1]))
        self.stages = {stage: raw_stages[stage] for stage in self.stage_order}

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

        mapping_path = mapping_path or "/mnt/jack-5/amismailov/ensembl_gene_mapping.csv"
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
        mapping_path="/mnt/jack-5/amismailov/ensembl_gene_mapping.csv"
    ):
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
    
    def _predict_with_model(self, model, dataset):
        if model["type"] == "xgb_booster":
            try:
                import xgboost as xgb
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "xgboost is required to load and run JSON models."
                ) from exc

            dmatrix = xgb.DMatrix(dataset.values, feature_names=list(dataset.columns))
            return model["model"].predict(dmatrix)

        return model["model"].predict(dataset)

    def predict(self, data_tpm, models=None, show_missing_report=False):
        if models is None:
            if self.models is None:
                self.models = self.load_models()
            models = self.models
            
        df = data_tpm.copy(deep=True).apply(pd.to_numeric, errors="coerce").fillna(0.0)

        all_stage_preds = []
        print('Prediction of miRNAs expression (cascade by stages)..')
        for stage in self.stage_order:
            stage_preds = pd.DataFrame(index=df.index)
            for mir in tqdm(self.stages[stage], desc=stage):
                model = models[stage][mir]
                current_features = self.features_by_mir[mir]

                missing_features = list(set(current_features) - set(df.columns))
                current_dataset = df.reindex(columns=current_features, fill_value=0.0).copy()

                if show_missing_report and missing_features:
                    print(f'For miR {mir} filled {len(missing_features)} missing features with zeroes.')

                stage_preds[mir] = self._predict_with_model(model, current_dataset)

            all_stage_preds.append(stage_preds)
            # Cascade: predictions from current stage become features for next stages.
            df = pd.concat([df, stage_preds], axis=1)

        return pd.concat(all_stage_preds, axis=1)

    def load_models(self, path=None):
        if path is None:
            path = self.path_models
        else:
            path = Path(path)

        try:
            import xgboost as xgb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "xgboost is required to load JSON models from stage folders."
            ) from exc

        models = {}
        for stage in self.stage_order:
            stage_dir = path / stage
            stage_models = {}
            for mir in self.stages[stage]:
                model_path = stage_dir / f"{mir}.json"
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")

                booster = xgb.Booster()
                booster.load_model(str(model_path))
                stage_models[mir] = {
                    "type": "xgb_booster",
                    "model": booster
                }
            models[stage] = stage_models

        self.models = models
        return models

    def predict_single_cell(self, data, mapping_path=None, show_missing_report=False):
        standardized = self.prepare_input(data, mapping_path=mapping_path)
        data_tpm = self.TPM(standardized, enforce_mrna_standard=False)
        return self.predict(data_tpm.T, show_missing_report=show_missing_report)

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
        all_mirs = set()
        for stage in self.stage_order:
            all_mirs.update(self.stages[stage])
        requested = list(mirnas)
        if not requested:
            raise ValueError("mirnas must be a non-empty list of miRNA names.")
        unknown = sorted(set(requested) - all_mirs)
        if unknown:
            raise ValueError(f"Unknown miRNAs (not in model stages): {unknown}")
        return requested

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
        mirnas,
        K,
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
        Raw counts are summed, then TPM/log2 and full cascade prediction.
        Cell types with fewer than K cells get NaN for all their barcodes.
        """
        self._validate_pseudobulk_k(K)
        mirnas = self._validate_mirna_targets(mirnas)

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
        ss_preds = self.predict_single_cell(
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

   