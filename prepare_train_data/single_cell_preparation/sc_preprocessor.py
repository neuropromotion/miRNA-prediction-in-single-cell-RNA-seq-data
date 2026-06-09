import os
import warnings

import numpy as np
import pandas as pd


class Pseudobulk_Sampler:
    def __init__(self, mrna_path, mirna_path, barcodes_path, gene_lengths_path,
                 selected_rna_path=None, selected_mirs=None, seed=None, log=False):
        self.log = log
        self.rng = np.random.default_rng(seed=seed)

        print("Loading data...")
        self.barcodes = pd.read_csv(barcodes_path, index_col=0)
        rna_raw = pd.read_csv(mrna_path, index_col=0)
        mir_raw = pd.read_csv(mirna_path, index_col=0)
        gene_lengths = pd.read_parquet(gene_lengths_path).set_index('gene_id')

        if selected_rna_path and os.path.isfile(selected_rna_path):
            with open(selected_rna_path, 'r') as f:
                self.selected_rna = [line.strip() for line in f if line.strip()]
        else:
            self.selected_rna = selected_rna_path

        self.selected_mirs = selected_mirs

        shared_genes = sorted(list(set(rna_raw.index) & set(gene_lengths.index)))
        self.rna = rna_raw.loc[shared_genes]
        self.gene_lengths_kb = gene_lengths.loc[shared_genes, 'gene_length_kb']

        print("Preprocessing miRNAs...")
        self.mir = self._preprocess_mirna(mir_raw)

        common_cells = sorted(list(set(self.rna.columns) & set(self.mir.columns) & set(self.barcodes['barcode'])))
        self.rna = self.rna[common_cells]
        self.mir = self.mir[common_cells]
        self.barcodes = self.barcodes[self.barcodes['barcode'].isin(common_cells)]

        self.tissues = sorted(self.barcodes['line'].unique().tolist())
        print(f"Ready. Cells: {len(common_cells)}, Tissues: {len(self.tissues)}")

    def _preprocess_mirna(self, df):
        df.index = df.index.str.lower()
        pattern = r'[-_](?:5p|3p)$'
        df.index = df.index.str.replace(pattern, '', regex=True)
        return df.groupby(level=0).sum()

    def get_tissues(self):
        return self.tissues

    def get_tissue_barcodes(self, tissue_type: str):
        if tissue_type not in self.get_tissues():
            raise ValueError('Unknown tissue type')
        tissue_specific_barcodes = self.barcodes[self.barcodes['line'].isin([tissue_type])]['barcode']
        return tissue_specific_barcodes

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

    def boot_single_cell(self, df, target_barcodes, N):
        """Return raw single-cell profiles with original barcodes (K=1)."""
        n_unique = len(target_barcodes)
        if N > n_unique:
            warnings.warn(
                f"K=1: requested N={N} samples but only {n_unique} unique cells available. "
                f"Returning {n_unique}.",
                stacklevel=2,
            )
            selected_barcodes = list(target_barcodes)
        elif N < n_unique:
            selected_barcodes = self.rng.choice(
                target_barcodes, size=N, replace=False
            ).tolist()
        else:
            selected_barcodes = list(target_barcodes)

        return df[selected_barcodes]

    def boot_knn(self, df, target_barcodes, K, N, tissue, n_hvg=2000, n_pca=30):
        """
        KNN pseudobulk bootstrap within tissue.

        For each of N bootstrap draws: pick a random anchor cell and sum counts
        of its K nearest neighbors (same PCA+HVG KNN as preprocessor.py).
        """
        data = df[target_barcodes].values
        n_cells = data.shape[1]

        neighbor_idx = self._knn_neighbor_indices(data, K=K, n_hvg=n_hvg, n_pca=n_pca)
        anchor_indices = self.rng.choice(n_cells, size=N, replace=False) # without replacement

        boot_matrix = np.zeros((data.shape[0], N), dtype=np.float64)
        for i, anchor in enumerate(anchor_indices):
            boot_matrix[:, i] = data[:, neighbor_idx[anchor]].sum(axis=1)

        cols = [f'boot_K{K}_{tissue}_{i}' for i in range(N)]
        return pd.DataFrame(boot_matrix, index=df.index, columns=cols)

    def generate(self, tissue: str, K: int = 30, N: int = 50, subset_barcodes: list = None,
                 n_hvg: int = 2000, n_pca: int = 30):
        if tissue not in self.tissues:
            raise ValueError(f"Unknown tissue: {tissue}")

        tissue_barcodes = self.barcodes[self.barcodes['line'] == tissue]['barcode'].tolist()

        if subset_barcodes is not None:
            target_barcodes = sorted(set(tissue_barcodes) & set(subset_barcodes))
            if not target_barcodes:
                raise ValueError(f"No barcodes found for tissue '{tissue}' in the provided subset.")
        else:
            target_barcodes = tissue_barcodes

        n_cells = len(target_barcodes)

        if K == 1:
            rna_boot = self.boot_single_cell(self.rna, target_barcodes, N)
            mir_boot = self.boot_single_cell(self.mir, target_barcodes, N)
        else:
            if n_cells < K:
                raise ValueError(
                    f"Need at least K={K} cells in tissue subset, got {n_cells}."
                )
            rna_boot = self.boot_knn(
                self.rna, target_barcodes, K, N, tissue, n_hvg=n_hvg, n_pca=n_pca
            )
            mir_boot = self.boot_knn(
                self.mir, target_barcodes, K, N, tissue, n_hvg=n_hvg, n_pca=n_pca
            )

        if self.selected_rna:
            rna_boot = rna_boot.loc[rna_boot.index.intersection(self.selected_rna)]
        if self.selected_mirs:
            mir_boot = mir_boot.loc[mir_boot.index.intersection(self.selected_mirs)]

        current_lengths = self.gene_lengths_kb.loc[rna_boot.index]
        rpk = rna_boot.div(current_lengths, axis=0)
        tpm = rpk.div(rpk.sum(axis=0), axis=1) * 1e6

        cpm = mir_boot.div(mir_boot.sum(axis=0), axis=1) * 1e6

        if self.log:
            tpm = np.log2(tpm + 1)
            cpm = np.log2(cpm + 1)

        return tpm, cpm
