import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc 
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

MIN_GENES = 300
MIN_COUNTS = 1000
MT_THRESHOLD = 20.0
GENE_UPPER_QUANTILE = 1 # shuted down
COUNT_UPPER_QUANTILE = 1 # shuted down
MIN_CELLS = 3

def _resolve_10x_file(path: Path, name: str) -> Path:
    """Find a 10x file, allowing any prefix before the standard name.

    ``matrix.mtx.gz`` and ``GSM8515681_MM07_matrix.mtx.gz`` both match.
    """
    path = Path(path)
    matches = sorted(
        p for p in path.iterdir()
        if p.is_file()
        and not p.name.startswith("._")
        and p.name.endswith(name)
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No file ending with {name!r} in {path}")
    raise FileNotFoundError(
        f"Several files ending with {name!r} in {path}: "
        + ", ".join(p.name for p in matches)
    )


def read_10x(path: Path) -> ad.AnnData:
    path = Path(path)

    adata = sc.read_mtx(_resolve_10x_file(path, "matrix.mtx.gz")).T

    genes = pd.read_csv(
        _resolve_10x_file(path, "features.tsv.gz"),
        header=None,
        sep="\t"
    )

    if genes.shape[1] == 1:
        gene_names = genes[0].astype(str)
    else:
        gene_names = genes[1].astype(str)
        adata.var["gene_ids"] = genes[0].astype(str).values

    adata.var_names = ad.utils.make_index_unique(
        pd.Index(gene_names.values)
    )

    barcodes = pd.read_csv(
        _resolve_10x_file(path, "barcodes.tsv.gz"),
        header=None,
        sep="\t"
    )

    adata.obs_names = barcodes[0].astype(str).values

    return adata

def _looks_like_barcodes(names, sample_size: int = 100) -> float:
    """Return fraction of names that look like cell barcodes."""
    sample = pd.Index(names[:sample_size]).astype(str)
    if len(sample) == 0:
        return 0.0
    barcode_like = sample.str.contains(r"[-_]", regex=True) & (sample.str.len() > 12)
    return float(barcode_like.mean())


def load_scrna_csv(path, transpose: str | bool = "auto"):
    """
    Загружает scRNA-seq dataset из .csv или .csv.gz в AnnData.

    По умолчанию ожидается:
        rows    = cells
        columns = genes

    Если transpose="auto", матрица автоматически транспонируется,
    когда строки похожи на гены, а столбцы — на barcodes
  (типичный формат GSE178481: genes x cells).
    """
    df = pd.read_csv(path, compression="infer", index_col=0)

    if transpose == "auto":
        rows_are_barcodes = _looks_like_barcodes(df.index)
        cols_are_barcodes = _looks_like_barcodes(df.columns)
        if cols_are_barcodes > rows_are_barcodes:
            df = df.T
    elif transpose:
        df = df.T

    adata = sc.AnnData(df)
    adata.obs_names = df.index.astype(str)
    adata.var_names = df.columns.astype(str)

    return adata


def load_scrna_counts_tsv(path):
    df = pd.read_csv(
        path,
        sep="\t",
        header=0,
        index_col=0,
        compression="gzip"
    )

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    adata = ad.AnnData(
        X=df.T.to_numpy(dtype=np.float32),
        obs=pd.DataFrame(index=df.columns.astype(str)),
        var=pd.DataFrame(index=df.index.astype(str))
    )

    adata.var_names_make_unique()
    return adata


def load_scrna_txt_gz(path, transpose=False):
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        compression="gzip",
        dtype={0: str, 1: str},
    )

    # Первые два столбца — ID и символы генов
    gene_ids = df.iloc[:, 0].astype(str)
    gene_names = df.iloc[:, 1].astype(str)

    # Остальные столбцы — экспрессия по клеткам
    X = df.iloc[:, 2:].apply(pd.to_numeric, errors="raise")

    # Строки исходного файла — гены, столбцы — клетки
    X.index = gene_names

    # Для AnnData нужны строки-клетки, столбцы-гены
    X = X.T

    adata = sc.AnnData(X)
    adata.obs_names = X.index.astype(str)
    adata.var_names = X.columns.astype(str).rename(None)
    adata.var_names_make_unique()

    # Сохраняем Ensembl ID отдельно
    adata.var["gene_id"] = gene_ids.values

    return adata

def annotate_qc(adata: ad.AnnData) -> None:
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

def apply_qc_filters(
    adata: ad.AnnData,
    verbose: bool = False,
    *,
    filter_genes: bool = True,
) -> ad.AnnData:
    before = adata.n_obs
    genes_before = adata.n_vars
    adata = adata[adata.obs["n_genes_by_counts"] >= MIN_GENES].copy()
    adata = adata[adata.obs["total_counts"] >= MIN_COUNTS].copy()
    
    max_genes = np.quantile(
        adata.obs["n_genes_by_counts"],
        GENE_UPPER_QUANTILE,
    )
    adata = adata[adata.obs["n_genes_by_counts"] <= max_genes].copy()

    max_counts = np.quantile(
        adata.obs["total_counts"],
        COUNT_UPPER_QUANTILE,
    )
    adata = adata[adata.obs["total_counts"] <= max_counts].copy()
    adata = adata[adata.obs["pct_counts_mt"] < MT_THRESHOLD].copy()
    # Per-sample gene filtering drops genes rare in one sample and, with an
    # inner concat, removes them from the whole integration. Prefer filtering
    # once after outer concat (see integrate_adatas).
    if filter_genes:
        sc.pp.filter_genes(adata, min_cells=MIN_CELLS)

    if verbose:
        print(
            f"Cells: {before} -> {adata.n_obs} | "
            f"Genes: {genes_before} -> {adata.n_vars}\n"
            f"{GENE_UPPER_QUANTILE}th percentile n_genes = {max_genes:.0f}\n"
            f"{COUNT_UPPER_QUANTILE}th percentile total_counts = {max_counts:.0f}"
        )

    return adata


def qc_filter(
    adata: ad.AnnData,
    verbose: bool = False,
    *,
    filter_genes: bool = True,
) -> ad.AnnData:
    annotate_qc(adata)
    adata = apply_qc_filters(adata, verbose=verbose, filter_genes=filter_genes)

    # Scrublet runs an internal PCA with default n_prin_comps=30.
    # That fails when too few cells/genes survive QC.
    n_prin = min(30, max(adata.n_obs - 1, 1), max(adata.n_vars - 1, 1))
    if adata.n_obs >= 10 and n_prin >= 2:
        sc.pp.scrublet(adata, n_prin_comps=n_prin)
        adata = adata[~adata.obs["predicted_doublet"]].copy()
    else:
        if verbose:
            print(
                f"Skipping scrublet: only {adata.n_obs} cells after QC "
                f"(need >= 10)."
            )
        adata.obs["predicted_doublet"] = False
        adata.obs["doublet_score"] = 0.0

    return adata

def main_pipeline(adata, n_components=15, resolution=0.1):
    adata = adata.copy()

    # ============================================================
    # 1. RAW COUNTS
    # ============================================================
    # Сохраняем исходные count-данные, они понадобятся дальше
    adata.layers["counts"] = adata.X.copy()

    # ============================================================
    # 2. NORMALIZATION + LOG TRANSFORMATION
    # ============================================================
    # Library-size normalization to 10,000 counts per cell
    sc.pp.normalize_total(adata, target_sum=1e4)

    # log(1 + x)
    sc.pp.log1p(adata)

    # Сохраняем log-normalized expression для DE и других анализов
    adata.layers["lognorm"] = adata.X.copy()

    # Также сохраняем log-normalized данные во встроенный .raw
    # .raw останется неизменным после последующего scaling
    adata.raw = adata

    # ============================================================
    # 3. HIGHLY VARIABLE GENES
    # ============================================================
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=2000,
        flavor="seurat",
    )

    # ============================================================
    # 4. SCALING
    # ============================================================
    # Теперь X становится scaled (Z-score)
    # Но counts и lognorm уже сохранены в layers
    sc.pp.scale(adata, max_value=10)

    # ============================================================
    # 5. PCA
    # ============================================================
    # PCA выполняется только на highly variable genes
    sc.tl.pca(
        adata,
        svd_solver="arpack",
        n_comps=n_components,
        use_highly_variable=True,
    )

    # ============================================================
    # 6. NEIGHBORS
    # ============================================================
    sc.pp.neighbors(
        adata,
        n_neighbors=10,
        n_pcs=n_components,
    )

    # ============================================================
    # 7. CLUSTERING
    # ============================================================
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added="clusters",
    )

    # ============================================================
    # 8. UMAP
    # ============================================================
    sc.tl.umap(
        adata,
        n_components=2,
    )

    return adata

def run_pipeline(path: Path, n_components=15, resolution=0.1, is_10x=True) -> ad.AnnData:
    if is_10x:
        adata = read_10x(path)

    elif path.name.endswith((".csv", ".csv.gz")):
        adata = load_scrna_csv(path) 

    elif path.name.endswith((".tsv", ".tsv.gz")):
        adata = load_scrna_counts_tsv(path)

    elif path.name.endswith(".txt.gz") or path.name.endswith(".txt"):
        adata = load_scrna_txt_gz(path)
        
    else:
        raise ValueError(f"Unsupported file format: {path.name}")

    adata = qc_filter(adata)
    adata = main_pipeline(adata, n_components=n_components, resolution=resolution)
    return adata



def find_top_markers(adata, top_k=10, min_logfc=0.5, max_pval_adj=0.05):
    """
    Find top differentially expressed genes for each cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated single-cell dataset containing cluster labels
        in adata.obs["clusters"].

    top_k : int
        Number of top marker genes to return for each cluster.

    min_logfc : float
        Minimum log2 fold change.

    max_pval_adj : float
        Maximum adjusted p-value.

    Returns
    -------
    dict
        Dictionary with cluster names as keys and lists of
        top marker genes as values.
    """

    # Differential expression analysis
    sc.tl.rank_genes_groups(
        adata,
        groupby="clusters",
        method="wilcoxon",
        layer="lognorm",
        use_raw=False,
    )

    # Convert results to DataFrame
    markers = sc.get.rank_genes_groups_df(
        adata,
        group=None
    )

    # Filter significant and upregulated genes
    markers = markers[
        (markers["pvals_adj"] < max_pval_adj) &
        (markers["logfoldchanges"] > min_logfc)
    ].copy()

    # Sort by cluster and log2 fold change
    markers = markers.sort_values(
        ["group", "logfoldchanges"],
        ascending=[True, False]
    )

    # Select top K genes for each cluster
    top_markers = (
        markers
        .groupby("group")
        .head(top_k)
        .groupby("group")["names"]
        .apply(list)
        .to_dict()
    )

    return top_markers


def add_mirna_predictions(
    adata: ad.AnnData,
    preds: str | Path | pd.DataFrame,
    *,
    barcode_col: str = "barcode",
    celltype_col: str | None = "CellType",
    obsm_key: str = "mirna",
    add_to_obs: bool = True,
    aggregate_folds: bool = False,
    inplace: bool = True,
) -> ad.AnnData:
    """
    Add predicted miRNA expression from a CSV/DataFrame to AnnData.

    Expects a table with a barcode column and numeric miRNA columns
    (optionally a CellType column). Rows are aligned to adata.obs_names.

    Parameters
    ----------
    adata
        AnnData object to annotate.
    preds
        Path to CSV or a DataFrame with predictions.
    barcode_col
        Column with cell barcodes.
    celltype_col
        If present in preds, copy into adata.obs. Pass None to skip.
    obsm_key
        Key under adata.obsm for the miRNA matrix (cells × miRNAs).
    add_to_obs
        If True, also write miRNA columns into adata.obs
        (needed for sc.pl.umap(..., color=...)).
    aggregate_folds
        If True, average columns that share the same miRNA name across
        fold prefixes (K1_, K2_, ...). Resulting names are bare miRNAs
        (e.g. hsa-mir-486-3p).
    inplace
        Modify adata in place. If False, work on a copy.

    Returns
    -------
    ad.AnnData
        Annotated AnnData (same object if inplace=True).
    """
    if not inplace:
        adata = adata.copy()

    if isinstance(preds, (str, Path)):
        preds_df = pd.read_csv(preds)
    else:
        preds_df = preds.copy()

    if barcode_col not in preds_df.columns:
        raise ValueError(
            f"Column '{barcode_col}' not found in predictions. "
            f"Available: {list(preds_df.columns[:10])}..."
        )

    preds_df = preds_df.set_index(barcode_col)
    preds_df.index = preds_df.index.astype(str)

    common = adata.obs_names.intersection(preds_df.index)
    if len(common) == 0:
        raise ValueError(
            "No overlapping barcodes between adata and predictions."
        )

    n_missing = adata.n_obs - len(common)
    if n_missing > 0:
        print(
            f"Warning: {n_missing}/{adata.n_obs} cells have no predictions; "
            "their values will be NaN."
        )

    preds_aligned = preds_df.reindex(adata.obs_names)

    # Optional CellType metadata
    if celltype_col is not None and celltype_col in preds_aligned.columns:
        adata.obs[celltype_col] = pd.Categorical(
            preds_aligned[celltype_col].values
        )
        meta_drop = [celltype_col]
    else:
        meta_drop = []

    mirna = preds_aligned.drop(columns=meta_drop, errors="ignore")
    # Keep only numeric prediction columns
    mirna = mirna.select_dtypes(include=[np.number])

    if mirna.shape[1] == 0:
        raise ValueError("No numeric miRNA prediction columns found.")

    if aggregate_folds:
        # K1_hsa-mir-xxx / K2_hsa-mir-xxx -> mean over folds -> hsa-mir-xxx
        groups: dict[str, list[str]] = {}
        for col in mirna.columns:
            name = col.split("_", 1)[1] if "_" in col else col
            groups.setdefault(name, []).append(col)
        mirna = pd.DataFrame(
            {
                name: mirna[cols].mean(axis=1)
                for name, cols in groups.items()
            },
            index=mirna.index,
        )

    adata.obsm[obsm_key] = mirna.to_numpy(dtype=np.float64)
    adata.uns[f"{obsm_key}_names"] = mirna.columns.tolist()

    if add_to_obs:
        for col in mirna.columns:
            adata.obs[col] = mirna[col].to_numpy(dtype=np.float64)

    print(
        f"Added {mirna.shape[1]} miRNA features "
        f"({len(common)}/{adata.n_obs} cells matched) "
        f"-> obsm['{obsm_key}']"
        + (" and obs" if add_to_obs else "")
    )

    return adata


def save_counts_csv(adata, output_path):
    # Check that counts layer exists
    if "counts" not in adata.layers:
        raise ValueError(
            "Layer 'counts' not found in adata.layers. "
            "Cannot save count matrix."
        )

    counts = adata.layers["counts"]

    # Get non-zero values for validation
    if hasattr(counts, "data"):
        values = np.asarray(counts.data)
    else:
        values = np.asarray(counts)

    # Check for negative values
    if (values < 0).any():
        raise ValueError(
            "Layer 'counts' contains negative values. "
            "Expected non-negative integer counts."
        )

    # Check that all values are integers
    if not np.all(np.equal(values, np.floor(values))):
        raise ValueError(
            "Layer 'counts' does not contain integer values. "
            "Expected raw count data, not normalized or scaled data."
        )

    # Convert counts to DataFrame
    counts_df = pd.DataFrame(
        counts.toarray() if hasattr(counts, "toarray") else counts,
        index=adata.obs_names,
        columns=adata.var_names,
    )

    # Convert counts from float to integer
    counts_df = counts_df.astype(np.int64)

    # Add barcode as the first column
    counts_df.index.name = "barcode"
    counts_df.reset_index(inplace=True)

    # Add CellType as the second column
    counts_df.insert(
        1,
        "CellType",
        adata.obs["CellType"].values
    )

    # Save
    counts_df.to_csv(
        output_path,
        index=False
    )

    return counts_df


###### INTEGRATION ######

def _fillna_zeros(X):
    """Replace NaNs introduced by outer concat with zeros (counts)."""
    import scipy.sparse as sp

    if sp.issparse(X):
        X = X.tocsr(copy=True)
        if X.data.size and np.isnan(X.data).any():
            X.data = np.nan_to_num(X.data, nan=0.0)
        return X
    X = np.asarray(X, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0)


def _merge_gene_ids(adatas: list[ad.AnnData], var_names: pd.Index) -> pd.Series | None:
    """Collapse per-sample gene_ids onto the union var index."""
    if not any("gene_ids" in a.var.columns for a in adatas):
        return None
    merged = pd.Series(index=pd.Index(var_names.astype(str)), dtype=object)
    for a in adatas:
        if "gene_ids" not in a.var.columns:
            continue
        ids = a.var["gene_ids"].astype(str)
        ids.index = a.var_names.astype(str)
        for gene, gid in ids.items():
            if gene not in merged.index:
                continue
            cur = merged.loc[gene]
            if pd.isna(cur) or str(cur) in ("", "nan", "None"):
                merged.loc[gene] = gid
    return merged


_INTEGRATION_REPS = {
    "harmony": "X_pca_harmony",
    "scanorama": "X_scanorama",
}


def _correct_batch(adata: ad.AnnData, method: str) -> str:
    """Batch-correct PCA. Returns the ``obsm`` key for neighbors."""
    method = method.lower()
    if method not in _INTEGRATION_REPS:
        raise ValueError(
            "method must be 'harmony' or 'scanorama', "
            f"got {method!r}"
        )
    if method == "harmony":
        sc.external.pp.harmony_integrate(adata, key="sample")
    else:
        # Samples are concatenated one after another, so each batch is contiguous.
        sc.external.pp.scanorama_integrate(adata, key="sample")
    return _INTEGRATION_REPS[method]


def integrate_adatas(
    adatas: list[ad.AnnData],
    sample_names: list[str] | None = None,
    n_components=15,
    resolution=0.1,
    *,
    join: str = "outer",
    min_cells_gene: int | None = MIN_CELLS,
    method: str = "harmony",
) -> ad.AnnData:
    """
    Combine and integrate multiple QC-filtered AnnData objects.

    Parameters
    ----------
    adatas
        List of QC-filtered AnnData objects.
    sample_names
        Optional names identifying the samples.
    n_components
        Number of PCA components.
    resolution
        Leiden clustering resolution.
    join
        Gene join for ``sc.concat``. ``"outer"`` (default) keeps the union of
        genes (missing → 0). ``"inner"`` keeps only genes shared by all samples.
    min_cells_gene
        If not None, run ``filter_genes(min_cells=...)`` once *after* concat.
        Set to ``None`` to keep every gene.
    method
        Batch correction after PCA. ``"harmony"`` (default) or ``"scanorama"``.

    Returns
    -------
    ad.AnnData
        Integrated AnnData object.
    """

    if len(adatas) < 2:
        raise ValueError(
            "At least two AnnData objects are required for integration."
        )

    if sample_names is None:
        sample_names = [f"sample_{i+1}" for i in range(len(adatas))]

    if len(sample_names) != len(adatas):
        raise ValueError(
            "The number of sample_names must match the number of AnnData objects."
        )

    if len(set(sample_names)) != len(sample_names):
        raise ValueError(
            "sample_names must be unique so they can be categorical batch labels. "
            f"Got: {list(sample_names)}"
        )

    n_genes_before = [a.n_vars for a in adatas]

    adata = sc.concat(
        adatas,
        axis="obs",
        join=join,
        label="sample",
        keys=sample_names,
        index_unique="-",
    )
    adata.X = _fillna_zeros(adata.X)

    gene_ids = _merge_gene_ids(adatas, adata.var_names)
    if gene_ids is not None:
        adata.var["gene_ids"] = gene_ids.reindex(adata.var_names.astype(str)).values

    if min_cells_gene is not None:
        sc.pp.filter_genes(adata, min_cells=min_cells_gene)

    print(
        f"Integration concat join={join!r}: per-sample genes {n_genes_before} "
        f"-> combined {adata.n_vars} genes × {adata.n_obs} cells"
    )

    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.layers["lognorm"] = adata.X.copy()
    adata.raw = adata

    # HVG for PCA only — does not drop genes from layers['counts']
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=2000,
        flavor="seurat",
        batch_key="sample",
    )

    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(
        adata,
        svd_solver="arpack",
        n_comps=n_components,
        use_highly_variable=True,
    )

    use_rep = _correct_batch(adata, method)
    adata.uns["integration"] = method.lower()

    sc.pp.neighbors(adata, n_neighbors=10, use_rep=use_rep)
    sc.tl.leiden(adata, resolution=resolution, key_added="clusters")
    sc.tl.umap(adata, n_components=2)

    return adata


def _sample_labels(paths: list[Path]) -> list[str]:
    """Unique batch labels for ``sc.concat``.

    ``mine/cervic/1`` (matrix in the sample folder) → ``1``.
    ``.../solo/3/data`` (shared leaf name) → ``3``.
    """
    paths = [Path(p) for p in paths]

    def _unique(labels: list[str]) -> bool:
        return len(set(labels)) == len(labels)

    names = [p.name for p in paths]
    if _unique(names):
        return names

    parents = [p.parent.name for p in paths]
    if _unique(parents):
        return parents

    labels = [f"{p.parent.name}_{p.name}" for p in paths]
    if _unique(labels):
        return labels

    raise ValueError(
        "Could not build unique sample names from paths: "
        + ", ".join(str(p) for p in paths)
    )


def run_pipeline_integrate(
    paths: list[Path],
    n_components=15,
    resolution=0.1,
    is_10x=True,
    *,
    join: str = "outer",
    min_cells_gene: int | None = MIN_CELLS,
    method: str = "harmony",
) -> ad.AnnData:
    """
    QC-filter each sample, then integrate with Harmony or Scanorama.

    Important: only raw counts after QC are passed to integrate_adatas.
    Do not run main_pipeline / run_pipeline per sample first — that would
    scale the data and break re-normalization / HVG during integration.

    Gene retention
    --------------
    - Per-sample ``filter_genes`` is disabled.
    - Samples are concatenated with ``join="outer"`` (union of genes).
    - Optional combined ``filter_genes(min_cells=...)`` runs once after concat.
    - ``var['gene_ids']`` is merged across samples when available.
    """

    if len(paths) < 2:
        raise ValueError(
            "At least two sample paths are required for integration."
        )

    adatas = []
    sample_names = _sample_labels(paths)

    for path in paths:
        if is_10x:
            adata = read_10x(path)
        
        elif path.name.endswith((".csv", ".csv.gz")):
            adata = load_scrna_csv(path, transpose=True) 

        elif path.name.endswith((".tsv", ".tsv.gz")):
            adata = load_scrna_counts_tsv(path)

        elif path.name.endswith(".txt.gz") or path.name.endswith(".txt"):
            adata = load_scrna_txt_gz(path)

        else:
            raise ValueError(f"Unsupported file format: {path.name}")

        # Keep all genes per sample; filter on the union after concat.
        adata = qc_filter(adata, filter_genes=False)
        adatas.append(adata)

    return integrate_adatas(
        adatas,
        sample_names=sample_names,
        n_components=n_components,
        resolution=resolution,
        join=join,
        min_cells_gene=min_cells_gene,
        method=method,
    )


def plot_dotplot(adata, n_genes=5):

    if "CellType" not in adata.obs.columns:
        raise ValueError("'CellType' not found in adata.obs")

    if "rank_genes_groups" not in adata.uns:
        raise ValueError(
            "Run sc.tl.rank_genes_groups() first."
        )

    # Groups from rank_genes_groups
    groups = adata.uns["rank_genes_groups"]["names"].dtype.names

    # Top N genes for each CellType
    top_genes = {
        group: list(
            adata.uns["rank_genes_groups"]["names"][group][:n_genes]
        )
        for group in groups
    }

    # Dotplot
    sc.pl.dotplot(
        adata,
        var_names=top_genes,
        groupby="CellType",
        standard_scale="var",
        dendrogram=False,
    )