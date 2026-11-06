import scanpy as sc
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 


import seaborn as sns
from scipy.stats import mannwhitneyu, ks_2samp

def plot_umap(adata, color='CellType'):
    sc.pl.umap(
    adata,
    color=color,
    legend_loc="on data",
    size=50,
    alpha=1,
    frameon=False,
    show=False,
    add_outline=True,
    outline_width=(0.2, 0.1),
    )
    
def paired_umap(rcc_adata, healthy_adata):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sc.pl.umap(
        rcc_adata,
        color="CellType",
        legend_loc="on data",
        size=50,
        alpha=1,
        frameon=False,
        ax=axes[0],
        show=False,
        add_outline=True,
        outline_width=(0.2, 0.1),
    )
    
    axes[0].set_title(f"RCC: {rcc_adata.shape[0]} cells")
    
    sc.pl.umap(
        healthy_adata,
        color="CellType",
        legend_loc="on data",
        size=50,
        alpha=1,
        frameon=False,
        ax=axes[1],
        show=False,
        add_outline=True,
        outline_width=(0.2, 0.1)
    )
    
    axes[1].set_title(f"Healthy: {healthy_adata.shape[0]} cells")
    
    plt.tight_layout()
    plt.show()


def plot_dotplot(adata, n_genes=5, rcc=True, cmap='viridis'):

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

    if rcc:
        sc.pl.dotplot(
            adata,
            var_names=top_genes,
            groupby="CellType",
            standard_scale="var",
            dendrogram=False, 
            cmap=cmap
        )
    else:
        sc.pl.dotplot(
            adata,
            var_names=top_genes,
            groupby="CellType",
            standard_scale="var",
            dendrogram=False,
            use_raw=False,
            layer="log1p",
            cmap=cmap
        )
        



def plot_umap_expression(adata, mir, cluster_col = 'CellType', cmap="inferno", size=50, alpha=1):
    sc.pl.umap(
        adata,
        color=[cluster_col, mir],
        legend_loc="on data",
        size=size,
        alpha=alpha,
        title=["Cell types", mir],
        cmap=cmap,
        frameon=False,
        add_outline=True,
        outline_width=(0.2, 0.1),
    )


    
def plot_paired_umap_expression(rcc_adata, healthy_adata, mir, cmap="inferno", size=50, alpha=1):

    expr_rcc = rcc_adata.obs[mir].to_numpy(dtype=float)
    expr_healthy = healthy_adata.obs[mir].to_numpy(dtype=float)

    vmin = np.nanmin(
        np.concatenate([expr_rcc, expr_healthy])
    )

    vmax = np.nanmax(
        np.concatenate([expr_rcc, expr_healthy])
    )

    sc.pl.umap(
        rcc_adata,
        color=["CellType", mir],
        legend_loc="on data",
        size=size,
        alpha=alpha,
        title=["RCC cell types", mir],
        cmap=cmap,
        frameon=False,
        add_outline=True,
        outline_width=(0.2, 0.1),
        vmin=vmin,
        vmax=vmax,
    )

    sc.pl.umap(
        healthy_adata,
        color=["CellType", mir],
        legend_loc="on data",
        size=size,
        alpha=alpha,
        title=["Healthy renal cell types", mir],
        cmap=cmap,
        frameon=False,
        add_outline=True,
        outline_width=(0.2, 0.1),
        vmin=vmin,
        vmax=vmax,
    )
 

def cliffs_delta(x, y):
    """
    Calculate Cliff's delta.

    Positive value:
        x tends to have higher values than y.

    Negative value:
        x tends to have lower values than y.

    Value close to 0:
        distributions are similar.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    # Compare every x with every y
    diff = x[:, None] - y[None, :]

    delta = (
        np.sum(diff > 0) -
        np.sum(diff < 0)
    ) / (len(x) * len(y))

    return delta



def compare_expression(
    df1,
    df2,
    gene,
    cluster1_name="Malignant cells",
    cluster2_name="Proximal tubule cells",
    cluster1_color="#C44E52",
    cluster2_color="#4C72B0",
    pseudocount=1e-6,
    max_plot_cells=5000,
    random_state=42,
):
    """
    Compare predicted gene expression between two clusters.

    Statistics are calculated using all available cells.
    For large datasets, only a random subset of cells is used
    for visualization to reduce memory usage.

    Left:
        Violin plot

    Right:
        KDE density plot

    Statistics:
        - mean expression
        - median expression
        - percentage of expressing cells
        - log2 fold change based on mean expression
        - Mann–Whitney U test
        - Cliff's delta for cluster-level comparisons

    Parameters
    ----------
    df1 : pandas.DataFrame
        First expression dataframe.

    df2 : pandas.DataFrame
        Second expression dataframe.

    gene : str
        Gene to compare.

    cluster1_name : str, default="Malignant cells"
        Cell type / cluster name in df1.
        Use "all" to compare all cells in df1.

    cluster2_name : str, default="Proximal tubule cells"
        Cell type / cluster name in df2.
        If cluster1_name == "all", all cells in df2 are used.

    cluster1_color : str
        Color for the first group.

    cluster2_color : str
        Color for the second group.

    pseudocount : float
        Small value added when calculating log2FC.

    max_plot_cells : int, default=5000
        Maximum number of cells per group used for plotting.
        Statistics are always calculated using all cells.

    random_state : int, default=42
        Random seed used for visualization subsampling.

    Returns
    -------
    stats : dict
        Summary statistics and Mann–Whitney U test results.
    """

    # =====================================================
    # SELECT DATA
    # =====================================================

    if gene not in df1.columns:
        raise ValueError(
            f"Gene '{gene}' was not found in df1."
        )

    if gene not in df2.columns:
        raise ValueError(
            f"Gene '{gene}' was not found in df2."
        )

    # Compare all cells
    if cluster1_name == "all":

        cluster1 = df1[gene]
        cluster2 = df2[gene]

        cluster2_name = "all"

    # Compare specific cell types
    else:

        if "CellType" not in df1.columns:
            raise ValueError(
                "'CellType' column was not found in df1."
            )

        if "CellType" not in df2.columns:
            raise ValueError(
                "'CellType' column was not found in df2."
            )

        cluster1 = df1[
            df1["CellType"] == cluster1_name
        ][gene]

        cluster2 = df2[
            df2["CellType"] == cluster2_name
        ][gene]

    # =====================================================
    # LABELS
    # =====================================================

    # Keep intentional naming:
    # "Malignant all" / "Healthy all"
    if cluster1_name == cluster2_name:
        cluster1_name = "Malignant " + cluster1_name
        cluster2_name = "Healthy " + cluster2_name

    # =====================================================
    # PREPARE DATA
    # =====================================================

    cluster1 = np.asarray(
        cluster1,
        dtype=np.float64,
    )

    cluster2 = np.asarray(
        cluster2,
        dtype=np.float64,
    )

    # Remove NaN / inf
    cluster1 = cluster1[np.isfinite(cluster1)]
    cluster2 = cluster2[np.isfinite(cluster2)]

    if len(cluster1) == 0 or len(cluster2) == 0:
        raise ValueError(
            f"No valid expression values found for '{gene}' "
            f"in one or both groups."
        )

    # =====================================================
    # STATISTICS
    # =====================================================

    # Mann–Whitney U
    mw_stat, mw_p = mannwhitneyu(
        cluster1,
        cluster2,
        alternative="two-sided",
    )

    # Cliff's delta
    #
    # Do not calculate it for "all" comparisons because
    # very large cell numbers make pairwise calculations
    # extremely memory-intensive.
    is_all_comparison = (
        cluster1_name == "Malignant all"
        and cluster2_name == "Healthy all"
    )

    if is_all_comparison:
        delta = None
    else:
        delta = cliffs_delta(
            cluster1,
            cluster2,
        )

    # Mean
    mean1 = np.mean(cluster1)
    mean2 = np.mean(cluster2)

    # Median
    median1 = np.median(cluster1)
    median2 = np.median(cluster2)

    # Percentage of expressing cells
    pct_expr1 = np.mean(cluster1 > 0) * 100
    pct_expr2 = np.mean(cluster2 > 0) * 100

    # log2 fold change
    log2fc = np.log2(
        (mean1 + pseudocount)
        /
        (mean2 + pseudocount)
    )

    # =====================================================
    # SUBSAMPLE FOR VISUALIZATION ONLY
    # =====================================================

    rng = np.random.default_rng(random_state)

    if len(cluster1) > max_plot_cells:
        plot1 = rng.choice(
            cluster1,
            size=max_plot_cells,
            replace=False,
        )
    else:
        plot1 = cluster1

    if len(cluster2) > max_plot_cells:
        plot2 = rng.choice(
            cluster2,
            size=max_plot_cells,
            replace=False,
        )
    else:
        plot2 = cluster2

    # =====================================================
    # DATAFRAME FOR PLOTTING
    # =====================================================

    plot_df = pd.DataFrame({
        "expression": np.concatenate(
            [plot1, plot2]
        ),
        "cluster": (
            [cluster1_name] * len(plot1)
            +
            [cluster2_name] * len(plot2)
        ),
    })

    palette = {
        cluster1_name: cluster1_color,
        cluster2_name: cluster2_color,
    }

    # =====================================================
    # FIGURE
    # =====================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    # =====================================================
    # LEFT: VIOLIN
    # =====================================================

    sns.violinplot(
        data=plot_df,
        x="cluster",
        y="expression",
        hue="cluster",
        palette=palette,
        inner="box",
        cut=0,
        legend=False,
        ax=axes[0],
    )

    axes[0].set_xlabel("")
    axes[0].set_ylabel(
        f"{gene} predicted expression"
    )
    axes[0].set_title(
        f"{gene}"
    )

    # =====================================================
    # RIGHT: KDE
    # =====================================================

    sns.kdeplot(
        data=plot_df,
        x="expression",
        hue="cluster",
        hue_order=[
            cluster1_name,
            cluster2_name,
        ],
        palette=palette,
        fill=True,
        common_norm=False,
        alpha=0.3,
        linewidth=2,
        ax=axes[1],
    )

    # Mean lines are based on ALL cells
    axes[1].axvline(
        mean1,
        color=cluster1_color,
        linestyle="--",
        linewidth=1.5,
    )

    axes[1].axvline(
        mean2,
        color=cluster2_color,
        linestyle="--",
        linewidth=1.5,
    )

    axes[1].set_xlabel(
        f"{gene} predicted expression"
    )
    axes[1].set_ylabel(
        "Density"
    )
    axes[1].set_title(
        "Expression distribution"
    )

    # =====================================================
    # STATISTICS TEXT
    # =====================================================

    if delta is None:
        delta_text = "N/A"
    else:
        delta_text = f"{delta:.2f}"

    stats_text = (
        f"{cluster1_name}: "
        f"mean = {mean1:.3f}, "
        f"median = {median1:.3f}, "
        f"expressing = {pct_expr1:.1f}%, "
        f"n = {len(cluster1)}\n"

        f"{cluster2_name}: "
        f"mean = {mean2:.3f}, "
        f"median = {median2:.3f}, "
        f"expressing = {pct_expr2:.1f}%, "
        f"n = {len(cluster2)}\n"

        f"MW p = {mw_p:.2e}  |  "
        f"log₂FC = {log2fc:.2f}  |  "
        f"Cliff's delta = {delta_text}"
    )

    fig.text(
        0.5,
        0.01,
        stats_text,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    plt.tight_layout(
        rect=[0, 0.12, 1, 1]
    )

    plt.show()

    # =====================================================
    # RETURN STATISTICS
    # =====================================================

    return {
        "gene": gene,

        "cluster1": cluster1_name,
        "cluster2": cluster2_name,

        "n_cluster1": len(cluster1),
        "n_cluster2": len(cluster2),

        "mean_cluster1": mean1,
        "mean_cluster2": mean2,

        "median_cluster1": median1,
        "median_cluster2": median2,

        "pct_expressing_cluster1": pct_expr1,
        "pct_expressing_cluster2": pct_expr2,

        "log2FC": log2fc,

        "MW_U": mw_stat,
        "MW_p": mw_p,

        "Cliffs_delta": delta,
    }






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu


def compare_expression_across_clusters(
    df1,
    df2,
    gene,
    cluster_pairs=None,
    cluster_col="CellType",
    malignant_color="#C44E52",
    healthy_color="#4C72B0",
    pseudocount=1e-6,
    figsize=(8, 5.5),
    sort_by=None,
    save_path=None,
    fc_threshold=1.0,
    neutral_color="#999999",
):
    """
    Compare predicted gene expression across matched clusters
    in malignant (df1) and healthy (df2) scRNA-seq datasets.

    Expression values must be log2(CPM + 1).

    Metrics
    -------
    log2FC:
        log2 ratio of mean CPM (malignant / healthy).

    Cliff's delta:
        Distributional effect size, computed from Mann-Whitney U.
        Positive = higher expression in malignant.
        Negative = higher expression in healthy.

    Parameters
    ----------
    df1, df2 : pandas.DataFrame
        Malignant and healthy expression datasets.

    gene : str
        Gene to compare.

    cluster_pairs : list of tuples
        (malignant_cluster, healthy_cluster).

    cluster_col : str
        Column containing cell-type annotations.

    pseudocount : float
        Pseudocount in CPM units for log2FC.

    figsize : tuple
        Figure size.

    sort_by : str or None
        None, "log2FC", "abs_log2FC",
        "Cliffs_delta", "abs_Cliffs_delta".

    save_path : str or None
        Optional path to save figure.

    Returns
    -------
    pd.DataFrame
        Per-cluster summary statistics.
    """

    if cluster_pairs is None:
        cluster_pairs = [
            ("Malignant cells", "Proximal tubule cells"),
            ("Endothelial cells", "Endothelial cells"),
            ("Pericytes", "Interstitial cells"),
            ("T and NK cells", "T and NK cells"),
            ("Phagocytes", "Phagocytes"),
            ("B cells", "B cells"),
        ]

    # -----------------------------------------
    # Validation
    # -----------------------------------------

    for name, df in [("df1", df1), ("df2", df2)]:
        if gene not in df.columns:
            raise ValueError(f"Gene '{gene}' not found in {name}")

        if cluster_col not in df.columns:
            raise ValueError(
                f"Column '{cluster_col}' not found in {name}"
            )

    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive.")

    results = []

    # -----------------------------------------
    # Calculate metrics for each cluster pair
    # -----------------------------------------

    for malignant_cluster, healthy_cluster in cluster_pairs:

        x = df1.loc[
            df1[cluster_col] == malignant_cluster,
            gene
        ].to_numpy(dtype=np.float64)

        y = df2.loc[
            df2[cluster_col] == healthy_cluster,
            gene
        ].to_numpy(dtype=np.float64)

        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

        if len(x) == 0 or len(y) == 0:
            raise ValueError(
                f"No valid cells for pair: "
                f"{malignant_cluster} / {healthy_cluster}"
            )

        # Clip small negative predictions to zero.
        x = np.maximum(x, 0)
        y = np.maximum(y, 0)

        # Restore linear CPM.
        x_cpm = np.expm1(x * np.log(2))
        y_cpm = np.expm1(y * np.log(2))

        mean_malignant = np.mean(x_cpm)
        mean_healthy = np.mean(y_cpm)

        log2fc = np.log2(
            (mean_malignant + pseudocount)
            / (mean_healthy + pseudocount)
        )

        # Cliff's delta from Mann-Whitney U.
        # No pairwise comparison matrix is created.
        u_stat = mannwhitneyu(
            x,
            y,
            alternative="two-sided",
            method="asymptotic",
        ).statistic

        delta = (
            2.0 * u_stat / (len(x) * len(y))
        ) - 1.0

        if malignant_cluster == healthy_cluster:
            label = malignant_cluster
        else:
            label = (
                f"{malignant_cluster}\n"
                f"vs {healthy_cluster}"
            )

        results.append({
            "cluster": label,
            "malignant_cluster": malignant_cluster,
            "healthy_cluster": healthy_cluster,
            "n_malignant": len(x),
            "n_healthy": len(y),
            "mean_CPM_malignant": mean_malignant,
            "mean_CPM_healthy": mean_healthy,
            "median_log2CPM_malignant": np.median(x),
            "median_log2CPM_healthy": np.median(y),
            "pct_expressing_malignant": np.mean(x > 0) * 100,
            "pct_expressing_healthy": np.mean(y > 0) * 100,
            "log2FC": log2fc,
            "Cliffs_delta": delta,
        })

    stats_df = pd.DataFrame(results)

    # -----------------------------------------
    # Optional sorting
    # -----------------------------------------

    allowed_sort = {
        None,
        "log2FC",
        "abs_log2FC",
        "Cliffs_delta",
        "abs_Cliffs_delta",
    }

    if sort_by not in allowed_sort:
        raise ValueError(f"sort_by must be one of {allowed_sort}")

    if sort_by is not None:
        key = (
            stats_df[sort_by[4:]].abs()
            if sort_by.startswith("abs_")
            else stats_df[sort_by]
        )

        stats_df = stats_df.iloc[
            np.argsort(-key.to_numpy())
        ].reset_index(drop=True)

    # -----------------------------------------
    # Plot
    # -----------------------------------------
    
    n = len(stats_df)
    positions = np.arange(n)
    
    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    
    # Colors are determined ONLY by log2FC
    fc_values = stats_df["log2FC"].to_numpy()
    
    colors = np.where(
        fc_values >= fc_threshold,
        malignant_color,
        np.where(
            fc_values <= -fc_threshold,
            healthy_color,
            neutral_color,
        ),
    )
    
    metrics = [
        ("log2FC", "log₂ fold change (mean CPM)"),
        ("Cliffs_delta", "Cliff's delta"),
    ]
    
    for ax, (metric, xlabel) in zip(axes, metrics):
    
        values = stats_df[metric].to_numpy()
    
        ax.axvline(
            0,
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=1,
        )
    
        # Threshold lines only on log2FC panel
        if metric == "log2FC":
            ax.axvline(
                fc_threshold,
                color=malignant_color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.7,
            )
    
            ax.axvline(
                -fc_threshold,
                color=healthy_color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.7,
            )
    
        for i, (value, color) in enumerate(zip(values, colors)):
    
            ax.plot(
                [0, value],
                [i, i],
                color=color,
                linewidth=2,
                alpha=0.55,
                zorder=2,
            )
    
            ax.scatter(
                value,
                i,
                s=85,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
    
        ax.set_xlabel(xlabel)
        ax.set_yticks(positions)
    
        ax.grid(
            axis="x",
            alpha=0.15,
        )
    
        ax.spines[["top", "right"]].set_visible(False)
    
    axes[0].set_yticklabels(stats_df["cluster"])
    axes[0].tick_params(axis="y", labelleft=True)
    axes[1].tick_params(axis="y", labelleft=False)
    
    # Symmetric x-axis for log2FC
    max_fc = max(
        np.max(np.abs(stats_df["log2FC"])),
        fc_threshold,
        0.5,
    )
    
    axes[0].set_xlim(
        -max_fc * 1.15,
        max_fc * 1.15,
    )
    
    # Cliff's delta range
    axes[1].set_xlim(-1.05, 1.05)
    
    axes[0].invert_yaxis()
    
    fig.suptitle(
        f"{gene} | RCC vs Healthy",
        fontsize=15,
        fontweight="bold",
    )
    
    # Legend
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=malignant_color,
            markeredgecolor="none",
            markersize=8,
            label=f"log₂FC ≥ +{fc_threshold:g}",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=neutral_color,
            markeredgecolor="none",
            markersize=8,
            label=f"|log₂FC| < {fc_threshold:g}",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=healthy_color,
            markeredgecolor="none",
            markersize=8,
            label=f"log₂FC ≤ −{fc_threshold:g}",
        ),
    ]
    
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
    
    plt.show()

    return stats_df



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from matplotlib.lines import Line2D


def compare_expression_across_clusters_v2(
    df1,
    df2,
    gene,
    cluster_pairs=None,
    cluster_col="CellType",
    malignant_color="#C44E52",
    healthy_color="#4C72B0",
    neutral_color="#999999",
    non_significant_color="#BDBDBD",
    pseudocount=1e-6,
    figsize=(9, 6),
    sort_by=None,
    save_path=None,
    fc_threshold=1.0,
    significance_threshold=0.01,
    include_global=True,
):
    """
    Compare predicted gene expression between malignant and healthy
    scRNA-seq datasets.

    Input expression must be log2(CPM + 1).

    Comparisons
    -----------
    1. Matched cell clusters defined by cluster_pairs.
    2. Global comparison: ALL cells in df1 vs ALL cells in df2.
       Each cell has equal weight, preserving natural cluster proportions.

    Statistics
    ----------
    log2FC:
        log2 ratio of mean CPM (malignant / healthy).

    Cliff's delta:
        Distributional effect size calculated from Mann-Whitney U.

    p_value:
        Two-sided Mann-Whitney U test on log2CPM values.

    p_adj_BH:
        Benjamini-Hochberg adjusted p-values across all comparisons,
        including the global comparison.

    IMPORTANT
    ---------
    These are cell-level tests. They do not correct for batch effects
    or account for biological sample-level dependencies.

    Parameters
    ----------
    df1, df2 : pandas.DataFrame
        Malignant and healthy expression datasets.

    gene : str
        Gene to compare.

    cluster_pairs : list of tuples or None
        (malignant_cluster, healthy_cluster).

    cluster_col : str
        Column containing cell-type annotations.

    include_global : bool
        Whether to include comparison of all cells in both datasets.

    significance_threshold : float
        BH-adjusted p-value threshold for statistical significance.

    fc_threshold : float
        Absolute log2FC threshold for color classification.

    sort_by : str or None
        None, "log2FC", "abs_log2FC",
        "Cliffs_delta", "abs_Cliffs_delta",
        "p_value", "p_adj_BH".

    save_path : str or None
        Optional path to save figure.

    Returns
    -------
    pandas.DataFrame
        Per-comparison statistics.
    """

    # -------------------------------------------------
    # Default cluster pairs
    # -------------------------------------------------

    if cluster_pairs is None:
        cluster_pairs = [
            ("Malignant cells", "Proximal tubule cells")
        ]

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    for name, df in [("df1", df1), ("df2", df2)]:

        if gene not in df.columns:
            raise ValueError(f"Gene '{gene}' not found in {name}")

    if cluster_pairs:
        for name, df in [("df1", df1), ("df2", df2)]:
            if cluster_col not in df.columns:
                raise ValueError(
                    f"Column '{cluster_col}' not found in {name}"
                )

    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("pseudocount must be positive.")

    if not 0 <= significance_threshold <= 1:
        raise ValueError(
            "significance_threshold must be between 0 and 1."
        )

    if fc_threshold < 0:
        raise ValueError("fc_threshold must be non-negative.")

    # -------------------------------------------------
    # Extract expression values
    # -------------------------------------------------

    def extract_expression(df, cluster=None, use_all=False):

        if use_all:
            values = df[gene].to_numpy(dtype=np.float64)

        else:
            values = df.loc[
                df[cluster_col] == cluster,
                gene
            ].to_numpy(dtype=np.float64)

        values = values[np.isfinite(values)]

        # Clip negative predictions to zero
        values = np.maximum(values, 0)

        return values

    # -------------------------------------------------
    # Calculate statistics for one comparison
    # -------------------------------------------------

    def calculate_comparison(x, y, label, malignant_name, healthy_name,
                              comparison_type):

        if len(x) == 0 or len(y) == 0:
            raise ValueError(
                f"No valid cells for comparison: {label}"
            )

        # Restore linear CPM
        x_cpm = np.expm1(x * np.log(2))
        y_cpm = np.expm1(y * np.log(2))

        mean_malignant = np.mean(x_cpm)
        mean_healthy = np.mean(y_cpm)

        # log2 fold change
        log2fc = np.log2(
            (mean_malignant + pseudocount)
            / (mean_healthy + pseudocount)
        )

        # Mann-Whitney U test
        test = mannwhitneyu(
            x,
            y,
            alternative="two-sided",
            method="asymptotic",
        )

        u_stat = test.statistic
        p_value = test.pvalue

        # Cliff's delta
        delta = (
            2.0 * u_stat / (len(x) * len(y))
        ) - 1.0

        return {
            "cluster": label,
            "comparison_type": comparison_type,
            "malignant_cluster": malignant_name,
            "healthy_cluster": healthy_name,

            "n_malignant": len(x),
            "n_healthy": len(y),

            "mean_CPM_malignant": mean_malignant,
            "mean_CPM_healthy": mean_healthy,

            "median_log2CPM_malignant": np.median(x),
            "median_log2CPM_healthy": np.median(y),

            "pct_expressing_malignant": np.mean(x > 0) * 100,
            "pct_expressing_healthy": np.mean(y > 0) * 100,

            "log2FC": log2fc,
            "Cliffs_delta": delta,
            "p_value": p_value,
        }

    results = []

    # -------------------------------------------------
    # 1. Matched cluster comparisons
    # -------------------------------------------------

    for malignant_cluster, healthy_cluster in cluster_pairs:

        x = extract_expression(
            df1,
            cluster=malignant_cluster
        )

        y = extract_expression(
            df2,
            cluster=healthy_cluster
        )

        if malignant_cluster == healthy_cluster:
            label = malignant_cluster
        else:
            label = (
                f"{malignant_cluster}\n"
                f"vs {healthy_cluster}"
            )

        results.append(
            calculate_comparison(
                x=x,
                y=y,
                label=label,
                malignant_name=malignant_cluster,
                healthy_name=healthy_cluster,
                comparison_type="cluster",
            )
        )

    # -------------------------------------------------
    # 2. Global comparison: ALL cells vs ALL cells
    # -------------------------------------------------

    if include_global:

        x_all = extract_expression(
            df1,
            use_all=True
        )

        y_all = extract_expression(
            df2,
            use_all=True
        )

        results.append(
            calculate_comparison(
                x=x_all,
                y=y_all,
                label="All cells\n(all clusters)",
                malignant_name="All malignant cells",
                healthy_name="All healthy cells",
                comparison_type="global",
            )
        )

    if not results:
        raise ValueError("No comparisons were requested.")

    stats_df = pd.DataFrame(results)

    # -------------------------------------------------
    # 3. Benjamini-Hochberg correction
    # -------------------------------------------------

    def bh_adjust(p_values):

        p_values = np.asarray(p_values, dtype=float)

        n = len(p_values)
        order = np.argsort(p_values)

        ranked_p = p_values[order]

        adjusted = (
            ranked_p * n / np.arange(1, n + 1)
        )

        # Enforce monotonicity
        adjusted = np.minimum.accumulate(
            adjusted[::-1]
        )[::-1]

        adjusted = np.clip(adjusted, 0, 1)

        result = np.empty(n)
        result[order] = adjusted

        return result

    stats_df["p_adj_BH"] = bh_adjust(
        stats_df["p_value"].to_numpy()
    )

    stats_df["significant_BH"] = (
        stats_df["p_adj_BH"] <= significance_threshold
    )

    # -------------------------------------------------
    # 4. Optional sorting
    # -------------------------------------------------

    allowed_sort = {
        None,
        "log2FC",
        "abs_log2FC",
        "Cliffs_delta",
        "abs_Cliffs_delta",
        "p_value",
        "p_adj_BH",
    }

    if sort_by not in allowed_sort:
        raise ValueError(
            f"sort_by must be one of {allowed_sort}"
        )

    if sort_by is not None:

        if sort_by.startswith("abs_"):

            column = sort_by[4:]

            stats_df = stats_df.iloc[
                np.argsort(
                    -np.abs(stats_df[column].to_numpy()),
                    kind="stable"
                )
            ]

        elif sort_by in {"p_value", "p_adj_BH"}:

            stats_df = stats_df.sort_values(
                sort_by,
                ascending=True
            )

        else:

            stats_df = stats_df.iloc[
                np.argsort(
                    -stats_df[sort_by].to_numpy(),
                    kind="stable"
                )
            ]

        stats_df = stats_df.reset_index(drop=True)

    # -------------------------------------------------
    # 5. Determine colors
    # -------------------------------------------------

    colors = []

    for _, row in stats_df.iterrows():

        # Non-significant comparisons are ALWAYS gray
        if row["p_adj_BH"] > significance_threshold:

            color = non_significant_color

        # Significant comparisons: classify by log2FC
        elif row["log2FC"] >= fc_threshold:

            color = malignant_color

        elif row["log2FC"] <= -fc_threshold:

            color = healthy_color

        else:

            color = neutral_color

        colors.append(color)

    # -------------------------------------------------
    # 6. Plot
    # -------------------------------------------------

    n = len(stats_df)
    positions = np.arange(n)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    metrics = [
        ("log2FC", "log₂ fold change (mean CPM)"),
        ("Cliffs_delta", "Cliff's delta"),
    ]

    for ax, (metric, xlabel) in zip(axes, metrics):

        values = stats_df[metric].to_numpy()

        ax.axvline(
            0,
            color="gray",
            linestyle="--",
            linewidth=1,
            zorder=1,
        )

        # FC threshold lines
        if metric == "log2FC":

            ax.axvline(
                fc_threshold,
                color=malignant_color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.7,
            )

            ax.axvline(
                -fc_threshold,
                color=healthy_color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.7,
            )

        for i, (value, color) in enumerate(
            zip(values, colors)
        ):

            ax.plot(
                [0, value],
                [i, i],
                color=color,
                linewidth=2,
                alpha=0.55,
                zorder=2,
            )

            ax.scatter(
                value,
                i,
                s=85,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )

            # Show BH-adjusted p-value above each comparison
            # ONLY in the log2FC panel
            if metric == "log2FC":

                p_adj = stats_df.iloc[i]["p_adj_BH"]

                ax.annotate(
                    f"BH={p_adj:.2g}",
                    xy=(value, i),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#333333",
                    annotation_clip=False,
                )

        ax.set_xlabel(xlabel)

        ax.set_yticks(positions)

        ax.grid(
            axis="x",
            alpha=0.15,
        )

        ax.spines[["top", "right"]].set_visible(False)

    # Cluster labels
    axes[0].set_yticklabels(stats_df["cluster"])

    axes[0].tick_params(
        axis="y",
        labelleft=True
    )

    axes[1].tick_params(
        axis="y",
        labelleft=False
    )

    # Allow room for p-value annotations above the first row
    axes[0].set_ylim(n - 0.4, -0.8)

    # Symmetric log2FC axis
    max_fc = max(
        np.max(np.abs(stats_df["log2FC"])),
        fc_threshold,
        0.5,
    )

    axes[0].set_xlim(
        -max_fc * 1.15,
        max_fc * 1.15,
    )

    # Cliff's delta range
    axes[1].set_xlim(-1.05, 1.05)

    fig.suptitle(
        f"{gene} | RCC vs Healthy",
        fontsize=15,
        fontweight="bold",
    )

    # -------------------------------------------------
    # 7. Legend
    # -------------------------------------------------

    legend_elements = [

        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=malignant_color,
            markeredgecolor="none",
            markersize=8,
            label=f"Significant: log₂FC ≥ +{fc_threshold:g}",
        ),

        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=neutral_color,
            markeredgecolor="none",
            markersize=8,
            label=f"Significant: |log₂FC| < {fc_threshold:g}",
        ),

        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=healthy_color,
            markeredgecolor="none",
            markersize=8,
            label=f"Significant: log₂FC ≤ −{fc_threshold:g}",
        ),

        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=non_significant_color,
            markeredgecolor="none",
            markersize=8,
            label=f"Not significant: BH p > {significance_threshold:g}",
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=8,
    )

    fig.tight_layout(
        rect=[0, 0.12, 1, 0.95]
    )

    # -------------------------------------------------
    # 8. Save and display
    # -------------------------------------------------

    if save_path is not None:

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    return stats_df
