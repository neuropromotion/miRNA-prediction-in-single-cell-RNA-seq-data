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
