"""Plotting and visualization functions for the mstcvi package."""

import numpy as np
import matplotlib.pyplot as plt

from .mst_utils import get_euclidean_mst, mst_partition_edges
from .indices import treelhouette_samples
from .style import CUT_EDGE_COLOR, MEAN_LINE_COLOR, cluster_color, cluster_marker


def plot_scatter(X, labels, *, ax=None, show_mst=False, use_markers=True, point_size=30, 
                M=0, mst_euclid_kwargs=None):
    """Scatter plot of 2D points with the MST overlaid.

    Points are coloured by cluster; MST edges are drawn underneath,
    with within-cluster ("kept") edges and between-cluster ("cut")
    edges in different styles.

    Parameters
    ----------
    X : ndarray, shape (n, 2)
        Input points. Must be 2D -- this is a 2D visual, not a
        dimensionality-reduction tool.
    labels : ndarray, shape (n,)
    ax : matplotlib.axes.Axes, optional
    M, mst_euclid_kwargs
        Forwarded to euclidean_mst.
    point_size : float, default=30
        Marker size, forwarded to ax.scatter.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if X.shape[1] != 2:
        raise ValueError(f"plot_scatter needs 2D points, got shape {X.shape}")


    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=(5, 5))


    if show_mst:
        mst_dist, mst_index = get_euclidean_mst(X, M=M, **(mst_euclid_kwargs or {}))
        same_cluster, edge_labels, _ = mst_partition_edges(mst_dist, mst_index, labels)

        for i, (u, v) in enumerate(mst_index):
            kept = same_cluster[i]
            ax.plot(
                X[[u, v], 0], X[[u, v], 1],
                color=cluster_color(edge_labels[i]) if kept else CUT_EDGE_COLOR,
                alpha=0.6 if kept else 1.0,
                linewidth=2.0 if kept else 4.0,
                linestyle="-" if kept else "--",
                zorder=1,
            )


    for pos, lab in enumerate(np.unique(labels)):
        mask = labels == lab
        ax.scatter(
            X[mask, 0], X[mask, 1],
            s=point_size, color=cluster_color(pos),
            marker=cluster_marker(pos) if use_markers else "o",
            zorder=2, label=f"cluster {lab}", alpha=0.8
        )

    ax.set_aspect("equal", adjustable="datalim")
    if own_figure:
        plt.tight_layout()
    
    return ax


def plot_treelhouette(X, labels, *, ax=None, show_cluster_scores=True, M=0, **mst_euclid_kwargs):
    """Silhouette-style bar plot of treelhouette lengths.

    One bar per within-cluster MST edge, sorted decreasingly within
    each cluster and a vertical line showing treelhouette_score.

    Parameters
    ----------
    X, labels, M, mst_euclid_kwargs
        See :func:`mstvi.indices.treelhouette_samples`.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    mst_dist, mst_index = get_euclidean_mst(X, M=M, **(mst_euclid_kwargs or {}))
    same_cluster, edge_labels, _ = mst_partition_edges(mst_dist, mst_index, labels)

    t = treelhouette_samples(X, labels, M=M, **mst_euclid_kwargs)
    score = float(np.mean(t))

    within_edge_labels = edge_labels[same_cluster]

    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=(10, 5))

    y0 = 0

    for lab in np.unique(within_edge_labels):
        mask = within_edge_labels == lab
        vals = np.sort(t[mask])[::1]
        n_edges = len(vals)

        y = np.arange(y0, y0 + n_edges)
        ax.barh(y, vals, height=1.0, align="edge", color=cluster_color(lab))

        total_edges = len(within_edge_labels)
        gap = max(1, round(total_edges * 0.03))

        y0_start = y0
        y0 += n_edges + gap

        if show_cluster_scores:
            cluster_mean = float(np.mean(vals))
            label_text = f"{lab}: {n_edges} | {cluster_mean:.2f}"
            y_top_of_cluster = y0_start + n_edges

            ax.text(
                1.02,
                y_top_of_cluster,
                label_text,
                transform=ax.get_yaxis_transform(),
                verticalalignment="top",
                horizontalalignment="left",
                fontsize=8,
                color=cluster_color(lab)
            )
            
    ax.axvline(score, color=MEAN_LINE_COLOR, linewidth=1.5, linestyle="--")
    ax.set_xlabel("treelhouette length $t_i$", fontsize=9)
    if show_cluster_scores:
        ax.text(
            1.02,
            -1.00,
            r"$C_j$: $n_j$ | avg $t_i$",
            transform=ax.get_yaxis_transform(),
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=8,
        )
    ax.set_yticks([])
    ax.set_title(f"treelhouette_score = {score:.3f}", fontsize=11)

    if own_figure:
        plt.tight_layout(rect=[0, 0, 0.82, 1.0])

    return ax