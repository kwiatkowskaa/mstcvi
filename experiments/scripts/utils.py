import pickle
import numpy as np
import matplotlib.pyplot as plt
from mstcvi import plot_treelhouette, plot_scatter

from scripts.datasets import iter_benchmark_datasets


def plot_treelhouette_summary(X, labels, *, show_mst=True, use_markers=True, point_size=20, 
                              title=None, figsize=(12, 5), M=0, **mst_euclid_kwargs):

    """Two-panel diagnostic figure for one partition of 2D data.

    Left: points coloured by cluster with the MST overlaid (within-cluster edges in
    the cluster colour, cut edges dashed).
    Right: treelhouette profile, one bar per within-cluster MST edge, sorted inside each
    cluster; the dashed line is the overall score and the labels on the right read
    "cluster: n_edges | mean t".
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left
    plot_scatter(X, labels, ax=ax1, show_mst=show_mst, use_markers=use_markers,
                point_size=point_size, M=M, **mst_euclid_kwargs)
    ax1.set_title("MST partition", fontsize=11)

    # Right
    plot_treelhouette(X, labels, ax=ax2, show_cluster_scores=True, M=M, **mst_euclid_kwargs)

    content_right = 0.84

    if title:
        fig.suptitle(title, fontsize=13, x=content_right / 2, y=0.98)

    plt.tight_layout(rect=[0, 0, content_right, 0.98])

    return fig, (ax1, ax2)



def plot_benchmark_treelhouette_summaries(batteries, data_path=None, result_path=None, *, 
                                          skip_large=True, point_size=30):
    """
    Iterates over benchmark datasets in the specified battery, loads precomputed 
    optimization results from pickle files, and renders the Treelhouette summary 
    plot for each valid (dataset, k) configuration.
    """

    if isinstance(batteries, str):
        batteries = [batteries]

    for ds in iter_benchmark_datasets(batteries, data_path=data_path, skip_large=skip_large):

        if ds.X.shape[1] == 2:

            ks = sorted({len(np.unique(ref)) for ref in ds.reference_labels})

            for k in ks:
                run_id = f"{ds.battery}__{ds.name}__k{k}"
                file_path = result_path / f"{run_id}.pkl"

                if not file_path.exists():
                    print(f"[missing] {run_id} -- no saved result, skipping")
                    continue

                with open(file_path, "rb") as f:
                    data = pickle.load(f)

                best_labels = data["result"]["best_labels"]

                print(f"\n=== {ds.battery}/{ds.name} n={ds.X.shape[0]} d={ds.X.shape[1]} k={k} ARI={data["result"]['best_ari']:.3f} ===")

                fig, axes = plot_treelhouette_summary(ds.X, best_labels, point_size=30)
                plt.show()
                plt.close(fig)