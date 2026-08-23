import numpy as np
import quitefastmst


def get_euclidean_mst(X, *, M=0, **mst_euclid_kwargs):
    """Compute the Euclidean MST of X using quitefastmst.mst_euclid.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input points.
    M : int, default=0
        Mutual reachability smoothing factor;
        M=0 gives the plain Euclidean MST.
    **mst_euclid_kwargs
        Extra kwargs forwarded to ``quitefastmst.mst_euclid``.

    Returns
    -------
    mst_dist : ndarray, shape (n - 1,)
        Weight of each MST edge.
    mst_index : ndarray, shape (n - 1, 2)
        Endpoints of each MST edge.
    """
    result = quitefastmst.mst_euclid(X, M=M, **mst_euclid_kwargs)
    mst_dist, mst_index = result[0], result[1]
    return np.asarray(mst_dist), np.asarray(mst_index)


def mst_partition_edges(mst_dist, mst_index, labels):
    """Split MST edges into within-cluster and between-cluster edges.

    Parameters
    ----------
    mst_dist : ndarray, shape (n - 1,)
        Weight of each MST edge.
    mst_index : ndarray, shape (n - 1, 2)
        Endpoints of each MST edge.
    labels : ndarray, shape (n,)
        Cluster label of each point.

    Returns
    -------
    same_cluster : bool ndarray, shape (n - 1,)
        True where an edge's two endpoints share a cluster.
    edge_labels : int ndarray, shape (n - 1,)
        Cluster id of each within-cluster edge, -1 for cut edges.
    cut_endpoint_labels : int ndarray, shape (n - 1, 2)
        The two cluster ids a cut edge connects, (-1, -1) for
        within-cluster edges.
    """
    labels = np.asarray(labels)
    unique_labels, contig_labels = np.unique(labels, return_inverse=True)

    u, v = mst_index[:, 0], mst_index[:, 1]
    same_cluster = contig_labels[u] == contig_labels[v]

    edge_labels = np.full(mst_dist.shape[0], -1, dtype=int)
    edge_labels[same_cluster] = contig_labels[u[same_cluster]]

    cut_endpoint_labels = np.full((mst_dist.shape[0], 2), -1, dtype=int)
    cut_mask = ~same_cluster
    cut_endpoint_labels[cut_mask, 0] = contig_labels[u[cut_mask]]
    cut_endpoint_labels[cut_mask, 1] = contig_labels[v[cut_mask]]

    return same_cluster, edge_labels, cut_endpoint_labels