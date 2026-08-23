import numpy as np

from .mst_utils import get_euclidean_mst, mst_partition_edges


def treelhouette_samples(X, labels, *, M=0, **mst_euclid_kwargs):
    """Compute the Treelhouette Coefficient for each within-cluster MST edge.

    For every MST edge lying within a cluster, define:

    - ``a_i``: the weight of the i-th edge,
    - ``b_i``: the smallest weight among the MST edges connecting that
      edge's cluster to any other cluster,

    and the treelhouette length::

        t_i = 1 - a_i / b_i   if a_i < b_i
        t_i = 0               if a_i == b_i
        t_i = b_i / a_i - 1   if a_i > b_i

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input points.
    labels : ndarray, shape (n,)
        Cluster label of each point.
    M : int, default=0
        Forwarded to ``quitefastmst.mst_euclid`` (mutual reachability
        smoothing factor; M=0 gives the plain Euclidean MST).
    **mst_euclid_kwargs
            Extra kwargs forwarded to ``quitefastmst.mst_euclid``.

    Returns
    -------
    t : ndarray, shape (n - k,)                                          NOTE to nie jest n - k!!!
        Treelhouette length of each within-cluster MST edge.
    """

    X, labels = _validate_params(X, labels)

    k = len(np.unique(labels))

    mst_dist, mst_index = get_euclidean_mst(
        X, M=M, **mst_euclid_kwargs
        )

    same_cluster, edge_labels, cut_endpoint_labels = mst_partition_edges(
        mst_dist, mst_index, labels
        )

    a = mst_dist[same_cluster]

    b_per_cluster = np.full(k, np.inf)

    cut_mask = cut_endpoint_labels[:, 0] != -1
    cut_weights = mst_dist[cut_mask]

    for col in (0, 1):
        cluster_ids = cut_endpoint_labels[cut_mask, col]
        np.minimum.at(b_per_cluster, cluster_ids, cut_weights)

    b = b_per_cluster[edge_labels[same_cluster]]

    t = np.zeros_like(a)

    cond_lt = a < b
    cond_gt = a > b

    t[cond_lt] = 1.0 - (a[cond_lt] / b[cond_lt])
    t[cond_gt] = (b[cond_gt] / a[cond_gt]) - 1.0

    return t


def treelhouette_score(X, labels, *, M=0, **mst_euclid_kwargs):
    """Compute the mean Treelhouette for each within-cluster MST edge.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input points.
    labels : ndarray, shape (n,)
        Cluster label of each point.
    M : int, default=0
        Forwarded to ``quitefastmst.mst_euclid`` (mutual reachability
        smoothing factor; M=0 gives the plain Euclidean MST).
    **mst_euclid_kwargs
            Extra kwargs forwarded to ``quitefastmst.mst_euclid``.

    Returns
    -------
    treelhouette : float
        Mean Treelhouette Coefficient.

    """

    t = treelhouette_samples(X, labels, M=M, **mst_euclid_kwargs)

    return float(np.mean(t))




def _validate_params(X, labels):
    """Validate and normalise the inputs shared by MST-based indices.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input points.
    labels : ndarray, shape (n,)
        Cluster label of each point.

    Returns
    -------
    X : ndarray, shape (n, d), dtype float64
    labels : ndarray, shape (n,)

    Raises
    ------
    TypeError
        If X or labels cannot be converted to a numeric array.
    ValueError
        If shapes are inconsistent, values are non-finite, or the
        number of clusters is outside the valid range [2, n-1].
    """
    # --- X ---
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array of shape (n, d), got ndim={X.ndim}")

    n, d = X.shape
    if d < 1:
        raise ValueError("X must have at least 1 feature")
    if n < 3:
        raise ValueError(f"X must contain at least 3 points, got n={n}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")

    # --- labels ---
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be a 1D array of shape (n,), got ndim={labels.ndim}")
    if labels.shape[0] != n:
        raise ValueError(f"labels length ({labels.shape[0]}) must match number of points in X ({n})")
    if np.issubdtype(labels.dtype, np.floating) and np.any(np.isnan(labels)):
        raise ValueError("labels contains NaN values")

    k = len(np.unique(labels))
    if k < 2 or k > n - 1:
        raise ValueError(f"number of clusters k must satisfy 2 <= k <= n-1 (n={n}), got k={k}")

    return X, labels
