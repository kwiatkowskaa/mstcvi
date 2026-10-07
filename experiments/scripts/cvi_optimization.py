"""Candidate generation and tabu-search hill climbing.

Reproduces the methodology of Gagolewski, Bartoszuk & Cena (2022),
"Are cluster validity measures (in)valid?" -- Algorithm 1 (Sec. 3.3)
and the candidate-solution recipe (Sec. 3.5).
"""

import warnings
import time

import numpy as np

import genieclust

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from mstcvi import get_euclidean_mst


def _random_partition(n, k, rng):
    """A uniformly random k-partition guaranteed to use every one of
    the k labels at least once.
    """
    labels = np.empty(n, dtype=int)
    labels[:k] = np.arange(k)
    labels[k:] = rng.integers(0, k, size=n - k)
    rng.shuffle(labels)
    return labels


def generate_initial_partitions(X, k_values, n_init=5, n_random=5, random_state=None):
    """
    Generates initial candidate partitions (C_1, ..., C_m).

    Any single candidate generator that raises an exception on a given
    dataset or that returns a partition with the wrong number of distinct clusters, 
    is skipped with a warning.
    
    Parameters
    ----------
    X : array-like
        The dataset.
    k_values : int or list of int
        List of cluster sizes (k) to generate partitions for.
    n_init : int, default=5
        Number of differently-seeded KMeans / GaussianMixture / spectral
        clustering candidates.
    n_random : int, default=5
        Number of uniformly random partitions per k.
    random_state : int, optional
        Seeds the randomised methods and the random partitions.    
        
    Returns
    -------
    candidates : list of ndarray
        List containing generated label vectors (partitions).
    candidate_names : list of str
        Names of candidates.
    """
    candidates = []
    candidate_names = []
    rng = np.random.default_rng(random_state)

    if isinstance(k_values, (int, np.integer)):
        k_values = [k_values]
    
    for k in k_values:
        
        models = {}
        
        # KMeans
        for seed in range(n_init):
            models[f"KMeans_{seed}"] = KMeans(n_clusters=k, random_state=seed, n_init=10)

        # GM
        for seed in range(n_init):
            models[f"GaussianMixture_{seed}"] = GaussianMixture(n_components=k, random_state=seed)
            
        # AgglomerativeClustering
        for linkage in ["single", "average", "complete", "ward"]:
            models[f"AC_{linkage}"] = AgglomerativeClustering(n_clusters=k, linkage=linkage)

        # Genie
        for g in [0.1, 0.3, 0.5, 0.7, 0.9]:
            models[f"Genie_G{g}"] = genieclust.Genie(n_clusters=k, gini_threshold=g)

        # SpectralClustering
        for affinity in ["nearest_neighbors", "rbf"]:
            for seed in range(n_init):
                models[f"Spectral_{affinity}_{seed}"] = SpectralClustering(
                    n_clusters=k, assign_labels="kmeans", 
                    affinity=affinity, random_state=seed)

        # Birch
        for threshold in [0.3, 0.5, 0.7]:
            models[f"Birch_t{threshold}"] = Birch(n_clusters=k, threshold=threshold, branching_factor=50)

        # Uniformly random partitions
        for _ in range(n_random):
            candidates.append(_random_partition(X.shape[0], k, rng))
            candidate_names.append("Random")


        for name, model in models.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    labels = model.fit_predict(X)
                    
                found_k = len(np.unique(labels))
                
                if found_k != k:
                    warnings.warn(f"candidate '{name}' produced {found_k} clusters, expected k={k} -- skipped", RuntimeWarning,)
                    continue

                candidates.append(labels)
                candidate_names.append(f"{name}_k{k}" if len(k_values) > 1 else name)

            except Exception as exc:
                warnings.warn(f"candidate generator '{name}' raised {exc!r} -- skipped", RuntimeWarning)
                
        
    seen_hashes = set()
    unique_candidates = []
    unique_candidate_names = []

    for labels, name in zip(candidates, candidate_names):

        _, canonical_labels = np.unique(labels, return_inverse=True)
        label_bytes = canonical_labels.tobytes()

        if label_bytes not in seen_hashes:
            seen_hashes.add(label_bytes)
            unique_candidates.append(labels)
            unique_candidate_names.append(name)

    return unique_candidates, unique_candidate_names



def optimize_cvi_tabu_search(
        X, I_func, candidate_partitions, k, P=250,
        reference_labels_list=None, candidate_names=None
        ):
    """
    Algorithm 1: Finding optimal partitions (w.r.t. a given CVI).
    Reproduction of algorithm used in 
    "Are cluster validity measures (in) valid?" (Sec. 3.3).
    
    Parameters
    ----------
    X : array-like
        The dataset.
    I_func : callable
        A function calculating the CVI: `score = I_func(X, labels)`.
        Higher score mean a better partition.
    candidate_partitions : list of ndarray
        Initial candidate solutions (C_1, ..., C_m).
    k : int
        Number of clusters.
    P : int
        Patience parameter. Upper bound for iterations without global improvement.
    reference_labels_list : list of ndarray, optional
        Reference partition(s) to track ARI. If omitted, ARI is not computed
    candidate_names : list of str, optional
        Names of candidates. Defaults to "C_1", "C_2", ... if not given.
        
    Returns
    -------
    dict
        best_labels, best_score, best_ari, best_candidate_name,
        tabu_list_size, n_evaluations, elapsed_seconds, history
        (one entry per starting candidate: start/end I and ARI).
    """
    t_start = time.perf_counter()
    n_samples = X.shape[0]

    candidate_partitions = list(candidate_partitions)
    candidate_names = (
        list(candidate_names) if candidate_names is not None
        else [f"C_{i + 1}" for i in range(len(candidate_partitions))]
    )

    # adding reference labels to candidates list
    if reference_labels_list is not None:
        for ref_idx, ref in enumerate(reference_labels_list):
            ref = np.asarray(ref)
            if len(np.unique(ref)) == k:
                candidate_partitions.append(ref)
                candidate_names.append(f"reference_{ref_idx}")
    
    m = len(candidate_partitions)
    mst_dist, mst_index = get_euclidean_mst(X)
    
    scored_candidates = []
    for name, C_cand in zip(candidate_names, candidate_partitions):
        score = I_func(X, C_cand, mst_dist=mst_dist, mst_index=mst_index)
        scored_candidates.append((score, np.array(C_cand), name))
        
    scored_candidates.sort(key=lambda item: item[0], reverse=True) # I(C_1) >= ... >= I(C_m)
    
    T = set() # 1. T - "tabu" list

    C_star_name = scored_candidates[0][2]
    C_star = scored_candidates[0][1].copy()
    I_C_star = scored_candidates[0][0] # 2. C* = C_1

    history = []
    n_evaluations = 0
        
    # 3. for C = C_1, C_2, ..., C_m do:
    for m_idx, (I_C, C, name) in enumerate(scored_candidates):
        start_ari = ari(reference_labels_list, C)
        
        C_current = C.copy()
        I_current = I_C

        p = 1 # 3.1. p = 1
        
        while True:
            print(f"\rCandidate {m_idx + 1:>3}/{m:<3} | Patience {p:>4}/{P:<4}   ", end="", flush=True)

            C_plus = None # 3.2. C+ = empty
            I_C_plus = -np.inf

            cluster_counts = np.bincount(C_current, minlength=k)
                        
            # 3.3. for each C' \in NEIGHBOURS(C) do:
            for i in range(n_samples):
                orig_label = C_current[i]

                if cluster_counts[orig_label] <= 1:
                    continue
                
                for target_label in range(k):
                    if target_label == orig_label:
                        continue
                    
                    C_prime = C_current.copy()
                    C_prime[i] = target_label
                    C_prime_tuple = tuple(C_prime)
                    
                    # 3.3.1. if C' \notin T and I(C') > I(C+), then C+ = C'
                    if C_prime_tuple not in T:
                        n_evaluations += 1
                        I_C_prime = I_func(X, C_prime, mst_dist=mst_dist, mst_index=mst_index)
                        if I_C_prime > I_C_plus:
                            I_C_plus = I_C_prime
                            C_plus = C_prime
                            
            # 3.4. if C+ == empty then continue to step 3
            if C_plus is None:
                break
                
            # 3.5. T = T U {C+} (never visit C+ again)
            T.add(tuple(C_plus))
            
            # 3.6. C = C+
            C_current = C_plus
            I_current = I_C_plus
            
            # 3.7. if I(C) > I(C*), then C* = C, else p = p + 1;
            if I_current > I_C_star:
                C_star_name = name
                C_star = C_current.copy()
                I_C_star = I_C_plus
                p = 1
            else:
                p += 1
            
            # 3.8. if p <= P, then go to step 3.2;
            if p > P:
                break

        end_ari = ari(reference_labels_list, C_current)

        history.append({
            "candidate_name": name,
            "start_I": I_C, "start_ari": start_ari,
            "end_I": I_C_plus, "end_ari": end_ari
        })

    best_ari = ari(reference_labels_list, C_star)
    elapsed = time.perf_counter() - t_start

    # 4. return C*

    label_width = 24

    print("\r" + " " * 40 + "\r", end="")
    print("Optimization complete!")
    print(f"    {'Best I(C*)':<{label_width}} = {I_C_star:.6f}")
    if best_ari is not None:
        print(f"    {'Best ARI(C*)':<{label_width}} = {best_ari:.6f}")
    print(f"    {'Best candidate name':<{label_width}} = {C_star_name}")
    print(f"    {'Candidates explored':<{label_width}} = {m}")
    print(f"    {'Tabu set size (|T|)':<{label_width}} = {len(T)}")
    print(f"    {'I_func evaluations':<{label_width}} = {n_evaluations:_}")
    print(f"    {'Elapsed time':<{label_width}} = {elapsed:.1f}s")

    return {
        "best_labels": C_star,
        "best_score": I_C_star,
        "best_ari": best_ari,
        "best_candidate_name": C_star_name,
        "tabu_list_size": len(T),
        "n_evaluations": n_evaluations,
        "elapsed_seconds": elapsed,
        "history": history,
    }


def ari(reference_labels_list, labels):
    """Best ARI of `labels` against any of the given reference partitions.
    """
    if reference_labels_list is None:
        return None
    return max(max(adjusted_rand_score(ref, labels), 0.0) for ref in reference_labels_list)