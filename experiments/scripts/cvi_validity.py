import numpy as np

import genieclust

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from mstcvi import get_euclidean_mst


def generate_initial_partitions(X, k_values, n_init=5, random_state=None):
    """
    Generates initial candidate partitions (C_1, ..., C_m).
    
    Parameters
    ----------
    X : array-like
        The dataset.
    k_values : int or list of int
        List of cluster sizes (k) to generate partitions for.
        
    Returns
    -------
    candidates : list of ndarray
        List containing generated label vectors (partitions).
    candidate_names : list of str
        Names of candidates.
    """
    candidates = []
    candidate_names = []

    if isinstance(k_values, (int, np.integer)):
        k_values = [k_values]
    
    for k in k_values:
        # KMeans
        for seed in range(n_init):
            km = KMeans(n_clusters=k, random_state=seed)
            candidates.append(km.fit_predict(X))
            candidate_names.append(f"KMeans_{seed}")

        # GM
        gmm = GaussianMixture(n_components=k, random_state=random_state)
        candidates.append(gmm.fit_predict(X))
        candidate_names.append("GaussianMixture")
            
        # AgglomerativeClustering
        for linkage in ["single", "average", "complete", "ward"]:
            agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            candidates.append(agg.fit_predict(X))
            candidate_names.append(f"AC_{linkage}")

        # Genie
        for g in [0.1, 0.3, 0.5, 0.7, 0.9]:
            genie = genieclust.Genie(n_clusters=k, gini_threshold=g)
            candidates.append(genie.fit_predict(X))
            candidate_names.append(f"Genie_G{g}")

        # SpectralClustering
        spectral = SpectralClustering(
            n_clusters=k, assign_labels="kmeans", 
            affinity="nearest_neighbors", random_state=random_state
            )
        candidates.append(spectral.fit_predict(X))
        candidate_names.append("Spectral")

        # # Birch
        # birch = Birch(n_clusters=k, threshold=0.5, branching_factor=50)
        # candidates.append(birch.fit_predict(X))
        # candidate_names.append(f"Birch")
        
    return candidates, candidate_names


def optimize_cvi_tabu_search(
        X, I_func, candidate_partitions, k, P=250,
        reference_labels_list=None, candidate_names=None
        ):
    """
    Algorithm 1: Finding optimal partitions (w.r.t. a given CVI).
    Reproduction of algorithm used in "Are cluster validity measures (in) valid?".
    
    Parameters
    ----------
    X : array-like
        The dataset.
    I_func : callable
        A function calculating the CVI: `score = I_func(X, labels)`.
        Higher score mean a better partition.
    candidate_partitions : list of ndarray
        Initial candidate solutions (C_1, ..., C_m).
    P : int
        Patience parameter. Upper bound for iterations without global improvement.
    reference_labels_list : list of ndarray, optional
        Reference partition(s) to track ARI. If omitted, ARI is not computed
    candidate_names : list of str, optional
        Names of candidates. Defaults to "C_1", "C_2", ... if not given.
        
    Returns
    -------
    dict
        best_labels, best_score, best_ari, tabu_list_size, history
        (one entry per starting candidate: start/end I and ARI).
    """
    n_samples = X.shape[0]
    m = len(candidate_partitions)

    mst_dist, mst_index = get_euclidean_mst(X)

    if candidate_names is None:
        candidate_names = [f"C_{i+1}" for i in range(m)]
    
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
    # 4. return C*
    print("\r" + " " * 40 + "\r", end="")
    print("Optimization complete!")
    print(f"    Best I(C*)         = {I_C_star:.6f}")
    if best_ari is not None:
        print(f"    Best ARI(C*)       = {best_ari:.6f}")
    print(f"    Candidates explored = {m}")
    print(f"    Unique partitions visited (|T|) = {len(T)}")
    print(f"    Best candidate name = {C_star_name}")

    return {
        "best_labels": C_star,
        "best_score": I_C_star,
        "best_ari": best_ari,
        "tabu_list_size": len(T),
        "history": history,
    }


def ari(reference_labels_list, labels):
    """Best ARI of `labels` against any of the given reference partitions.
    """
    if reference_labels_list is None:
        return None
    return max(max(adjusted_rand_score(ref, labels), 0.0) for ref in reference_labels_list)