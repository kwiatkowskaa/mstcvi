"""Batch-runs the Algorithm-1 CVI-optimisation experiment (see
cvi_validity.py) across every dataset in a battery.
"""

import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .cvi_optimization import generate_initial_partitions, optimize_cvi_tabu_search
from .datasets import iter_benchmark_datasets


def run_optimization_experiment(
    battery,
    I_func,
    I_name,
    data_path,
    results_dir,
    *,
    P=100,
    n_init=5,
    random_state=42,
    skip_large=True,
    verbose=True,
):
    """Run the CVI-optimization search on every dataset in `battery`,
    for every distinct k found among its reference labellings.


    Parameters
    ----------
    battery : str
        Battery name.
    I_func : callable(X, labels, **mst_kwargs) -> float
        The CVI being evaluated.
    I_name : str
        Short identifier used in result filenames.
    data_path : Path
        Local clustering-data-v1 path.
    results_dir : Path
        Where per-run pickles and the running summary CSV are written.
    P, n_init, random_state, skip_large
        Forwarded to optimize_cvi_tabu_search / generate_initial_partitions
        / iter_benchmark_datasets.

    Returns
    -------
    pd.DataFrame
        One row per (dataset, k) combination
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{battery}__{I_name}__summary.csv"

    rows = []
    for ds in iter_benchmark_datasets([battery], data_path=data_path, skip_large=skip_large):
        ks = sorted({len(np.unique(ref)) for ref in ds.reference_labels})

        for k in ks:
            run_id = f"{ds.battery}__{ds.name}__k{k}"
            pickle_path = results_dir / f"{run_id}.pkl"

            if pickle_path.exists():
                if verbose:
                    print(f"[already done] {run_id}")
                with open(pickle_path, "rb") as f:
                    rows.append(pickle.load(f)["summary_row"])
                continue

            if verbose:
                print(f"\n=== {ds.battery}/{ds.name}  n={ds.X.shape[0]} d={ds.X.shape[1]} k={k} ===")

            try:
                candidate_partitions, candidate_names = generate_initial_partitions(
                    ds.X, k, n_init=n_init, random_state=random_state
                )
                result = optimize_cvi_tabu_search(
                    ds.X, I_func, candidate_partitions, k=k, P=P,
                    reference_labels_list=ds.reference_labels,
                    candidate_names=candidate_names,
                )
            except Exception:
                print(f"[FAILED] {run_id} -- see traceback below, continuing with the next run")
                traceback.print_exc()
                continue

            summary_row = {
                "I_name": I_name,
                "battery": ds.battery, "dataset": ds.name,
                "n": ds.X.shape[0], "d": ds.X.shape[1], "k": k,
                "best_score": result["best_score"],
                "best_ari": result["best_ari"],
                "best_candidate_name": result["best_candidate_name"],
                "tabu_list_size": result["tabu_list_size"],
                "n_evaluations": result["n_evaluations"],
                "elapsed_seconds": result["elapsed_seconds"],
            }

            with open(pickle_path, "wb") as f:
                pickle.dump({"summary_row": summary_row, "result": result}, f)

            rows.append(summary_row)
            pd.DataFrame(rows).to_csv(summary_path, index=False)

    return pd.DataFrame(rows)