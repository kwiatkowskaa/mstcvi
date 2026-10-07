"""Iterator for benchmark datasets."""

from pathlib import Path
from typing import NamedTuple, Iterator

import clustbench
import numpy as np

from config import DATA_PATH


BATTERIES = ("fcps", "graves", "other", "sipu", "uci", "wut", "g2mg", "h2mg", "mnist")
DEFAULT_BATTERIES = ("fcps", "graves", "other", "sipu", "uci", "wut")


# Datasets excluded because at least one reference labelling gives n*(k-1) > 50,000,
# making Algorithm 1's O(n(k-1))-per-step neighbourhood search too costly.
LARGE_DATASETS_TO_SKIP = frozenset({
    "mnist/digits", "mnist/fashion",
    "other/chameleon_t7_10k", "other/chameleon_t8_8k",
    "sipu/a1", "sipu/a2", "sipu/a3",
    "sipu/birch1", "sipu/birch2", "sipu/d31",
    "sipu/s1", "sipu/s2", "sipu/s3", "sipu/s4",
    "sipu/worms_2", "sipu/worms_64",
})

class BenchmarkDataset(NamedTuple):
    battery: str
    name: str
    X: np.ndarray
    reference_labels: list[np.ndarray]  # l >= 1 reference labels


def iter_benchmark_datasets(batteries, data_path=DATA_PATH, skip_large=True) -> Iterator[BenchmarkDataset]:
    """
    Iterates over clustering benchmark datasets from specified batteries.
    
    Fetches, preprocesses datasets (removing zero-variance features and
    adding minimal noise to ensure unique points) and filtering out
    noise points (label == 0) from both the feature matrix X and reference labels.
    
    Parameters
    ----------
    batteries : tuple[str, ...]
        Names of clustbench dataset batteries to iterate over.
    data_path : Path, default=DATA_PATH
        Local path to the cloned clustering-data-v1 repository.
    skip_large : bool, default=True
        Skip datasets listed in LARGE_DATASETS_TO_SKIP

    Yields
    ------
    BenchmarkDataset
        A named tuple containing battery name, dataset name, feature matrix X,
        and a list of reference label arrays.
    """
    for battery in batteries:
        for name in clustbench.get_dataset_names(battery, path=data_path):
            full_name = f"{battery}/{name}"

            if skip_large and full_name in LARGE_DATASETS_TO_SKIP:
                continue

            b = clustbench.load_dataset(
                battery, name, path=data_path, preprocess=True, random_state=42
            )
            labels = b.labels if isinstance(b.labels, list) else [b.labels]

            # removing noise
            labels_filtered = []
            valid_mask = None
            
            for ref_labels in labels:
                ref_arr = np.asarray(ref_labels)
                mask = ref_arr != 0
                
                if valid_mask is None:
                    valid_mask = mask
                else:
                    valid_mask = valid_mask & mask

            if valid_mask is not None and not np.all(valid_mask):
                X_filtered = b.data[valid_mask]
                
                for ref_labels in labels:
                    ref_arr = np.asarray(ref_labels)
                    labels_filtered.append(ref_arr[valid_mask])
            else:
                X_filtered = b.data
                labels_filtered = [np.asarray(ref) for ref in labels]

            yield BenchmarkDataset(battery, name, X_filtered, labels_filtered)