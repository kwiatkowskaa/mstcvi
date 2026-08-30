"""Iterator for benchmark datasets."""

from pathlib import Path
from typing import NamedTuple, Iterator

import clustbench
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "clustering-data-v1"

BATTERIES = ("fcps", "graves", "other", "sipu", "uci", "wut", "g2mg", "h2mg", "mnist")

DEFAULT_BATTERIES = ("fcps", "graves", "other", "sipu", "uci", "wut")


class BenchmarkDataset(NamedTuple):
    battery: str
    name: str
    X: np.ndarray
    reference_labels: list[np.ndarray]  # l >= 1 reference labels


def iter_benchmark_datasets(batteries, DATA_PATH=DATA_PATH) -> Iterator[BenchmarkDataset]:
    """
    Iterates over clustering benchmark datasets from specified batteries.
    
    Fetches and preprocesses datasets (removing zero-variance features and
    adding minimal noise to ensure unique points).
    
    Parameters
    ----------
    batteries : tuple[str, ...]
        Names of clustbench dataset batteries to iterate over.

    Yields
    ------
    BenchmarkDataset
        A named tuple containing battery name, dataset name, feature matrix X,
        and a list of reference label arrays.
    """
    for battery in batteries:
        for name in clustbench.get_dataset_names(battery, path=DATA_PATH):
            b = clustbench.load_dataset(
                battery, name, path=DATA_PATH, preprocess=True, random_state=42
            )
            labels = b.labels if isinstance(b.labels, list) else [b.labels]
            yield BenchmarkDataset(battery, name, b.data, labels)