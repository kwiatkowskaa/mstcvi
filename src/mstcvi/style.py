"""Visual style definitions and palette for plots."""

import matplotlib.pyplot as plt

# --- Cluster colours -----------------------------------------------
CLUSTER_COLORS = [
    "#568f8b",
    "#cd7e59",
    "#1d4a60",
    "#ddb247",
    "#d15252",
    "#8ecae6",
    "#775b59",
    "#b4d2b1",
    "#5f0f40",
    "#c88fbb",
]

# --- Cluster markers -----------------------------------------------
MARKER_SHAPES = ["o", "s", "^", "D", "P"]  # koło, kwadrat, trójkąt, romb, trójkąt w dół, plus


# --- Semantic colours ----------------------------------------------
CUT_EDGE_COLOR = "#FF2400"    # b_i
MEAN_LINE_COLOR = "#FF2400"   # treelhouette_score


def cluster_color(position: int) -> str:
    """Colour for the `position`-th cluster (0-indexed, cycles)."""
    return CLUSTER_COLORS[position % len(CLUSTER_COLORS)]

def cluster_marker(position: int) -> str:
    """Marker shape for the `position`-th cluster (0-indexed, cycles)."""
    return MARKER_SHAPES[position % len(MARKER_SHAPES)]