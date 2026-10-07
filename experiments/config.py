from pathlib import Path

def get_project_root(marker_folder="data") -> Path:
    """Zwraca główny katalog projektu (experiments) niezależnie od miejsca wywołania."""
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker_folder).exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = get_project_root("data")
DATA_PATH = PROJECT_ROOT / "data" / "clustering-data-v1"
RESULT_PATH = PROJECT_ROOT / "results"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"