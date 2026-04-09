import os
import runpy
from pathlib import Path

import constants as const


ROOT = Path(__file__).resolve().parent


def ensure_directories():
    required_dirs = [
        ROOT / "data" / "main",
        ROOT / "tables",
        ROOT / "visuals",
        ROOT / "artifacts",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def run_script(script_name):
    script_path = ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script_path}")
    runpy.run_path(str(script_path), run_name="__main__")


def main():
    os.chdir(ROOT)
    ensure_directories()

    print("Running data preparation for all commodity x region modeling.", flush=True)
    run_script("data_preparation.py")

    print("Running full main.py pipeline with all eligible commodity x region series.", flush=True)
    const.MODEL_SERIES_CAP = 10**9
    const.MODEL_SERIES_CAP_FALLBACK = 10**9
    run_script("main.py")


if __name__ == "__main__":
    main()
