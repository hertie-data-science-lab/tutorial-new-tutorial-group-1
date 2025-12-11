#!/usr/bin/env python3
"""
Minimal setup script for the RID dataset.

Place this script in your `data/` directory and run:

    python setup_rid_data_minimal.py

It will:
- rsync ONLY:
    - images_roof_centered_geotiff
    - masks_superstructures_reviewed
  from the TUM server
- download them directly into THIS folder (data/)
- skip downloading if the folder already exists and is non-empty
"""

import subprocess
from pathlib import Path

SERVER = "rsync://m1655470@dataserv.ub.tum.de/m1655470"
BASE_DIR = Path(__file__).resolve().parent     # data/
DATA_DIR = BASE_DIR

FOLDERS = [
    "images_roof_centered_geotiff",
    "masks_superstructures_reviewed",
]


def folder_has_data(path: Path) -> bool:
    """Return True if folder exists and contains at least one file."""
    return path.exists() and any(path.iterdir())


def rsync_folder(remote_name: str):
    """Rsync a single folder from the server into DATA_DIR."""
    dst = DATA_DIR / remote_name

    # Skip if already present and non-empty
    if folder_has_data(dst):
        print(f"✔ Skipping '{remote_name}' — folder already exists and is not empty.")
        return

    # Ensure directory exists
    dst.mkdir(parents=True, exist_ok=True)

    src = f"{SERVER}/{remote_name}/"
    cmd = [
        "rsync",
        "-av",
        "--progress",
        src,
        str(dst) + "/",     # ensure trailing slash
    ]

    print(f">>> Downloading '{remote_name}' from server...")
    print("    ", " ".join(cmd))
    print(">>> You may be asked for the password (m1655470).")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit("ERROR: rsync not found. Please install rsync and try again.")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ERROR: rsync for {remote_name} failed with exit code {e.returncode}.")


def main():
    print("=== Minimal RID dataset setup (only 2 folders, with existence check) ===")
    print(f"Data directory: {DATA_DIR}")
    print()

    for folder in FOLDERS:
        rsync_folder(folder)
        print()

    print("=== Done! ===")
    print("Folders now present:")
    for folder in FOLDERS:
        print(" •", DATA_DIR / folder)


if __name__ == "__main__":
    main()
