"""This script converts all .mat files in a directory to USBMD format."""

import os
import sys

# pylint: disable=C0413
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from jaxus.data import hdf5_from_matlab_raw
from jaxus import log


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "directory",
    type=Path,
    help="The directory containing the .mat files to convert to USBMD format.",
    default=None,
    nargs="?",
)
args = parser.parse_args()

if args.directory is None:
    # Create a Tkinter root window
    root = tk.Tk()
    root.withdraw()
    selected_directory = filedialog.askdirectory()
else:
    selected_directory = Path(args.directory)


def find_raw_data_dir(selected_directory):
    """Move up in the directory structure until a folder with the name
    'VERASONICS_ROOT.txt' is found. This folder is assumed to be the root of the
    Verasonics data. If the folder is not found, the selected directory is returned."""
    current_directory = selected_directory
    while not os.path.isfile(os.path.join(current_directory, "VERASONICS_ROOT.txt")):
        current_directory = os.path.dirname(current_directory)
        # Check if we are at the root of the file system
        if current_directory == os.path.dirname(current_directory):
            log.info(
                "VERASONICS_ROOT.txt not found in the directory structure. "
                "Assuming selected directory is the root of the Verasonics data."
            )
            return selected_directory
    return Path(current_directory) / "raw"


# Check if a directory was selected
if selected_directory:
    answer = messagebox.askyesno(
        "?", "Do you want to reconvert and overwrite existing files?"
    )

    # Destroy the Tkinter root window
    root.destroy()

    # Convert the selected directory to a Path object
    raw_data_dir = find_raw_data_dir(Path(selected_directory))
    print(f"raw_data_dir: {raw_data_dir}")
    output_dir = raw_data_dir.parent / "hdf5_format"

    # Create the output directory if it does not exist
    if not output_dir.exists():
        output_dir.mkdir()

    # Continue with the rest of your code...
    for root, dirs, files in os.walk(selected_directory):
        for mat_file in files:
            if not mat_file.endswith(".mat"):  # or "epic" in mat_file.lower():
                print(f"skipping {mat_file}")
                continue
            mat_file = Path(mat_file)
            log.info(f"Converting {log.yellow(mat_file)}")

            relative_path = (Path(root) / Path(mat_file)).relative_to(raw_data_dir)
            output_path = output_dir / (relative_path.with_suffix(".hdf5"))

            full_path = raw_data_dir / relative_path

            if output_path.is_file():
                if answer:
                    log.info("Exists. Deleting...")
                    output_path.unlink(missing_ok=False)
                else:
                    log.info("Exists. Skipping...")
                    continue

            try:
                hdf5_from_matlab_raw(
                    full_path,
                    output_path,
                )
            except Exception as e:
                # Print error message without raising it
                print(e)
                log.error(f"Failed to convert {mat_file}")
                continue
else:
    log.info("No directory selected. Aborting...")
