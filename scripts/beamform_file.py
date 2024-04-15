import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import h5py

from jaxus.convenience import *

# Create a Tkinter root window
root = tk.Tk()
root.withdraw()

# Prompt the user to select a directory and turn into Path object
selected_file = Path(filedialog.askopenfile())

with h5py.File(selected_file, "r") as f:
    rf_data = f["rf_data"][:]
    sampling_frequency = f["sampling_frequency"][()]
    carrier_frequency = f["center_frequency"][()]
