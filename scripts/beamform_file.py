import tkinter as tk
from pathlib import Path
from tkinter import filedialog, simpledialog
import argparse
import matplotlib.pyplot as plt

import h5py

from jaxus import (
    load_hdf5,
    beamform_das,
    log_compress,
    plot_beamformed,
    CartesianPixelGrid,
    find_t_peak,
    use_dark_style,
)
import jaxus.utils.log as log
import jax.numpy as jnp

parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path, default=None, nargs="?")
parser.add_argument("--frame", type=int, default=None)
# Add variable number of transmits
parser.add_argument("--transmits", type=str, nargs="+", default=None)
args = parser.parse_args()

if args.file is None:
    # Create a Tkinter root window
    root = tk.Tk()
    root.withdraw()

    # Prompt the user to select a directory and turn into Path object
    selected_file = filedialog.askopenfile().name
    if selected_file is None:
        log.error("No file selected.")
        exit()
    selected_file = Path(str(selected_file))
else:
    selected_file = args.file

log.info(f"Selected file: {log.yellow(selected_file)}")

with h5py.File(selected_file, "r") as f:
    try:
        n_frames, n_tx, _, _, _ = f["data"]["raw_data"].shape
    except KeyError:
        log.error("The selected file does not contain the correct data.")
        exit()

# Check if the frame was selected
if args.frame is None:
    frame = simpledialog.askinteger(
        "Input", f"Select a frame to beamform. [0-{n_frames-1}]"
    )
    if frame is None:
        log.error("No frame selected. Using 0.")
        frame = 0
else:
    frame = args.frame

# Check if the transmits were selected
if args.transmits is None:
    input = simpledialog.askstring(
        "Input", "Select transmits to beamform. [0-127] or [0 1 2 3]"
    )
else:
    input = " ".join(args.transmits)
# Parse the input if it is in the format %i-%i
if "-" in input:
    start, end = input.split("-")
    transmits = list(range(int(start), int(end) + 1))
# Parse the input if it is in the format %i %i %i
else:
    transmits = list(map(int, input.split()))

if input is None:
    log.error("No transmits selected. Using all transmits.")
    transmits = list(range(n_tx))

current_python_file = Path(__file__).resolve()

log.info(
    f"Command to reproduce:'\npython {current_python_file} {selected_file} --frame {frame} --transmits {input}"
)

data = load_hdf5(
    selected_file,
    frames=[frame],
    transmits=transmits,
    reduce_probe_to_2d=True,
)

wavelength = data["sound_speed"] / data["center_frequency"]

pixel_grid = CartesianPixelGrid(
    n_x=1024,
    n_z=1024 + 512,
    dx_wl=0.25,
    dz_wl=0.25,
    z0=1e-3,
    wavelength=wavelength,
)

t_peak = find_t_peak(data["waveform_samples_two_way"][0][:]) * jnp.ones(1)

im_das = beamform_das(
    rf_data=data["raw_data"],
    pixel_positions=pixel_grid.pixel_positions_flat,
    probe_geometry=data["probe_geometry"],
    t0_delays=data["t0_delays"],
    initial_times=data["initial_times"],
    sampling_frequency=data["sampling_frequency"],
    carrier_frequency=data["center_frequency"],
    sound_speed=data["sound_speed"],
    tx_apodizations=data["tx_apodizations"],
    rx_apodization=jnp.ones(data["tx_apodizations"].shape[1]),
    f_number=0.5,
    t_peak=t_peak,
    iq_beamform=True,
    progress_bar=True,
)

im_das = log_compress(im_das, normalize=True)
im_das = im_das.reshape((pixel_grid.n_rows, pixel_grid.n_cols))


use_dark_style()

fig, ax = plt.subplots()
plot_beamformed(
    ax,
    im_das,
    extent_m=pixel_grid.extent,
    title="DAS Beamforming",
    probe_geometry=data["probe_geometry"],
    vmin=-60,
)
plt.show()
print("done")
