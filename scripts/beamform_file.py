import tkinter as tk
from pathlib import Path
from tkinter import filedialog, simpledialog, messagebox
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

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

# ======================================================================================
# Parse input
# ======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path, default=None, nargs="?")
parser.add_argument("--frames", type=str, nargs="+", default=None)
# Add variable number of transmits
parser.add_argument("--transmits", type=str, nargs="+", default=None)
parser.add_argument("--show", action=argparse.BooleanOptionalAction)
parser.add_argument("--fnumber", type=float, default=1.0)
parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
# Create a Tkinter root window
root = tk.Tk()
root.withdraw()

if args.file is None:

    # Prompt the user to select a directory and turn into Path object
    selected_file = filedialog.askopenfile()
    if selected_file is None:
        log.error("No file selected.")
        exit()
    selected_file = Path(str(selected_file.name))
else:
    selected_file = args.file

log.info(f"Selected file: {log.yellow(selected_file)}")

with h5py.File(selected_file, "r") as f:
    try:
        n_frames, n_tx, _, _, _ = f["data"]["raw_data"].shape
    except KeyError:
        log.error("The selected file does not contain the correct data.")
        exit()


# --------------------------------------------------------------------------------------
# Interpret frame
# --------------------------------------------------------------------------------------
# Check if the frame was selected
if args.frames is None:
    input_frame = simpledialog.askstring(
        "Input", f"Select a frame to beamform. 0-{n_frames-1} or 0 1 2 3 or all"
    )
else:
    input_frame = " ".join(args.frames)

if input_frame is None:
    log.error("No frame selected. Using 0.")
    frames = [0]
elif input_frame == "all":
    frames = list(range(n_frames))
elif "-" in str(input_frame):
    start, end = input_frame.split("-")
    frames = list(range(int(start), int(end) + 1))
else:
    frames = list(map(int, input_frame.split()))


# --------------------------------------------------------------------------------------
# Interpret transmit
# --------------------------------------------------------------------------------------
# Check if the transmits were selected
if args.transmits is None:
    input_transmit = simpledialog.askstring(
        "Input", f"Select transmits to beamform. [0-{n_tx-1}] or [0 1 2 3]"
    )
else:
    if isinstance(args.transmits, list):
        input_transmit = " ".join(args.transmits)
    else:
        input_transmit = args.transmits

# Parse the input if it is in the format %i-%i
if input_transmit == "all":
    transmits = list(range(n_tx))
elif "-" in input_transmit:
    print(input_transmit)
    start, end = input_transmit.split("-")
    transmits = list(range(int(start), int(end) + 1))
# Parse the input if it is in the format %i %i %i
else:
    print(input_transmit)
    transmits = list(map(int, input_transmit.split()))

if input_transmit is None:
    log.error("No transmits selected. Using all transmits.")
    transmits = list(range(n_tx))


# --------------------------------------------------------------------------------------
# Interpret show
# --------------------------------------------------------------------------------------
# Check if show was supplied
if args.show is not None:
    show = args.show
else:
    # Ask yes no
    input_show = messagebox.askquestion("show", "Show images in popup window?")
    show = input_show == "yes"


# --------------------------------------------------------------------------------------
# Report the command to reproduce the results
# --------------------------------------------------------------------------------------
current_python_file = Path(__file__).resolve()

log.info(
    f"Command to reproduce:'\n"
    f"python {current_python_file} {selected_file} "
    f"--frames {input_frame} "
    f"--transmits {input_transmit} "
    f"{'--show'if show else '--no-show'} "
    f"{'--fnumber '+str(args.fnumber) if args.fnumber != 1.0 else ''}"
    f"{'--save' if args.save else ''}"
)

if args.save:
    output_dir = Path("results", f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

for frame in frames:

    data = load_hdf5(
        selected_file,
        frames=[frame],
        transmits=transmits,
        reduce_probe_to_2d=True,
    )

    wavelength = data["sound_speed"] / data["center_frequency"]
    n_ax = data["raw_data"].shape[2]
    dz_wl = 0.25
    n_z = int(0.25 * n_ax / (2 * dz_wl))

    pixel_grid = CartesianPixelGrid(
        n_x=1024 + 256,
        n_z=n_z,
        dx_wl=0.25,
        dz_wl=dz_wl,
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
        sound_speed_lens=1000,
        lens_thickness=1.5e-3,
        tx_apodizations=data["tx_apodizations"],
        rx_apodization=jnp.ones(data["tx_apodizations"].shape[1]),
        f_number=args.fnumber,
        t_peak=t_peak,
        iq_beamform=True,
        progress_bar=True,
        pixel_chunk_size=2**21,
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
    if args.save:
        output_path = output_dir / f"frame_{str(frame).zfill(3)}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
