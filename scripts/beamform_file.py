import argparse
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxus import (
    beamform_das,
    find_t_peak,
    fix_extent,
    get_pixel_grid_from_extent,
    hdf5_get_n_frames,
    hdf5_get_n_tx,
    load_hdf5,
    log_compress,
    plot_beamformed,
    save_hdf5_image,
    use_dark_style,
)
from jaxus.utils import interpret_range, log

# ======================================================================================
# Parse input
# ======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path, default=None, nargs="?")
parser.add_argument("--frames", type=str, nargs="+", default="unspecified")
# Add variable number of transmits
parser.add_argument("--transmits", type=str, nargs="+", default="unspecified")
parser.add_argument("--show", action=argparse.BooleanOptionalAction)
parser.add_argument("--fnumber", type=float, default=1.0)
parser.add_argument("--dynamic-range", type=float, default=60)
parser.add_argument("--extent", type=float, nargs=4, default=None)
parser.add_argument("--save-path", type=Path, default=None)
parser.add_argument("--lens-thickness", type=float, default=1.0)
parser.add_argument("--lens-sound-speed", type=float, default=1000.0)
args = parser.parse_args()
# Create a Tkinter root window

root = None


def init_tk(root):
    if root is not None:
        return root
    root = tk.Tk()
    root.withdraw()
    return root


if args.file is None:
    root = init_tk(root)
    # Prompt the user to select a directory and turn into Path object
    selected_file = filedialog.askopenfile()
    if selected_file is None:
        log.error("No file selected.")
        exit()
    selected_file = Path(str(selected_file.name))
else:
    selected_file = args.file

log.info(f"Selected file: {log.yellow(selected_file)}")

n_frames = hdf5_get_n_frames(selected_file)
n_tx = hdf5_get_n_tx(selected_file)


input_frame = args.frames
if input_frame == "unspecified":
    root = init_tk(root)
    input_frame = simpledialog.askstring(
        "Frames",
        f"Which frames do you want to beamform? [0-{n_frames-1}]\n (e.g. 0 1 2-5, or all)",
    )
    if input_frame == "":
        input_frame = "all"

input_transmit = args.transmits
if input_transmit == "unspecified":
    root = init_tk(root)
    input_transmit = simpledialog.askstring(
        "Transmits",
        f"What transmits do you want to beamform? [0-{n_tx-1}]\n (e.g. 0 1 2-5, or all)",
    )
    if input_transmit == "":
        input_transmit = "all"

if isinstance(input_frame, list):
    input_frame = " ".join(input_frame)
if isinstance(input_transmit, list):
    input_transmit = " ".join(input_transmit)

frames = interpret_range(input_frame, n_frames)
transmits = interpret_range(input_transmit, n_tx)


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
    f"{'--save-path'+str(args.save_path) if args.save_path else ''}"
)

normalization_factor = None

for frame in frames:

    data = load_hdf5(
        selected_file,
        frames=[frame],
        transmits=transmits,
        reduce_probe_to_2d=True,
    )

    wavelength = data["sound_speed"] / data["center_frequency"]
    n_ax = data["raw_data"].shape[2]

    if args.extent is not None:
        extent = fix_extent([float(coord) * 1e-3 for coord in args.extent])
    else:
        extent = [
            data["probe_geometry"][:, 0].min() - 2e-3,
            data["probe_geometry"][:, 0].max() + 2e-3,
            1e-3,
            n_ax * wavelength / 8,
        ]

    print(f"extent: {extent}")
    pixel_size = (wavelength / 2, wavelength / 4)

    pixel_grid = get_pixel_grid_from_extent(extent=extent, pixel_size=pixel_size)

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
        sound_speed_lens=args.lens_sound_speed,
        lens_thickness=args.lens_thickness * 1e-3,
        tx_apodizations=data["tx_apodizations"],
        rx_apodization=jnp.ones(data["tx_apodizations"].shape[1]),
        f_number=args.fnumber,
        t_peak=t_peak,
        angles=data["polar_angles"],
        focus_distances=data["focus_distances"],
        iq_beamform=True,
        progress_bar=True,
        pixel_chunk_size=2**21,
    )

    dynamic_range = abs(float(args.dynamic_range))

    im_das = log_compress(im_das, normalize=False)
    im_das = im_das.reshape((pixel_grid.n_x, pixel_grid.n_z))

    if normalization_factor is None:
        normalization_factor = np.max(im_das)

    im_das = im_das - normalization_factor

    use_dark_style()

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.5))
    ax = axes[1]
    ax_rf = axes[0]
    from jaxus import plot_rf

    print(pixel_grid.extent_m)
    plot_beamformed(
        ax,
        im_das,
        extent_m=pixel_grid.extent_m,
        title="Beamformed RF data",
        probe_geometry=data["probe_geometry"],
        vmin=-dynamic_range,
    )
    plot_rf(
        ax_rf, rf_data=data["raw_data"][0, 0, :, :, 0], aspect="auto", title="RF data"
    )
    plt.tight_layout()

    aspect1 = ax.get_aspect()
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim()
    dx1, dy1 = xlim1[1] - xlim1[0], ylim1[1] - ylim1[0]
    xlim2 = ax_rf.get_xlim()
    ylim2 = ax_rf.get_ylim()
    dx2, dy2 = xlim2[1] - xlim2[0], ylim2[1] - ylim2[0]
    aspect2 = aspect1 * (dy1 * dx2) / (dx1 * dy2)
    ax_rf.set_aspect(aspect2)

    if args.save_path is not None:
        path = Path(args.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path_addition = f"frame_{frame:04d}"
        path = path.with_name(path.stem + f"_{path_addition}" + path.suffix)

        if not path.suffix in [".hdf5", ".png"]:
            path = path.with_suffix(".png")
        # plt.savefig(path, dpi=300, bbox_inches="tight")
        if path.suffix == ".hdf5":
            save_hdf5_image(path, im_das, extent=pixel_grid.extent_m, scale="db")
            log.info(f"Saved to {log.yellow(path)}")
        elif path.suffix == ".png":
            # import cv2

            # im_das = np.clip((im_das.T + dynamic_range) / dynamic_range * 255, 0, 255)
            # cv2.imwrite(path, im_das)
            # log.info(f"Saved to {log.yellow(path)}")
            fig.savefig(path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
