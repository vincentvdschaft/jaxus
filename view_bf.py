import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from jaxus import plot_beamformed, plot_to_darkmode
from jaxus.beamforming import (
    CartesianPixelGrid,
    beamform_das,
    beamform_dmas,
    beamform_mv,
    find_t_peak,
    log_compress,
)
from jaxus.data import load_usbmd

path = Path(
    r"C:\Users\vince\Documents\3_resources\data\verasonics\usbmd\2024-04-09\L11-5v_carotid_cross_0000.hdf5"
)
# path = Path(r"tests/output.h5")

data_dict = load_usbmd(
    path,
    frames=[
        0,
    ],
    transmits=[137, 138, 139],
    reduce_probe_to_2d=True,
)

wavelength = data_dict["sound_speed"] / data_dict["center_frequency"]

pixel_grid = CartesianPixelGrid(
    n_x=256, n_z=256, dx_wl=0.5, dz_wl=0.5, z0=1e-3, wavelength=wavelength
)

images_das = beamform_das(
    rf_data=data_dict["raw_data"],
    probe_geometry=data_dict["probe_geometry"],
    t0_delays=data_dict["t0_delays"],
    sampling_frequency=data_dict["sampling_frequency"],
    sound_speed=data_dict["sound_speed"],
    carrier_frequency=data_dict["center_frequency"],
    f_number=1.5,
    pixel_positions=pixel_grid.pixel_positions_flat,
    t_peak=find_t_peak(data_dict["waveform_samples_two_way"], 250e6) * np.ones(3),
    initial_times=data_dict["initial_times"],
    rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
    iq_beamform=True,
    progress_bar=True,
)

images_das = log_compress(images_das, normalize=True)
images_das = np.reshape(
    images_das, (images_das.shape[0], pixel_grid.n_rows, pixel_grid.n_cols)
)


images_mv = beamform_mv(
    rf_data=data_dict["raw_data"],
    probe_geometry=data_dict["probe_geometry"],
    t0_delays=data_dict["t0_delays"],
    sampling_frequency=data_dict["sampling_frequency"],
    sound_speed=data_dict["sound_speed"],
    carrier_frequency=data_dict["center_frequency"],
    f_number=1.5,
    pixel_positions=pixel_grid.pixel_positions_flat,
    t_peak=find_t_peak(data_dict["waveform_samples_two_way"], 250e6) * np.ones(3),
    initial_times=data_dict["initial_times"],
    rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
    iq_beamform=True,
    subaperture_size=64,
    diagonal_loading=0.1,
    pixel_chunk_size=32,
    progress_bar=True,
)

images_mv = log_compress(images_mv, normalize=True)
images_mv = np.reshape(
    images_mv, (images_mv.shape[0], pixel_grid.n_rows, pixel_grid.n_cols)
)


images_dmas = beamform_dmas(
    rf_data=data_dict["raw_data"],
    probe_geometry=data_dict["probe_geometry"],
    t0_delays=data_dict["t0_delays"],
    sampling_frequency=data_dict["sampling_frequency"],
    sound_speed=data_dict["sound_speed"],
    carrier_frequency=data_dict["center_frequency"],
    f_number=2.5,
    pixel_positions=pixel_grid.pixel_positions_flat,
    t_peak=find_t_peak(data_dict["waveform_samples_two_way"], 250e6) * np.ones(3),
    initial_times=data_dict["initial_times"],
    rx_apodization=np.ones(data_dict["probe_geometry"].shape[0]),
    iq_beamform=True,
    pixel_chunk_size=1024,
    progress_bar=True,
)

images_dmas = log_compress(images_dmas, normalize=True)
images_dmas = np.reshape(
    images_dmas, (images_dmas.shape[0], pixel_grid.n_rows, pixel_grid.n_cols)
)

print("fone")
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
plot_beamformed(
    axes[0],
    images_das[0],
    pixel_grid.extent,
    title="DAS",
    probe_geometry=data_dict["probe_geometry"],
)
plot_beamformed(
    axes[1],
    images_mv[0],
    pixel_grid.extent,
    title="MV",
    probe_geometry=data_dict["probe_geometry"],
    vmin=-60,
)
plot_beamformed(
    axes[2],
    images_dmas[0],
    pixel_grid.extent,
    title="DMAS",
    probe_geometry=data_dict["probe_geometry"],
    vmin=-100,
)
plot_to_darkmode(fig, axes)

plt.show()
