"""This module contains convenience functions wrapping the core functionality."""

from pathlib import Path
from typing import Union

import numpy as np

import jaxus.utils.log as log
from jaxus.beamforming import Beamformer, CartesianPixelGrid, log_compress
from jaxus.containers import Medium, Probe, Pulse, Receive, Transmit
from jaxus.data import generate_usbmd_dataset

from .rf_simulator import simulate_rf_data


def simulate_rf(
    probe: Probe,
    transmit: Union[Transmit, list],
    receive: Receive,
    medium: Medium,
    tx_angle_sensitivity=True,
    rx_angle_sensitivity=True,
    wavefront_only=False,
):
    """Simulates RF data for a given probe, transmit, receive, and medium.

    ### Parameters
        `probe` (`Probe`): The probe object.
        `transmit` (`Transmit` or `list`): The transmit object.
        `receive` (`Receive`): The receive object.
        `medium` (`Medium`): The medium object.
        `tx_angle_sensitivity` (`bool`, optional): Whether to include the transmit angle
            sensitivity. Defaults to True.
        `rx_angle_sensitivity` (`bool`, optional): Whether to include the receive angle
            sensitivity. Defaults to True.
        `wavefront_only` (`bool`, optional): Whether to simulate the wavefront only.
            Defaults to False.

    ### Returns
        `np.ndarray`: The RF data of shape (n_tx, n_ax, n_el).
    """

    if isinstance(transmit, Transmit):
        transmit = [transmit]
    elif not isinstance(transmit, list):
        raise ValueError("transmit must be a list or a Transmit object.")

    rf_data_list = []

    for tx in transmit:
        # Compute the wavelength of the carrier frequency
        # (used to compute the element width in wavelengths)
        wavelength = medium.sound_speed / tx.carrier_frequency

        # Simulate the RF data
        rf_data = simulate_rf_data(
            n_ax=receive.n_ax,
            scatterer_positions=medium.scatterer_positions,
            scatterer_amplitudes=medium.scatterer_amplitudes,
            t0_delays=tx.t0_delays,
            probe_geometry=probe.probe_geometry,
            element_angles=np.zeros(probe.n_el),
            tx_apodization=tx.tx_apodization,
            initial_time=receive.initial_time,
            element_width_wl=probe.element_width / wavelength,
            sampling_frequency=receive.sampling_frequency,
            carrier_frequency=tx.carrier_frequency,
            sound_speed=medium.sound_speed,
            attenuation_coefficient=medium.attenuation_coefficient,
            tx_angle_sensitivity=tx_angle_sensitivity,
            rx_angle_sensitivity=rx_angle_sensitivity,
            wavefront_only=wavefront_only,
            waveform_function=tx.waveform.get_waveform_function(),
        )
        rf_data_list.append(rf_data)

    # Stack the RF data along the first axis
    rf_data = np.stack(rf_data_list, axis=0)

    return rf_data


def simulate_to_usbmd(
    path: Union[str, Path],
    probe: Probe,
    transmit: Union[Transmit, list],
    receive: Receive,
    medium: Medium,
    tx_angle_sensitivity=True,
    rx_angle_sensitivity=True,
    wavefront_only=False,
):
    """Simulates RF data and beamforms it to the ultrasound image domain.

    ### Parameters
        `probe` (`Probe`): The probe object.
        `transmit` (`Transmit`): The transmit object.
        `receive` (`Receive`): The receive object.
        `medium` (`Medium`): The medium object.
        `tx_angle_sensitivity` (`bool`, optional): Whether to include the transmit angle
            sensitivity. Defaults to True.
        `rx_angle_sensitivity` (`bool`, optional): Whether to include the receive angle
            sensitivity. Defaults to True.
        `wavefront_only` (`bool`, optional): Whether to simulate the wavefront only.
            Defaults to False.

    ### Returns
        `np.ndarray`: The beamformed image of shape (n_tx, n_ax, n_az).
    """
    if isinstance(transmit, Transmit):
        transmit = [transmit]
    elif not isinstance(transmit, list):
        raise TypeError("transmit must be a list or a Transmit object.")

    # Simulate the RF data
    rf_data = simulate_rf(
        probe=probe,
        transmit=transmit,
        receive=receive,
        medium=medium,
        tx_angle_sensitivity=tx_angle_sensitivity,
        rx_angle_sensitivity=rx_angle_sensitivity,
        wavefront_only=wavefront_only,
    )
    # Add frame dimension and channel dimension
    rf_data = rf_data[None, :, :, :, None]

    # Beamform the RF data
    wavelength = medium.sound_speed / probe.center_frequency
    dx_wl = 0.25
    dz_wl = 0.25

    depth_m = receive.n_ax / receive.sampling_frequency * medium.sound_speed / 2

    pixel_grid = CartesianPixelGrid(
        n_x=probe.aperture / (wavelength * dx_wl),
        n_z=depth_m / (wavelength * dz_wl),
        dx_wl=dx_wl,
        dz_wl=dz_wl,
        wavelength=wavelength,
        z0=1e-3,
    )

    # Ensure that all transmit frequencies are the same
    if not all(
        tx.carrier_frequency == transmit[0].carrier_frequency for tx in transmit
    ):
        log.warning(
            "All transmit frequencies must be the same. Using the first one. "
            "Beamforming may be inaccurate."
        )

    t0_delays = np.stack([tx.t0_delays for tx in transmit], axis=0)
    initial_times = np.stack([receive.initial_time for _ in transmit], axis=0)
    t_peak = np.stack([tx.waveform.t_peak for tx in transmit], axis=0)
    tx_apodization = np.stack([tx.tx_apodization for tx in transmit], axis=0)

    beamformer = Beamformer(
        pixel_grid=pixel_grid,
        probe_geometry=probe.probe_geometry.T,
        t0_delays=t0_delays,
        initial_times=initial_times,
        sampling_frequency=receive.sampling_frequency,
        carrier_frequency=transmit[0].carrier_frequency,
        sound_speed=medium.sound_speed,
        t_peak=t_peak,
        rx_apodization=np.ones(probe.n_el),
        f_number=1.5,
        z0=1e-3,
    )

    beamformed = beamformer.beamform(rf_data)
    beamformed = log_compress(beamformed, normalize=True)

    waveforms = []
    for tx in transmit:
        waveform_fn = tx.waveform.get_waveform_function_array()
        t = np.arange(1024) / 250e6
        waveforms.append(waveform_fn(t))

    # ==================================================================================
    # Generate USBMD dataset
    # ==================================================================================
    generate_usbmd_dataset(
        path,
        raw_data=rf_data,
        beamformed_data=beamformed,
        probe_geometry=probe.probe_geometry.T,
        sampling_frequency=receive.sampling_frequency,
        center_frequency=probe.center_frequency,
        sound_speed=medium.sound_speed,
        t0_delays=t0_delays,
        initial_times=initial_times,
        tx_apodizations=tx_apodization,
        probe_name="custom",
        focus_distances=np.zeros(len(transmit)),
        polar_angles=np.zeros(len(transmit)),
        bandwidth_percent=200,
        element_width=probe.element_width,
        lens_correction=0.0,
        description="Custom simulated dataset",
        azimuth_angles=np.zeros(len(transmit)),
        time_to_next_transmit=np.zeros(len(transmit) - 1),
        tgc_gain_curve=np.ones(receive.n_ax),
        waveform_samples_one_way=waveforms,
        waveform_samples_two_way=waveforms,
    )

    return beamformed
