"""This module contains convenience functions wrapping the core functionality."""

from pathlib import Path
from typing import Union

import numpy as np

import jaxus.utils.log as log
from jaxus.beamforming import CartesianPixelGrid, beamform, log_compress
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

    beamformed = beamform(
        rf_data,
        pixel_positions=pixel_grid.pixel_positions_flat,
        probe_geometry=probe.probe_geometry.T,
        t0_delays=t0_delays,
        initial_times=initial_times,
        sampling_frequency=receive.sampling_frequency,
        carrier_frequency=transmit[0].carrier_frequency,
        sound_speed=medium.sound_speed,
        t_peak=t_peak,
        rx_apodization=np.ones(probe.n_el),
        f_number=1.5,
        iq_beamform=True,
    )

    beamformed = np.reshape(
        beamformed, (beamformed.shape[0], pixel_grid.rows, pixel_grid.cols)
    )

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


# TODO: Fix
# def beamform(
#     rf_data: np.ndarray,
#     pixel_grid: PixelGrid,
#     probe_geometry: np.ndarray,
#     t0_delays: np.ndarray,
#     initial_times: np.ndarray,
#     sampling_frequency: Union[float, int],
#     carrier_frequency: Union[float, int],
#     sound_speed: Union[float, int],
#     t_peak: Union[float, int],
#     rx_apodization: np.ndarray = None,
#     f_number: Union[float, int] = 3,
#     z0: Union[float, int] = 0,
#     normalize: bool = False,
#     iq_beamform: bool = False,
#     transmits: Union[None, int, list] = None,
#     progress_bar: bool = False,
# ):
#     """Beamforms all frames of the rf data using the given parameters. Also performs
#     log-compression of the beamformed data.

#     ### Args:
#         `rf_data` (`np.ndarray`): The RF data to beamform of shape
#             `(n_frames, n_tx, n_ax, n_el, n_ch)`. This can be either rf data with n_ch=1
#             or complex IQ data with n_ch=2.
#         `pixel_grid` (`PixelGrid`): The pixel grid to beamform to.
#         `probe_geometry` (`np.ndarray`): The probe geometry in meters of shape
#             (n_elements, 2).
#         `t0_delays` (`np.ndarray`): The transmit delays of shape (n_tx, n_el). These are
#             the times between t=0 and every element firing in seconds. (t=0 is when the
#             first element fires.)
#         `initial_times` (`np.ndarray`): The time between t=0 and the first sample being
#             recorded. (t=0 is when the first element fires.)
#         `sampling_frequency` (`float`): The sampling frequency in Hz.
#         `carrier_frequency` (`float`): The center frequency of the RF data in Hz.
#         `sound_speed` (`float`): The speed of sound in m/s.
#         `t_peak` (`float`): The time between t=0 and the peak of the waveform to
#             beamform to. (t=0 is when the first element fires)
#         `f_number` (`float`): The f-number to use for the beamforming. The f-number is
#             the ratio of the focal length to the aperture size. Elements that are more
#             to the side of the current pixel than the f-number are not used in the
#             beamforming. Default is 3.
#         `normalize` (`bool`): Whether to normalize the beamformed and compounded image
#             such that the brightest pixel is 0dB. Default is False.
#         `iq_beamform` (`bool`): Set to True to do the beamforming after converting the
#             RF data to IQ data. Cannot be False if the input is IQ-data. Default is
#             False.
#         `transmits` (`None`, `int`, `list`): The transmits to beamform. Set to None to
#             use all transmits. Defaults to None.
#         `progress_bar` (`bool`): Whether to show a progress bar. Default is False.

#     ### Returns:
#         `beamformed_images` (`np.ndarray`), `x_vals` (`np.ndarray`), `z_vals`
#             (`np.ndarray`): The beamformed and log-compressed images of shape
#             `(n_frames, n_z, n_x)`, the x-axis of the pixel grid in meters and the
#             z-axis of the pixel grid in meters.
#     """

#     # Input error checking is performed in the Beamformer class

#     if rx_apodization is None:
#         n_el = probe_geometry.shape[0]
#         rx_apodization = hamming(n_el)

#     # Initialize the beamformer object
#     beamformer = Beamformer(
#         pixel_grid,
#         probe_geometry,
#         t0_delays,
#         initial_times,
#         sampling_frequency,
#         carrier_frequency,
#         sound_speed,
#         t_peak,
#         rx_apodization,
#         f_number,
#         z0,
#         iq_beamform,
#     )

#     # Beamform the data
#     beamformed_images = beamformer.beamform(rf_data, transmits, progress_bar)

#     if not iq_beamform:
#         # Perform envelope detection
#         beamformed_images = detect_envelope_beamformed(
#             beamformed_images, 0.5 * carrier_frequency / sampling_frequency
#         )

#     beamformed_images = log_compress(beamformed_images, normalize)

#     xlims = pixel_grid.xlim
#     x_vals = np.linspace(xlims[0], xlims[1], beamformed_images.shape[2])
#     zlims = pixel_grid.zlim
#     z_vals = np.linspace(zlims[0], zlims[1], beamformed_images.shape[1])

#     return (
#         beamformed_images,
#         x_vals,
#         z_vals,
#     )
