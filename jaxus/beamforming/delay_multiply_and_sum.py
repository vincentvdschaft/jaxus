"""This module contains Delay-And-Sum (DAS) beamforming functionality.

The core function of this module is the `beamform` function that performs Delay-And-Sum
beamforming on RF or IQ data.
"""

from functools import partial
from typing import Callable, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, vmap
from jax.lax import gather
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal.windows import hamming
from tqdm import tqdm

from jaxus.utils.checks import (
    check_frequency,
    check_initial_times,
    check_pos_array,
    check_posfloat,
    check_shapes_consistent,
    check_sound_speed,
    check_standard_rf_data,
    check_standard_rf_or_iq_data,
    check_t0_delays,
)

from .beamform import get_f_number_mask, rf2iq, to_complex_iq, tof_correct_pixel


@jit
def _beamform_pixel(
    rf_data,
    pixel_pos,
    t0_delays,
    initial_time,
    sound_speed,
    sound_speed_lens,
    lens_thickness,
    t_peak,
    probe_geometry,
    carrier_frequency,
    sampling_frequency,
    f_number,
    tx_apodization,
    rx_apodization,
):
    """Beamforms a single pixel of a single frame and single transmit. Further
    processing such as log-compression and envelope detection are not performed.

    Parameters
    ----------
    rf_data : jnp.ndarray
        The IQ data to beamform of shape `(n_samples, n_el)`.
    pixel_pos : jnp.ndarray
        The position of the pixel to beamform to in meters of shape `(2,)`.
    t0_delays : jnp.ndarray
        The transmit delays of shape (n_tx, n_el). These are the times between t=0 and
        every element firing in seconds. (t=0 is when the first element fires.)
    initial_time : jnp.ndarray
        The time between t=0 and the first sample being recorded. (t=0 is when the first
        element fires.)
    sound_speed : float
        The speed of sound in m/s.
    t_peak : float
        The time between t=0 and the peak of the waveform to beamform to.
    probe_geometry : jnp.ndarray
        The probe geometry in meters of shape `(n_el, 2)`.
    carrier_frequency : float
        The center frequency of the RF data in Hz.
    sampling_frequency : float
        The sampling frequency in Hz.
    f_number : float
        The f-number to use for the beamforming. The f-number is the ratio of the focal
        length to the aperture size. Elements that are more to the side of the current
        pixel than the f-number are not used in the beamforming.
    tx_apodization : jnp.ndarray
        The apodization of the transmit elements.
    rx_apodization : jnp.ndarray
        The apodization of the receive elements.


    Returns
    -------
    float
        The beamformed value for the pixel.
    """

    tof_corrected = tof_correct_pixel(
        rf_data=rf_data,
        pixel_pos=pixel_pos,
        t0_delays=t0_delays,
        initial_time=initial_time,
        sound_speed=sound_speed,
        sound_speed_lens=sound_speed_lens,
        lens_thickness=lens_thickness,
        t_peak=t_peak,
        probe_geometry=probe_geometry,
        carrier_frequency=carrier_frequency,
        sampling_frequency=sampling_frequency,
        tx_apodization=tx_apodization,
        iq_beamform=True,
    )
    # Traditional f-number mask
    f_number_mask = get_f_number_mask(pixel_pos, probe_geometry, f_number)

    # Compute the inverse of the square root of the magnitude of the TOF corrected data
    tof_corrected_sqrt_inv = 1 / jnp.sqrt(jnp.abs(tof_corrected))

    # Replace NaNs with 1.0. This is needed for values that are 0.
    tof_corrected_sqrt_inv = jnp.nan_to_num(
        tof_corrected_sqrt_inv, nan=1.0, posinf=1.0, neginf=1.0
    )

    # Apply the inverse square root. This changes the unit from [V] to [sqrt(V)]
    tof_corrected_sqrt = tof_corrected * tof_corrected_sqrt_inv

    # Apply the f-number mask and the receive apodization
    tof_corrected = tof_corrected_sqrt * f_number_mask * rx_apodization

    # Compute the correlation matrix
    corr = tof_corrected[:, None] * tof_corrected[None, :]

    # Remove the diagonal
    corr = corr - jnp.eye(corr.shape[0]) * corr

    z = jnp.sum(corr) / 2
    return z


@jit
def dmas_beamform_transmit(
    rf_data,
    pixel_positions,
    probe_geometry,
    t0_delays,
    initial_time,
    sampling_frequency,
    carrier_frequency,
    sound_speed,
    sound_speed_lens,
    lens_thickness,
    t_peak,
    tx_apodization,
    rx_apodization,
    f_number,
):
    """Beamforms a single transmit using the given parameters. The input data must be IQ data. The beamforming can be performed on all transmits or
    a subset of transmits. The beamforming is performed using the Delay-And-Sum (DAS)
    algorithm. The beamforming can be performed before or after the data is converted
    to complex IQ data.

    Parameters
    ----------
    rf_data : jnp.ndarray
        The IQ data to beamform of shape `(n_samples, n_el)`.
    pixel_positions : jnp.ndarray
        The position of the pixel to beamform to in meters of shape `(n_pixels, 2)`.
    probe_geometry : jnp.ndarray
        The probe geometry in meters of shape `(n_el, 2)`.
    t0_delays : jnp.ndarray
        The transmit delays of shape `(n_tx, n_el)`. These are the times between t=0 and
        every element firing in seconds. (t=0 is when the first element fires.)
    initial_time : jnp.ndarray
        The time between t=0 and the first sample being recorded. (t=0 is when the first
        element fires.)
    sampling_frequency : float
        The sampling frequency in Hz.
    carrier_frequency : float
        The center frequency of the RF data in Hz.
    sound_speed : float
        The speed of sound in m/s.
    t_peak : float
        The time between t=0 and the peak of the waveform to beamform to. (t=0 is when
        the first element fires)
    tx_apodization : jnp.ndarray
        The apodization of the transmit elements.
    rx_apodization : jnp.ndarray
        The apodization of the receive elements.
    f_number : float
        The f-number to use for the beamforming. The f-number is the ratio of the focal
        length to the aperture size. Elements that are more to the side of the current
        pixel than the f-number are not used in the beamforming. Default is 3.

    Returns
    -------
    float
        The beamformed value for the pixel.
    """
    return vmap(
        _beamform_pixel,
        in_axes=(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )(
        rf_data,
        pixel_positions,
        t0_delays,
        initial_time,
        sound_speed,
        sound_speed_lens,
        lens_thickness,
        t_peak,
        probe_geometry,
        carrier_frequency,
        sampling_frequency,
        f_number,
        tx_apodization,
        rx_apodization,
    )


def beamform_dmas(
    rf_data,
    pixel_positions: jnp.ndarray,
    probe_geometry: jnp.ndarray,
    t0_delays: jnp.ndarray,
    initial_times: jnp.ndarray,
    sampling_frequency: float,
    carrier_frequency: float,
    sound_speed: float,
    sound_speed_lens: float,
    lens_thickness: float,
    t_peak: jnp.ndarray,
    tx_apodizations: jnp.ndarray,
    rx_apodization: jnp.ndarray,
    f_number: float,
    transmits: jnp.ndarray = None,
    pixel_chunk_size: int = 1048576,
    progress_bar: bool = False,
):
    """Beamforms RF data using the given parameters. The input data can be
    either RF or IQ data. The beamforming can be performed on all transmits or a
    subset of transmits. The beamforming is performed using the Delay-And-Sum (DAS)
    algorithm. The beamforming can be performed before or after the data is
    converted to complex IQ data.

    Parameters
    ----------
    rf_data : jnp.ndarray
        The RF data to beamform of shape `(n_frames, n_tx, n_samples, n_el, 1)`.
    pixel_positions : jnp.ndarray
        The position of the pixel to beamform to in meters of shape `(n_pixels, 2)`.
    probe_geometry : jnp.ndarray
        The probe geometry in meters of shape `(n_el, 2)`.
    t0_delays : jnp.ndarray
        The transmit delays of shape `(n_tx, n_el)`. These are the times between t=0 and
        every element firing in seconds. (t=0 is when the first element fires.)
    initial_times : jnp.ndarray
        The time between t=0 and the first sample being recorded. (t=0 is when the first
        element fires.)
    sampling_frequency : float
        The sampling frequency in Hz.
    carrier_frequency : float
        The center frequency of the RF data in Hz.
    sound_speed : float
        The speed of sound in m/s.
    t_peak : jnp.ndarray
        The time between t=0 and the peak of the waveform to beamform to. (t=0 is when
        the first element fires)
    tx_apodizations : jnp.ndarray
        The apodization of the transmit elements.
    rx_apodization : jnp.ndarray
        The apodization of the receive elements.
    f_number : float
        The f-number to use for the beamforming. The f-number is the ratio of the focal
        length to the aperture size. Elements that are more to the side of the current
        pixel than the f-number are not used in the beamforming. Default is 3.

    Returns
    -------
    jnp.ndarray
        The beamformed image of shape `(n_frames, n_z, n_x)`
    """
    # Perform input error checking
    rf_data = check_standard_rf_or_iq_data(rf_data)
    if rf_data.shape[4] != 1:
        raise ValueError("Only IQ data with one channel is supported")

    check_frequency(carrier_frequency)
    check_frequency(sampling_frequency)
    check_sound_speed(sound_speed)
    check_pos_array(probe_geometry, name="probe_geometry")
    check_t0_delays(t0_delays, transmit_dim=True)
    n_tx = rf_data.shape[1]
    n_pixels = pixel_positions.shape[0]

    # ==================================================================================
    # Interpret the transmits argument
    # ==================================================================================
    if transmits is None:
        transmits = np.arange(n_tx)

    # ==================================================================================
    # Convert to complex IQ data if necessary
    # ==================================================================================
    # Convert to complex IQ data, demodulating if necessary
    iq_data = to_complex_iq(
        rf_data=rf_data,
        carrier_frequency=carrier_frequency,
        sampling_frequency=sampling_frequency,
    )
    input_data = iq_data
    beamformed_dtype = jnp.complex64

    n_frames = rf_data.shape[0]

    # Initialize the beamformed images to zeros
    beamformed_images = jnp.zeros((n_frames, n_pixels), dtype=beamformed_dtype)

    # ==================================================================================
    # Define progress bar functions
    # ==================================================================================
    progbar_func_frames = lambda x: (
        tqdm(x, desc="Beamforming frame", colour="yellow", leave=False)
        if progress_bar and n_frames > 1
        else x
    )
    progbar_func_transmits = lambda x: (
        tqdm(x, desc="Transmit", colour="yellow", leave=False)
        if progress_bar and len(transmits) > 1
        else x
    )
    progbar_func_pixels = lambda x: (
        tqdm(x, desc="Pixel chunks", colour="yellow", leave=False)
        if progress_bar and len(start_indices) > 1
        else x
    )

    for tx in transmits:
        assert 0 <= tx < n_tx, "Transmit index out of bounds"

    # ==================================================================================
    # Define pixel chunks
    # ==================================================================================
    start_indices = (
        jnp.arange(jnp.ceil(n_pixels / pixel_chunk_size).astype(int)) * pixel_chunk_size
    )

    for frame in progbar_func_frames(range(n_frames)):

        # Beamform every transmit individually and sum the results
        for tx in progbar_func_transmits(transmits):
            beamformed_chunks = []
            for ind0 in progbar_func_pixels(start_indices):
                pixel_chunk = pixel_positions[ind0 : ind0 + pixel_chunk_size]
                # Perform beamforming
                beamformed_chunks.append(
                    dmas_beamform_transmit(
                        input_data[frame, tx],
                        pixel_chunk,
                        probe_geometry,
                        t0_delays[tx],
                        initial_times[tx],
                        sampling_frequency,
                        carrier_frequency,
                        sound_speed,
                        sound_speed_lens,
                        lens_thickness,
                        t_peak[tx],
                        tx_apodizations[tx],
                        rx_apodization,
                        f_number=f_number,
                    )
                )

            # Concatenate the beamformed chunks
            beamformed_transmit = jnp.concatenate(beamformed_chunks)

            # Reshape and add to the beamformed images
            beamformed_images = beamformed_images.at[frame].add(beamformed_transmit)

    return beamformed_images
