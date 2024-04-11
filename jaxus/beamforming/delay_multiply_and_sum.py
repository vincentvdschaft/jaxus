"""This module contains Delay-And-Sum (DAS) beamforming functionality.

The core function of this module is the `beamform` function that performs Delay-And-Sum
beamforming on RF or IQ data.

As a convenience, the `Beamformer` class is provided to store the beamforming parameters
and make it easy to beamform many frames without having to pass all the parameters to
the beamforming function every time.
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

from .beamform import _tof_correct_pixel, rf2iq, to_complex_iq


# Jit and mark iq_beamform as static_argnum, because a different value should trigger
# recompilation of the function
@partial(jit, static_argnums=(11,))
def beamform_pixel(
    rf_data,
    pixel_pos,
    t0_delays,
    initial_time,
    sound_speed,
    t_peak,
    probe_geometry,
    carrier_frequency,
    sampling_frequency,
    f_number,
    rx_apodization,
    iq_beamform=False,
):
    """Beamforms a single pixel of a single frame and single transmit. Further
    processing such as log-compression and envelope detection are not performed.

    ### Args:
        `rf_data` (`jnp.ndarray`): The RF or IQ data to beamform of shape
            `(n_samples, n_elements)`.
        `pixel_pos` (`jnp.ndarray`): The position of the pixel to beamform to in meters
            of shape `(2,)`.
        `t0_delays` (`jnp.ndarray`): The transmit delays of shape (n_tx, n_el). These
            are the times between t=0 and every element firing in seconds. (t=0 is
            when the first element fires.)
        `initial_time` (`jnp.ndarray`): The time between t=0 and the first sample being
            recorded. (t=0 is when the first element fires.)
        `sound_speed` (`float`): The speed of sound in m/s.
        `t_peak` (`float`): The time between t=0 and the peak of the waveform to
            beamform to.
        `probe_geometry` (`jnp.ndarray`): The probe geometry in meters of shape
            (n_el, 2).
        `carrier_frequency` (`float`): The center frequency of the RF data in Hz.
        `sampling_frequency` (`float`): The sampling frequency in Hz.
        `f_number` (`float`): The f-number to use for the beamforming. The f-number is
            the ratio of the focal length to the aperture size. Elements that are more
            to the side of the current pixel than the f-number are not used in the
            beamforming.
        `rx_apodization` (`jnp.ndarray`): The apodization of the receive elements.
        `iq_beamform` (`bool`): Set to True to do demodulation first and then beamform
            the IQ data. Set to False to beamform the RF data directly. In this case
            envelope detection is done after beamforming. Default is False.


    ### Returns:
        `bf_value` (`float`): The beamformed value for the pixel.
    """

    tof_corrected = _tof_correct_pixel(
        rf_data,
        pixel_pos,
        t0_delays,
        initial_time,
        sound_speed,
        t_peak,
        probe_geometry,
        carrier_frequency,
        sampling_frequency,
        iq_beamform,
    )
    # Traditional f-number mask
    f_number_mask = get_custom_f_number_mask(pixel_pos, probe_geometry, f_number)

    tof_corrected = tof_corrected * f_number_mask * rx_apodization

    # Compute the correlation matrix
    corr = tof_corrected[:, None] * tof_corrected[None, :]

    # Take the square root of the correlation matrix
    corr_sqrt = jnp.sqrt(jnp.abs(corr))

    # Replace NaNs and Infs with ones
    corr_sqrt_inv = jnp.nan_to_num(1 / corr_sqrt, nan=1.0, posinf=1.0, neginf=1.0)

    # Remove the diagonal
    corr = corr - jnp.eye(corr.shape[0]) * corr

    z = jnp.sum(corr / corr_sqrt_inv) / 2
    return z


@partial(jit, static_argnums=(11,))
def dmas_beamform_transmit(
    rf_data,
    pixel_positions,
    probe_geometry,
    t0_delays,
    initial_time,
    sampling_frequency,
    carrier_frequency,
    sound_speed,
    t_peak,
    rx_apodization,
    f_number,
    iq_beamform,
):
    """Beamforms a single transmit using the given parameters. The input data can be
    either RF or IQ data. The beamforming can be performed on all transmits or
    a subset of transmits. The beamforming is performed using the Delay-And-Sum (DAS)
    algorithm. The beamforming can be performed before or after the data is converted
    to complex IQ data.

    ### Parameters:
        `rf_data` (`jnp.ndarray`): The RF or IQ data to beamform of shape
            `(n_samples, n_elements)`.
        `pixel_positions` (`jnp.ndarray`): The position of the pixel to beamform to in
            meters of shape `(n_pixels, 2)`.
        `probe_geometry` (`jnp.ndarray`): The probe geometry in meters of shape
            (n_elements, 2).
        `t0_delays` (`jnp.ndarray`): The transmit delays of shape (n_tx, n_el). These are
            the times between t=0 and every element firing in seconds. (t=0 is when the
            first element fires.)
        `initial_time` (`jnp.ndarray`): The time between t=0 and the first sample being
            recorded. (t=0 is when the first element fires.)
        `sampling_frequency` (`float`): The sampling frequency in Hz.
        `carrier_frequency` (`float`): The center frequency of the RF data in Hz.
        `sound_speed` (`float`): The speed of sound in m/s.
        `t_peak` (`float`): The time between t=0 and the peak of the waveform to beamform
            to. (t=0 is when the first element fires)
        `rx_apodization` (`jnp.ndarray`): The apodization of the receive elements.
        `f_number` (`float`): The f-number to use for the beamforming. The f-number is the
            ratio of the focal length to the aperture size. Elements that are more to the
            side of the current pixel than the f-number are not used in the beamforming.
            Default is 3.
        `iq_beamform` (`bool`): Whether to beamform the IQ data directly. Default is False.

    ### Returns:
        `bf_value` (`float`): The beamformed value for the pixel.
    """
    return vmap(
        beamform_pixel,
        in_axes=(None, 0, None, None, None, None, None, None, None, None, None, None),
    )(
        rf_data,
        pixel_positions,
        t0_delays,
        initial_time,
        sound_speed,
        t_peak,
        probe_geometry,
        carrier_frequency,
        sampling_frequency,
        f_number,
        rx_apodization,
        iq_beamform,
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
    t_peak: jnp.ndarray,
    rx_apodization: jnp.ndarray,
    f_number: float,
    iq_beamform: bool,
    transmits: jnp.ndarray = None,
    pixel_chunk_size: int = 1048576,
    progress_bar: bool = False,
):
    """Beamforms RF data using the given parameters. The input data can be
    either RF or IQ data. The beamforming can be performed on all transmits or a
    subset of transmits. The beamforming is performed using the Delay-And-Sum (DAS)
    algorithm. The beamforming can be performed before or after the data is
    converted to complex IQ data.

    ### Args:
        `rf_data` (`jnp.ndarray`): The RF or IQ data to beamform of shape
            `(n_frames, n_tx, n_samples, n_elements, n_ch)`.
        `pixel_positions` (`jnp.ndarray`): The position of the pixel to beamform to in
            meters of shape `(n_pixels, 2)`.
        `probe_geometry` (`jnp.ndarray`): The probe geometry in meters of shape
            (n_elements, 2).
        `t0_delays` (`jnp.ndarray`): The transmit delays of shape (n_tx, n_el). These
            are the times between t=0 and every element firing in seconds. (t=0 is when
            the first element fires.)
        `initial_times` (`jnp.ndarray`): The time between t=0 and the first sample being
            recorded. (t=0 is when the first element fires.)
        `sampling_frequency` (`float`): The sampling frequency in Hz.
        `carrier_frequency` (`float`): The center frequency of the RF data in Hz.
        `sound_speed` (`float`): The speed of sound in m/s.
        `t_peak` (`jnp.ndarray`): The time between t=0 and the peak of the waveform to
            beamform to. (t=0 is when the first element fires)
        `rx_apodization` (`jnp.ndarray`): The apodization of the receive elements.
        `f_number` (`float`): The f-number to use for the beamforming. The f-number is
            the ratio of the focal length to the aperture size. Elements that are more
            to the side of the current pixel than the f-number are not used in the
            beamforming. Default is 3.
        `iq_beamform` (`bool`): Whether to beamform the IQ data directly. Default is
            False.
        `transmits` (`jnp.ndarray`): The transmits to beamform. Default is None, which
            means all transmits are beamformed.
        `pixel_chunk_size` (`int`): The number of pixels to beamform at once. Default is
            1048576.
        `progress_bar` (`bool`): Whether to display a progress bar. Default is False

    ### Returns:
        `bf` (`jnp.ndarray`): The beamformed image of shape
            `(n_frames, n_z, n_x)`
    """
    # Perform input error checking
    rf_data = check_standard_rf_or_iq_data(rf_data)
    check_frequency(carrier_frequency)
    check_frequency(sampling_frequency)
    check_sound_speed(sound_speed)
    check_pos_array(probe_geometry, name="probe_geometry")
    check_t0_delays(t0_delays, transmit_dim=True)
    n_tx = rf_data.shape[1]
    n_pixels = pixel_positions.shape[0]

    # Check if iq_beamform and rf_data are compatible
    if not iq_beamform and rf_data.shape[-1] != 1:
        raise ValueError(
            "iq_beamform is False and rf_data has more than one channel. "
            "This is not allowed. Set iq_beamform to True or supply RF data with "
            "only one channel."
        )

    # ==================================================================================
    # Interpret the transmits argument
    # ==================================================================================
    if transmits is None:
        transmits = np.arange(n_tx)

    # ==================================================================================
    # Convert to complex IQ data if necessary
    # ==================================================================================
    if iq_beamform:
        # Convert to complex IQ data, demodulating if necessary
        iq_data = to_complex_iq(
            rf_data=rf_data,
            carrier_frequency=carrier_frequency,
            sampling_frequency=sampling_frequency,
        )
        input_data = iq_data
        beamformed_dtype = jnp.complex64
    else:
        input_data = rf_data[..., 0]
        beamformed_dtype = jnp.float32

    n_frames = rf_data.shape[0]

    # Initialize the beamformed images to zeros
    beamformed_images = jnp.zeros((n_frames, n_pixels), dtype=beamformed_dtype)

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
                        t_peak[tx],
                        rx_apodization,
                        f_number=f_number,
                        iq_beamform=iq_beamform,
                    )
                )

            # Concatenate the beamformed chunks
            beamformed_transmit = jnp.concatenate(beamformed_chunks)

            # Reshape and add to the beamformed images
            beamformed_images = beamformed_images.at[frame].add(beamformed_transmit)

    return beamformed_images


def get_f_number_mask(pixel_pos, probe_geometry, f_number):
    """Computes the f-number mask for a pixel."""
    return jnp.abs(probe_geometry[:, 0] - pixel_pos[0]) < (pixel_pos[1] / f_number)


def get_custom_f_number_mask(pixel_pos, probe_geometry, f_number):
    """Computes the f-number mask for a pixel. This is not a traditional f-number mask
    but a custom mask that has a smooth transition from 1 to 0."""
    return jnp.exp(
        -(
            (
                np.sqrt(np.log(2))
                * (
                    jnp.abs(probe_geometry[:, 0] - pixel_pos[0])
                    / (pixel_pos[1] / f_number)
                )
            )
            ** 2
        )
    )