"""Implementation of the Minimum Variance Beamformer.

Source: Adaptive Beamforming Applied to Medical Ultrasound Imaging
        by Johan-Fredrik Synnevag
"""

from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from tqdm import tqdm

from jaxus.utils.checks import (
    check_frequency,
    check_pos_array,
    check_sound_speed,
    check_t0_delays,
)

from .beamform import (
    PixelGrid,
    check_standard_rf_or_iq_data,
    get_f_number_mask,
    rf2iq,
    to_complex_iq,
    tof_correct_pixel,
)


def mv_beamform_pixel(
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
    subaperture_size,
    epsilon=1 / 16,
    iq_beamform=False,
):
    """Beamforms a single pixel of a single frame and single transmit.

    Parameters
    ----------
    rf_data : jnp.ndarray
        The RF or IQ data to beamform of shape `(n_samples, n_el)`.
    pixel_pos : jnp.ndarray
        The position of the pixel to beamform to in meters of shape `(n_pixels, 2)`.
    t0_delays : jnp.ndarray
        The transmit delays of shape `(n_tx, n_el)`. These are the times between t=0 and
        every element firing in seconds. (t=0 is when the first element fires.)
    initial_time : jnp.ndarray
        The time between t=0 and the first sample being recorded. (t=0 is when the first
        element fires.)
    sound_speed : float
        The speed of sound in m/s.
    sound_speed_lens : float
        The speed of sound in the lens in m/s.
    lens_thickness : float
        The thickness of the lens in meters.
    t_peak : float
        The time between t=0 and the peak of the waveform to beamform to. (t=0 is when
        the first element fires)
    probe_geometry : jnp.ndarray
        The probe geometry in meters of shape `(n_el, 2)`.
    carrier_frequency : float
        The center frequency of the RF data in Hz.
    sampling_frequency : float
        The sampling frequency in Hz.
    f_number : float
        The f-number to use for the beamforming. The f-number is the ratio of the focal
        length to the aperture size. Elements that are more to the side of the current
        pixel than the f-number are not used in the beamforming. Default is 3.
    rx_apodization : jnp.ndarray
        The apodization of the receive elements.
    subaperture_size : int
        The size of the subaperture to use for the spatial smoothing. Default is 16.
    tx_apodization : jnp.ndarray, optional
        The apodization of the transmit elements. Default is all ones.
    epsilon : float, optional
        The amount of diagonal loading to use. To increase robustness, an identity
        matrix is added to the covariance matrix multiplied by epsilon/subaperture_size.
        Default is 1/16.
    iq_beamform : bool, optional
        Whether to beamform the IQ data directly. Default is False.

    Returns
    -------
    z : float
        The beamformed value for the pixel.
    """

    tof_corrected = tof_correct_pixel(
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
        tx_apodization,
        iq_beamform,
    )

    f_number_mask = get_f_number_mask(pixel_pos, probe_geometry, f_number)

    tof_corrected = tof_corrected * rx_apodization * f_number_mask

    # ==========================================================================
    # Compute R_l
    # ==========================================================================
    N = tof_corrected.shape[0]
    l = subaperture_size

    # Create a batch of subvectors
    # Each subvector is vec[n:n+L], for n in range(N-L+1)
    indices = jnp.arange(N - l + 1)[:, None] + jnp.arange(l)

    # This will be a (N-L+1) x L matrix
    subvectors = tof_corrected[indices]

    # Vectorized outer product for a batch of subvectors
    outer_product = vmap(lambda x: jnp.outer(x, x))(subvectors)

    # Midpoints of the subaperatures
    mid_indices = indices[:, subaperture_size // 2]
    midpoints = probe_geometry[mid_indices, 0]

    dx = midpoints - pixel_pos[0]
    dz = pixel_pos[1]

    mask = dz / dx < f_number

    outer_product *= mask[:, None, None]

    # Sum the outer products
    R_L = jnp.sum(outer_product, axis=0)

    # Normalize
    R_L = R_L / (N - l + 1)

    # Add epsilon to the diagonal
    R_L = R_L + epsilon * jnp.trace(R_L) * jnp.eye(l)

    # ==========================================================================
    # Compute the weights
    # ==========================================================================
    # The weights are computed as w = (R^-1 @ a) / (a^H @ R^-1 @ a)

    # Compute R^-1 @ a
    R_La = jnp.linalg.solve(R_L, jnp.ones(l))
    # R_La = jnp.ones(l)

    R_La = jnp.nan_to_num(R_La, nan=1.0, posinf=1.0, neginf=1.0)

    # Divide by (a^H @ R^-1 @ a)
    # Since the data is already TOF corrected, a is all ones, thus we can just
    # sum the elements of R_La
    weights = R_La / jnp.sum(R_La)

    weights = jnp.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)

    # ==========================================================================
    # Compute the beamformed value
    # ==========================================================================
    z = vmap(lambda x: jnp.dot(weights.conj().T, x))(subvectors)
    z = jnp.sum(z)
    # Divide by the number of subvectors
    z = z / (N - l + 1)

    return z


def beamform_mv(
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
    subaperture_size: int,
    diagonal_loading: float = 1.0,
    iq_beamform: bool = True,
    transmits: jnp.ndarray = None,
    pixel_chunk_size: int = 1048576,
    progress_bar: bool = False,
):
    """Beamforms a single transmit using the given parameters using minimum variance
    beamforming with spatial smoothing. The input data can be either RF or IQ data. The
    beamforming can be performed on all transmits or a subset of transmits. The
    The beamforming can be performed before or after the data is converted to complex
    IQ data.

    The algorithm is based on the paper "Adaptive Beamforming Applied to Medical
    Ultrasound Imaging" by Johan-Fredrik Synnevag.

    Like in traditional minimum variance beamforming, the algorithm computes the
    covariance matrix of the data and then computes the weights that minimize the
    variance of the beamformed image. However, in this implementation, the covariance
    matrix is computed over a set of subapertures and then averaged. This is known as
    spatial smoothing and improves the robustness of the beamforming.

    To further improve the robustness, diagonal loading is used. This is a technique
    where an identity matrix is added to the covariance matrix multiplied by a
    constant. This constant is the diagonal loading parameter. As suggested in the paper
    by Synnevag, the diagonal loading is set to 1/subaperture_size and can be adjusted
    using the scaling parameter `diagonal_loading`.

    Parameters
    ----------
    rf_data : jnp.ndarray
        The RF or IQ data to beamform of shape `(n_samples, n_el)`.
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
    sound_speed_lens : float
        The speed of sound in the lens in m/s.
    lens_thickness : float
        The thickness of the lens in meters.
    t_peak : jnp.ndarray
        The time between t=0 and the peak of the waveform to beamform to. (t=0 is when
        the first element fires)
    tx_apodizations : jnp.ndarray
        The apodization of the transmit elements of shape `(n_tx, n_el)`.
    rx_apodization : jnp.ndarray
        The apodization of the receive elements.
    f_number : float
        The f-number to use for the beamforming. The f-number is the ratio of the focal
        length to the aperture size. Elements that are more to the side of the current
        pixel than the f-number are not used in the beamforming. Default is 3.
    subaperture_size : int
        The size of the subaperture to use for the spatial smoothing. Default is 16.
    diagonal_loading : float, optional
        The amount of diagonal loading to use. To increase robustness, an identity
        matrix is added to the covariance matrix multiplied by epsilon/subaperture_size.
        Default is 1/16.
    iq_beamform : bool, optional
        Whether to beamform the IQ data directly. Default is True.
    transmits : jnp.ndarray, optional
        The transmits to beamform. Default is all transmits.
    pixel_chunk_size : int, optional
        The size of the pixel chunks to use. Default is 1048576.
    progress_bar : bool
        Whether to display a progress bar. Default is False

    Returns
    -------
    jnp.ndarray
        The beamformed image of shape `(n_frames, n_z, n_x)`
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
                    mv_beamform_transmit(
                        input_data[frame, tx],
                        pixel_positions=pixel_chunk,
                        probe_geometry=probe_geometry,
                        t0_delays=t0_delays[tx],
                        initial_time=initial_times[tx],
                        sampling_frequency=sampling_frequency,
                        carrier_frequency=carrier_frequency,
                        sound_speed=sound_speed,
                        sound_speed_lens=sound_speed_lens,
                        lens_thickness=lens_thickness,
                        t_peak=t_peak[tx],
                        f_number=f_number,
                        tx_apodization=tx_apodizations[tx],
                        rx_apodization=rx_apodization,
                        iq_beamform=iq_beamform,
                        subaperture_size=subaperture_size,
                        epsilon=diagonal_loading,
                    )
                )

            # Concatenate the beamformed chunks
            beamformed_transmit = jnp.concatenate(beamformed_chunks)

            # Reshape and add to the beamformed images
            beamformed_images = beamformed_images.at[frame].add(beamformed_transmit)

    return beamformed_images


@partial(
    jit,
    static_argnums=(14, 15, 16),
)
def mv_beamform_transmit(
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
    subaperture_size,
    epsilon,
    iq_beamform,
):
    """Beamforms a single transmit using the given parameters. The input data can be
    either RF or IQ data. The beamforming can be performed on all transmits or
    a subset of transmits. The beamforming is performed using the Delay-And-Sum (DAS)
    algorithm. The beamforming can be performed before or after the data is converted
    to complex IQ data.

    Parameters
    ----------
    rf_data : jnp.ndarray
        The RF or IQ data to beamform of shape `(n_samples, n_el)`.
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
    sound_speed_lens : float
        The speed of sound in the lens in m/s.
    lens_thickness : float
        The thickness of the lens in meters.
    t_peak : float
        The time between t=0 and the peak of the waveform to beamform to. (t=0 is when
        the first element fires)
    tx_apodization : jnp.ndarray
        The apodization of the transmit elements of shape `(n_el,)`.
    rx_apodization : jnp.ndarray
        The apodization of the receive elements of shape `(n_el,)`.
    f_number : float
        The f-number to use for the beamforming. The f-number is the ratio of the focal
        length to the aperture size. Elements that are more to the side of the current
        pixel than the f-number are not used in the beamforming.
    iq_beamform : bool
        Whether to beamform the IQ data directly. Default is False.

    Returns
    -------
    float
        The beamformed value for the pixel.
    """
    return vmap(
        mv_beamform_pixel,
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
        subaperture_size,
        epsilon,
        iq_beamform,
    )
