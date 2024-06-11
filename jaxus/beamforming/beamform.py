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

from .pixelgrid import PixelGrid
from .lens_correction import compute_lensed_travel_time


def beamform_das(
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

    Parameters
    ----------
    rf_data : jnp.ndarray
        The RF or IQ data to beamform of shape
        `(n_frames, n_tx, n_samples, n_el, n_ch)`.
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
        The apodization of the receive elements of shape `(n_el,)`.
    f_number : float
        The f-number to use for the beamforming. The f-number is the ratio of the focal
        length to the aperture size. Elements that are more to the side of the current
        pixel than the f-number are not used in the beamforming. Default is 3.
    iq_beamform : bool
        Whether to beamform the IQ data directly. Default is False.
    transmits : jnp.ndarray
        The transmits to beamform. Default is None, which means all transmits are
        beamformed.
    pixel_chunk_size : int
        The number of pixels to beamform at once. Default is 1048576.
    progress_bar : bool
        Whether to display a progress bar. Default is False

    Returns
    -------
    bf : jnp.ndarray
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
                    das_beamform_transmit(
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
                        iq_beamform=iq_beamform,
                    )
                )

            # Concatenate the beamformed chunks
            beamformed_transmit = jnp.concatenate(beamformed_chunks)

            # Reshape and add to the beamformed images
            beamformed_images = beamformed_images.at[frame].add(beamformed_transmit)

    return beamformed_images


def rf2iq(rf_data, carrier_frequency, sampling_frequency, bandwidth=None, padding=512):
    """Converts RF data to complex valued IQ data.

    Parameters
    ----------
    rf_data : np.ndarray
        The RF data of shape `(n_frames, n_tx, n_ax, n_el, n_ch)`
    carrier_frequency : float
        The center frequency of the RF data in Hz.
    sampling_frequency : float
        The sampling frequency in Hz.
    bandwidth : float
        The bandwidth of the RF data, normalized to the Nyquist frequency (0.5 fs).
        Should be between 0 and 1.0, where 1.0 corresponds to the full bandwidth up to
        the Nyquist frequency. Default is 0.5.
    padding : int, default=512
        The number of samples to pad the RF data with on both sides before performing
        the Hilbert transform. This helps combat edge effects.

    Returns
    -------
    iq_data : np.ndarray
        The IQ data of shape `(n_frames, n_tx, n_ax, n_el)`
    """
    # ==================================================================================
    # Perform error checking
    # ==================================================================================
    rf_data = check_standard_rf_data(rf_data)
    carrier_frequency = check_frequency(carrier_frequency)
    sampling_frequency = check_frequency(sampling_frequency)

    iq_data = np.zeros(rf_data.shape[:-1], dtype=np.complex64)
    batch_size = 16

    if bandwidth is None:
        bandwidth = 2 * carrier_frequency / (0.5 * sampling_frequency)

    # Remove the channel dimension
    rf_data = rf_data[..., 0]

    n_ax = rf_data.shape[2] + 2 * padding
    for batch in range(0, rf_data.shape[0], batch_size):
        ind0 = batch
        ind1 = min(batch + batch_size, rf_data.shape[0])
        # ==================================================================================
        # Convert the RF data to IQ data
        # ==================================================================================
        rf_data_batch = rf_data[ind0:ind1]

        # Pad the RF data with 256 zeros to avoid edge effects from the Hilbert transform
        rf_data_batch = np.pad(
            rf_data_batch, ((0, 0), (0, 0), (padding, padding), (0, 0))
        )

        # Apply the Hilbert transform and obtain the analytic signal (cutting of the negative
        # frequencies). This allows us to shift the signal to baseband by multiplying with
        # exp(-j*2*pi*f0*t) where f0 is the center frequency.
        rf_data_batch = hilbert(rf_data_batch, axis=2)

        # Shift the signal to baseband
        t = np.arange(n_ax) / sampling_frequency
        # Add dummy dimensions to t to allow broadcasting
        t = t[None, None, :, None]

        iq_data_batch = rf_data_batch * np.exp(-1j * 2 * np.pi * carrier_frequency * t)

        # Low-pass filter to only retain the baseband signal
        filter_order = 5
        # We have shifted the signal to baseband. This means that the highest frequency
        # in the signal is now half the bandwidth. We can therefore low-pass filter the
        # signal with a cutoff frequency of bandwidth / 2.
        critical_freq = bandwidth / 2
        b, a = butter(filter_order, critical_freq, "low")

        # Filter using a forward backward filter to prevent phase shift
        iq_data_batch = filtfilt(b, a, iq_data_batch, axis=2)

        # Remove the padding and store the result
        iq_data[ind0:ind1] = iq_data_batch[:, :, padding:-padding]

    return iq_data


def to_complex_iq(
    rf_data: Union[np.ndarray, jax.numpy.ndarray],
    carrier_frequency: Union[float, int],
    sampling_frequency: Union[float, int],
):
    """Converts RF data or 2-channel IQ data to complex valued IQ data.

    Parameters
    ----------
    rf_data : np.ndarray or jnp.ndarray
        The RF or IQ data to convert of shape
        `(n_frames, n_tx, n_samples, n_el, n_ch)`.
    carrier_frequency : float
        The center frequency of the RF data in Hz.
    sampling_frequency : float
        The sampling frequency in Hz.

    Returns
    iq_data : np.ndarray : jnp.ndarray
        The IQ data of shape `(n_frames, n_tx, n_samples, n_el)`
    """
    # Input error checking
    rf_data = check_standard_rf_or_iq_data(rf_data)
    carrier_frequency = check_frequency(carrier_frequency)
    sampling_frequency = check_frequency(sampling_frequency)

    # If the data is 5d and has 1 channel we assume it is RF data and convert it to IQ
    # by demodulating it to baseband
    if rf_data.shape[4] == 1:
        return rf2iq(rf_data, carrier_frequency, sampling_frequency)

    # If the data is 5d and has 2 channels we assume it is already IQ data and only have
    # to turn it from 2-channel into complex numbers
    elif rf_data.shape[4] == 2:
        return rf_data[..., 0] + 1j * rf_data[..., 1]

    else:
        raise ValueError(
            "rf_data must be of shape (n_frames, n_tx, n_samples, n_el, n_ch). "
            f"Got shape {rf_data.shape}."
        )


def detect_envelope_beamformed(bf_data, dz_wl):
    """Performs envelope detection on the beamformed data by turning it into IQ-data and
    taking the absolute value.

    Parameters
    ----------
    rf_data : np.ndarray
        The RF data to perform envelope detection on of shape
        `(n_frames, n_rows, n_cols)`.
    dz_wl : float
        The pixel size/spacing in the z-direction in wavelengths of the beamforming
        grid. (Wavelengths are defined as sound_speed/carrier_frequency.)

    Returns
    -------
    np.ndarray
        The envelope detected RF data of the same shape as the input.
    """
    if not isinstance(bf_data, (np.ndarray, jnp.ndarray)):
        raise TypeError(f"bf_data must be a ndarray. Got {type(bf_data)}.")
    if not bf_data.ndim == 3:
        raise ValueError(
            "bf_data must be a 3D array of shape (n_frames, n_rows, n_cols). "
            f"Got shape {bf_data.shape}."
        )
    if not isinstance(dz_wl, (float, int)):
        raise TypeError(f"dz_wl must be a float or int. Got {type(dz_wl)}.")
    if dz_wl <= 0:
        raise ValueError(f"dz_wl must be positive. Got {dz_wl}.")

    # If the beamformed data is complex IQ data we can just take the absolute value
    if np.iscomplexobj(bf_data):
        return np.abs(bf_data)

    if dz_wl >= 0.25:
        # Apply initial filtering to remove high frequencies. This prevents streaks in
        # the Hilbert transform image when the beamforming grid is a bit too coarse.
        b, a = butter(2, 0.8, "low")
        bf_data = filtfilt(b, a, bf_data, axis=1)

    iq_data = rf2iq(
        bf_data[:, None, ..., None],
        carrier_frequency=1,
        sampling_frequency=1 / dz_wl,
        bandwidth=1.0,
        padding=512,
    )
    return np.abs(iq_data)[:, 0, :, :]


def log_compress(beamformed: jnp.ndarray, normalize: bool = False):
    """Log-compresses the beamformed image.

    Parameters
    ----------
    beamformed : jnp.ndarray
        The beamformed image to log-compress of any shape.
    normalize : bool, default=False
        Whether to normalize the beamformed image.

    Returns
    -------
    beamformed : jnp.ndarray
        The log-compressed image of the same shape as the input in dB.
    """
    if not isinstance(beamformed, (jnp.ndarray, np.ndarray)):
        raise TypeError(f"beamformed must be a ndarray. Got {type(beamformed)}.")
    if not isinstance(normalize, bool):
        raise TypeError(f"normalize must be a bool. Got {type(normalize)}.")

    beamformed = jnp.abs(beamformed)
    if normalize:
        beamformed = beamformed / jnp.clip(jnp.max(beamformed), 1e-12)
    beamformed = 20 * jnp.log10(beamformed + 1e-12)

    return beamformed


def hilbert_get_analytical_signal(x: jnp.ndarray, axis=-1) -> jnp.ndarray:
    """Returns the analytical signal of `x` using the Hilbert transform.

    Parameters
    ----------
    x : jnp.ndarray
        The input signal.
    axis : int, default=-1
        The axis along which to compute the Hilbert transform.

    Returns
    -------
    analytical_signal : jnp.ndarray
        The analytical signal of `x`.
    """
    size = x.shape[axis]
    size_power_of_two = 2 ** jnp.ceil(jnp.log2(size)).astype(jnp.int32)

    x_fft = jnp.fft.fft(x, size_power_of_two, axis=axis)
    multiplier = jnp.zeros(size_power_of_two, dtype=jnp.complex64)
    # Set the positive frequencies to 2
    multiplier = jax.ops.index_update(multiplier, jax.ops.index[: size // 2 + 1], 2)
    # Set the DC frequency to 1
    multiplier = jax.ops.index_update(multiplier, jax.ops.index[0], 1)

    x_fft = x_fft * multiplier

    x_analytical = jnp.fft.ifft(x_fft, axis=axis)

    return x_analytical[..., :size]


def find_t_peak(signal, sampling_frequency=250e6):
    """Finds the peak of the signal and returns the time between t=0 and the peak.

    Parameters
    signal : np.ndarray
        The signal to find the peak of.
    sampling_frequency : float
        The sampling frequency in Hz.

    Returns
    float
        The time between t=0 and the peak of the signal.
    """
    # Perform input error checking
    sampling_frequency = check_frequency(sampling_frequency)

    # Find the peak of the signal
    peak_index = np.argmax(signal)

    # Convert the peak index to time
    t_peak = peak_index / sampling_frequency

    return t_peak


def tof_correct_pixel(
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
    iq_beamform=False,
):
    """Performs time-of-flight correction for a single pixel. The RF data is indexed
    along the axial dimension to find the samples left and right of the exact
    time-of-flight for each element. The samples are then interpolated to find the
    interpolated sample.

    Parameters
    ----------
    rf_data : jnp.ndarray
        The RF or IQ data to beamform of shape `(n_samples, n_el)`.
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
    sound_speed_lens : float
        The speed of sound in the lens in m/s.
    lens_thickness : float
        The thickness of the lens in meters.
    t_peak : float
        The time between t=0 and the peak of the waveform to beamform to.
    probe_geometry : jnp.ndarray
        The probe geometry in meters of shape (n_el, 2).
    carrier_frequency : float
        The center frequency of the RF data in Hz.
    sampling_frequency : float
        The sampling frequency in Hz.
    tx_apodization : jnp.ndarray
        The apodization of the transmit elements.
    iq_beamform : bool, default=False
        Set to True for IQ beamforming. In this case the function will perform phase
        rotation.

    Returns
    -------
    jnp.ndarray
        The interpolated RF or IQ data of shape `(n_el,)`.
    """

    n_ax = rf_data.shape[-2]

    # Compute the distance from the pixel to each element of shape (n_el,)
    travel_time = vmap(compute_lensed_travel_time, in_axes=(0, None, None, None, None))(
        probe_geometry,
        pixel_pos,
        lens_thickness,
        sound_speed_lens,
        sound_speed,
    )

    # This would be the line without lens correction
    # dist_to_elements = jnp.linalg.norm(pixel_pos[None] - probe_geometry, axis=1)

    offset = jnp.where(tx_apodization > 0.0, 0.0, 10.0)

    # Compute the transmit and receive times of flight (TOF) of shape (n_el,)
    tof_tx = jnp.min(t0_delays + travel_time + offset)
    tof_rx = travel_time

    # Compute the float sample index of the TOF of shape (n_el,)
    t_sample = tof_tx + tof_rx + t_peak - initial_time
    sample_index = t_sample * sampling_frequency

    # Compute the actual sample indices before and after the float sample index
    sample_min = jnp.floor(sample_index).astype(jnp.int32)
    sample_max = jnp.ceil(sample_index).astype(jnp.int32)

    # Clip the sample indices to the valid range
    sample_min_clipped = jnp.clip(sample_min, 0, n_ax - 1)
    sample_max_clipped = jnp.clip(sample_max, 0, n_ax - 1)

    # Index the samples along the element axis using take_along_axis
    rf_min = jnp.take_along_axis(rf_data, sample_min_clipped[None], axis=0)[0]
    rf_max = jnp.take_along_axis(rf_data, sample_max_clipped[None], axis=0)[0]

    # Compute the convex combination of the two samples
    alpha = sample_index - sample_min
    rf_interp = (1 - alpha) * rf_min + alpha * rf_max

    # Replace out of bounds indices with zeros
    rf_interp = jnp.where(sample_index < 0.0, 0.0, rf_interp)
    rf_interp = jnp.where(sample_index >= rf_data.shape[0] - 1, 0.0, rf_interp)

    # When doing IQ beamforming, just indexing the samples is not enough to
    # achieve a proper delay. We need to correct the phase by doing phase
    # rotation.
    if iq_beamform:
        # Apply phase rotation to beamform the IQ data directly
        phase = jnp.exp(1j * 2 * jnp.pi * carrier_frequency * (tof_tx + tof_rx))
        rf_interp *= phase

    return rf_interp


# Jit and mark iq_beamform as static_argnum, because a different value should trigger
# recompilation of the function
@partial(jit, static_argnums=(14,))
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
    iq_beamform=False,
):
    """Beamforms a single pixel of a single frame and single transmit. Further
    processing such as log-compression and envelope detection are not performed.

    This function does time-of-flight correction and then applies a f-number mask to
    the beamformed data.

    Parameters
    ----------
    rf_data : jnp.ndarray
        The RF or IQ data to beamform of shape `(n_samples, n_el)`.
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
    sound_speed_lens : float
        The speed of sound in the lens in m/s.
    lens_thickness : float
        The thickness of the lens in meters.
    t_peak : float
        The time between t=0 and the peak of the waveform to beamform to.
    probe_geometry : jnp.ndarray
        The probe geometry in meters of shape (n_el, 2).
    carrier_frequency : float
        The center frequency of the RF data in Hz.
    sampling_frequency : float
        The sampling frequency in Hz.
    f_number : float
        The f-number to use for the beamforming. The f-number is the ratio of the focal
        length to the aperture size. Elements that are more to the side of the current
        pixel than the f-number are not used in the beamforming.
    tx_apodization : jnp.ndarray
        The transmit apodization of the transmit elements.
    rx_apodization : jnp.ndarray
        The apodization of the receive elements.
    iq_beamform : bool, default=False
        Set to True for IQ beamforming. In this case the function will perform phase
        rotation.
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
        iq_beamform=iq_beamform,
    )

    # Traditional f-number mask
    f_number_mask = get_f_number_mask(pixel_pos, probe_geometry, f_number)

    return jnp.sum(tof_corrected * f_number_mask * rx_apodization)


@partial(jit, static_argnums=(14,))
def das_beamform_transmit(
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
        The position of the pixels to beamform to in meters of shape `(n_pixels, 2)`.
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
    t_peak : jnp.ndarray
        The time between t=0 and the peak of the waveform to beamform to. (t=0 is when
        the first element fires)
    tx_apodization : jnp.ndarray
        The apodization of the transmit elements.
    rx_apodization : jnp.ndarray
        The apodization of the receive elements.
    f_number : float
        The f-number to use for the beamforming. The f-number is the ratio of the focal
        length to the aperture size. Elements that are more to the side of the current
        pixel than the f-number are not used in the beamforming.
    iq_beamform : bool
        Whether to beamform the IQ data directly. Default is False.

    Returns
    -------
    jnp.ndarray
        The beamformed image of shape `(n_pixels,)`.
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
        iq_beamform,
    )


def rect(theta):
    """Computes the rectangular window for a given angle, where the windows is 1 at theta=0
    and 0 at theta=pi/2."""
    return 1.0


def hann(theta):
    """Computes the Hann window for a given angle, where the windows is 1 at theta=0
    and 0 at theta=pi/2."""
    return jnp.cos(theta) ** 2


def tukey(theta, alpha=0.8):
    """Computes the Tukey window for a given angle, where the windows is 1 at theta=0
    and 0 at theta=pi/2. The parameter alpha controls the fraction of the window that
    is tapered. alpha=0 corresponds to a rectangular window and alpha=1 corresponds to
    a Hann window.
    """
    theta = jnp.abs(theta)
    return jnp.where(
        theta <= (1 - alpha) * jnp.pi / 2,
        1.0,
        jnp.cos((theta - (1 - alpha) * jnp.pi / 2) / alpha) ** 2,
    )


@partial(jit, static_argnums=(2, 3))
def get_f_number_mask(pixel_pos, probe_geometry, f_number, window_fn=rect):
    """Computes the f-number mask for a pixel for all elements in the probe geometry
    with a given f-number and window function.

    Parameters
    ----------
    pixel_pos : ndarray, shape (2,)
        The position of the pixel in the x-z plane.
    probe_geometry : ndarray, shape (n_elements, 2)
        The positions of the probe elements in the x-z plane.
    f_number : float
        The f-number to use. This is the ratio of the depth over the aperture size.
    window_fn : callable, optional
        The window function to use. This function should accept an angle in radians and
        return a value between 0 and 1. The window should be defined for angles between
        -pi/2 and pi/2.

    Returns
    -------
    mask : ndarray, shape (n_elements,)
        The mask for the pixel for each element in the probe geometry.
    """

    # Compute the angle between the pixel and the probe element positions
    theta = jnp.arctan(jnp.abs(probe_geometry[:, 0] - pixel_pos[0]) / pixel_pos[1])

    # Handle the case where the pixel is at the same height as the probe elements
    theta = jnp.nan_to_num(theta, nan=jnp.pi / 2)

    # Compute the angle at the edge of the cone. This is where the mask should start
    # being zero.
    theta_max = jnp.arctan(0.5 / f_number)

    # Apply the window function. It is scaled from [-pi/2, pi/2] to
    # [-theta_max, theta_max]
    mask_val = window_fn(theta * jnp.pi / 2 / theta_max)

    # Set values outside the cone to zero
    mask_val = jnp.where(theta <= theta_max, mask_val, 0.0)

    return mask_val
