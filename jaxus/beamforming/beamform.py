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


class PixelGrid:
    def __init__(self, pixel_positions_cartesian: np.ndarray):
        """Initializes a PixelGrid object.

        ### Args:
            `pixel_positions_cartesian` (`np.ndarray`): The positions of the pixels in
                meters of shape (n_rows, n_cols, 2).
        """

        self.pixel_positions = pixel_positions_cartesian

    @property
    def pixel_positions(self):
        """The positions of all pixels in the beamforming grid in meters of shape
        (n_row, n_col, 2)."""
        return np.copy(self._pixel_positions)

    @pixel_positions.setter
    def pixel_positions(self, value):
        """Sets the pixel positions. The pixel positions must be a 3D array with the
        first dimension being the rows, the second dimension being the columns, and
        the third dimension being the x and z positions."""
        if not isinstance(value, np.ndarray):
            raise TypeError("pixel_positions must be a numpy array.")
        if value.ndim != 3:
            raise ValueError("pixel_positions must be a 3D array.")
        if value.shape[2] != 2:
            raise ValueError("pixel_positions must have size (n_rows, n_cols, 2).")
        self._pixel_positions = value.astype(np.float32)

    @property
    def pixel_positions_flat(self):
        """The positions of all pixels in the beamforming grid in meters of shape
        (n_pixels, 2)."""
        return np.reshape(self.pixel_positions, (-1, 2))

    @property
    def n_pixels(self):
        """The number of pixels in the beamforming grid."""
        return self.pixel_positions.shape[0] * self.pixel_positions.shape[1]

    @property
    def n_cols(self):
        """The number of columns in the pixel grid."""
        return self.pixel_positions.shape[1]

    @property
    def n_rows(self):
        """The number of rows in the pixel grid."""
        return self.pixel_positions.shape[0]

    @property
    def collim(self):
        """The col-axis limits of the pixel grid. in meters. For a cartesian
        grid these values are in meters. For a polar grid these are in radians. For a
        polar grid these are the extreme values at furthest left and right.
        """
        return (self.pixel_positions[0, 0, 0], self.pixel_positions[0, -1, 0])

    @property
    def rowlim(self):
        """The row-axis limits of the pixel grid in meters. For a polar grid
        these are the extreme values at the center"""
        return (self.pixel_positions[0, 0, 1], self.pixel_positions[-1, 0, 1])

    @property
    def xlim(self):
        """The smallest and largest x-position in the grid in meters."""
        xmin = np.min(self.pixel_positions[:, :, 0])
        xmax = np.max(self.pixel_positions[:, :, 0])
        return (xmin, xmax)

    @property
    def zlim(self):
        """The smallest and largest z-position in the grid in meters."""
        zmin = np.min(self.pixel_positions[:, :, 1])
        zmax = np.max(self.pixel_positions[:, :, 1])
        return (zmin, zmax)

    @property
    def extent(self):
        """The extent of the beamformed image. (xmin, xmax, zmax, zmin) in meters."""
        return (
            np.min(self.pixel_positions[:, :, 0]),
            np.max(self.pixel_positions[:, :, 0]),
            np.max(self.pixel_positions[:, :, 1]),
            np.min(self.pixel_positions[:, :, 1]),
        )

    @property
    def extent_mm(self):
        """The extent of the beamformed image in mm."""
        return tuple([val * 1e3 for val in self.extent])


class CartesianPixelGrid(PixelGrid):
    def __init__(self, n_x, n_z, dx_wl, dz_wl, z0, wavelength):
        """Creates a CartesianPixelGrid object. Stores the pixel positions in meters in
        a 2D array of shape (2, n_rows, n_cols).

        ### Args:
            `n_x` (`int`): The number of pixels in the beamforming grid in the
                x-direction.
            `n_z` (`int`): The number of pixels in the beamforming grid in the
                z-direction.
            `dx_wl` (`float`): The pixel size/spacing in the x-direction in wavelengths.
                (Wavelengths are defined as sound_speed/carrier_frequency.)
            `dz_wl` (`float`): The pixel size/spacing in the z-direction in wavelengths.
                (Wavelengths are defined as sound_speed/carrier_frequency.)
            `z0` (`float`): The start-depth of the beamforming plane in meters.
            `wavelength` (`float`): The wavelength to define the grid spacing in meters.
        """

        # Construct the grid of pixel positions
        x_vals = (np.arange(n_x) - n_x / 2) * dx_wl * wavelength
        z_vals = np.arange(n_z) * dz_wl * wavelength + z0

        x_grid, z_grid = np.meshgrid(x_vals, z_vals)

        self.pixel_positions = np.stack((x_grid, z_grid), axis=2)

        super().__init__(self.pixel_positions)


class PolarPixelGrid(PixelGrid):
    def __init__(self, n_ax, n_theta, dax_wl, dtheta_rad, z0, wavelength):

        total_arc = (n_theta - 1) * dtheta_rad

        # Construct the grid of pixel positions
        ax_vals = (np.arange(n_ax) - n_ax / 2) * dax_wl * wavelength + z0
        theta_vals = np.arange(n_theta) * dtheta_rad - total_arc / 2

        ax_grid, theta_grid = np.meshgrid(ax_vals, theta_vals)

        x_grid = ax_grid * np.sin(theta_grid)
        z_grid = ax_grid * np.cos(theta_grid)

        self.pixel_positions = np.stack((x_grid, z_grid))

        super().__init__(self.pixel_positions)


def rf2iq(rf_data, carrier_frequency, sampling_frequency, padding=2048):  #
    """Converts RF data to complex valued IQ data.

    ### Args:
        `rf_data` (`np.ndarray`): The RF data of shape (n_frames, n_tx, n_ax, n_el, n_ch)
        `carrier_frequency` (`float`): The center frequency of the RF data in Hz.
        `sampling_frequency` (`float`): The sampling frequency in Hz.
        `padding` (`int`): The number of samples to pad the RF data with on both sides
            before performing the Hilbert transform. This helps combat edge effects.
            Default is 256.

    ### Returns:
        `iq_data` (`np.ndarray`): The IQ data of shape `(n_frames, n_tx, n_ax, n_el)`
    """
    # ==================================================================================
    # Perform error checking
    # ==================================================================================
    rf_data = check_standard_rf_data(rf_data)
    carrier_frequency = check_frequency(carrier_frequency)
    sampling_frequency = check_frequency(sampling_frequency)

    iq_data = np.zeros(rf_data.shape[:-1], dtype=np.complex64)
    batch_size = 16

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
        Wn = min(0.8 * carrier_frequency / (0.5 * sampling_frequency), 0.5)
        b, a = butter(filter_order, Wn, "low")

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

    ### Args:
        `rf_data` (`np.ndarray`, `jnp.ndarray`): The RF or IQ data to convert of shape
            (n_frames, n_tx, n_samples, n_elements, n_ch).
        `carrier_frequency` (`float`): The center frequency of the RF data in Hz.
        `sampling_frequency` (`float`): The sampling frequency in Hz.

    ### Returns:
        `iq_data` (`np.ndarray`, `jnp.ndarray`): The IQ data of shape
            `(n_frames, n_tx, n_samples, n_elements)`
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
            "rf_data must be of shape (n_frames, n_tx, n_samples, n_elements, n_ch). "
            f"Got shape {rf_data.shape}."
        )


def detect_envelope_beamformed(bf_data, dz_wl):
    """Performs envelope detection on the beamformed data by turning it into IQ-data and
    taking the absolute value.

    ### Args:
        `rf_data` (`np.ndarray`): The RF data to perform envelope detection on of shape
            (n_frames, n_rows, n_cols)
        `dz_wl` (`float`): The pixel size/spacing in the z-direction in wavelengths of
            the beamforming grid. (Wavelengths are defined as
            sound_speed/carrier_frequency.)

    ### Returns:
        `envelope` (`np.ndarray`): The envelope detected RF data of the same shape as
            the input.
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

    iq_data = rf2iq(bf_data[:, None, ..., None], 1, 1 / dz_wl, padding=2048)
    return np.abs(iq_data)[:, 0, :, :]


def log_compress(beamformed: jnp.ndarray, normalize: bool = False):
    """Log-compresses the beamformed image.

    ### Args:
        `beamformed` (`jnp.ndarray`): The beamformed image to log-compress of any shape.
        `normalize` (`bool`): Whether to normalize the beamformed image. Default is
            False.

    ### Returns:
        `beamformed` (`jnp.ndarray`): The log-compressed image of the same shape as the
            input in dB.
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

    ### Arguments:
        `x` (`jnp.ndarray`): The input signal.
        `axis` (`int`): The axis along which to compute the Hilbert transform.
            Default is -1.

    ### Returns:
        `analytical_signal` (`jnp.ndarray`): The analytical signal of `x`.
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

    ### Args:
        `signal` (`np.ndarray`): The signal to find the peak of.
        `sampling_frequency` (`float`): The sampling frequency in Hz.

    ### Returns:
        `t_peak` (`float`): The time between t=0 and the peak of the signal.
    """
    # Perform input error checking
    sampling_frequency = check_frequency(sampling_frequency)

    # Find the peak of the signal
    peak_index = np.argmax(signal)

    # Convert the peak index to time
    t_peak = peak_index / sampling_frequency

    return t_peak


def _tof_correct_pixel(
    rf_data,
    pixel_pos,
    t0_delays,
    initial_time,
    sound_speed,
    t_peak,
    probe_geometry,
    carrier_frequency,
    sampling_frequency,
    iq_beamform=False,
):
    n_ax = rf_data.shape[-2]

    # Compute the distance from the pixel to each element of shape (n_el,)
    dist_to_elements = jnp.linalg.norm(pixel_pos[None] - probe_geometry, axis=1)

    # Compute the transmit and receive times of flight (TOF) of shape (n_el,)
    tof_tx = jnp.min(t0_delays + dist_to_elements / sound_speed)
    tof_rx = dist_to_elements / sound_speed

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

    return jnp.sum(tof_corrected * f_number_mask * rx_apodization)


@partial(jit, static_argnums=(11,))
def das_beamform_transmit(
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


def beamform_das(
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

    progbar_func = lambda x: tqdm(x, desc="Beamforming") if progress_bar else x
    for tx in transmits:
        assert 0 <= tx < n_tx, "Transmit index out of bounds"

    # ==================================================================================
    # Define pixel chunks
    # ==================================================================================
    start_indices = (
        jnp.arange(jnp.ceil(n_pixels / pixel_chunk_size).astype(int)) * pixel_chunk_size
    )

    for frame in progbar_func(range(n_frames)):

        # Beamform every transmit individually and sum the results
        for tx in transmits:
            beamformed_chunks = []
            for ind0 in start_indices:
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


class Beamformer:
    """DAS beamformer class that contains the parameters required for beamforming. The
    beamformer object makes it easy to beamform many frames without having to pass
    all the parameters to the beamforming function every time."""

    def __init__(
        self,
        pixel_grid: PixelGrid,
        probe_geometry: np.ndarray,
        t0_delays: np.ndarray,
        initial_times: np.ndarray,
        sampling_frequency: Union[float, int],
        carrier_frequency: Union[float, int],
        sound_speed: Union[float, int],
        t_peak: Union[np.ndarray, float, int],
        rx_apodization: np.ndarray,
        f_number: Union[float, int] = 3,
        iq_beamform: bool = False,
    ):
        self._beamform_func = None

        self._pixel_grid = None
        self._pixel_positions_flat = None
        self._probe_geometry = None
        self._t0_delays = None
        self._initial_times = None
        self._sampling_frequency = None
        self._carrier_frequency = None
        self._sound_speed = None
        self._t_peak = None
        self._f_number = None
        self._rx_apodization = None
        self._beamform_func = None
        self.iq_beamform = iq_beamform

        Beamformer.configure(
            self,
            pixel_grid=pixel_grid,
            probe_geometry=probe_geometry,
            t0_delays=t0_delays,
            initial_times=initial_times,
            sampling_frequency=sampling_frequency,
            carrier_frequency=carrier_frequency,
            sound_speed=sound_speed,
            t_peak=t_peak,
            rx_apodization=rx_apodization,
            f_number=f_number,
            iq_beamform=iq_beamform,
        )

    # ==================================================================================
    # Properties
    # ==================================================================================
    @property
    def n_x(self):
        """The number of pixels in the beamforming grid in the x-direction."""
        return self._pixel_grid.n_cols

    @property
    def n_z(self):
        """The number of pixels in the beamforming grid in the z-direction."""
        return self._pixel_grid.n_rows

    @property
    def dx_wl(self):
        """The pixel size/spacing in the x-direction in wavelengths. (Wavelengths are
        defined as sound_speed/carrier_frequency.)"""
        return self._dx_wl

    @property
    def dz_wl(self):
        """The pixel size/spacing in the z-direction in wavelengths. (Wavelengths are
        defined as sound_speed/carrier_frequency.)"""
        return self._dz_wl

    @property
    def x_vals(self):
        """The x-axis of the pixel grid in meters."""
        return self._x_vals

    @property
    def z_vals(self):
        """The z-axis of the pixel grid in meters."""
        return self._z_vals

    @property
    def pixel_positions(self):
        """The positions of the pixels in meters of shape (2, n_rows, n_cols). The first
        column is the x position and the second column is the z position."""
        return self._pixel_grid.pixel_positions

    @property
    def probe_geometry(self):
        """The probe geometry in meters of shape (n_elements, 2). The first column is
        the x position and the second column is the z position."""
        return np.copy(self._probe_geometry)

    @property
    def t0_delays(self):
        """The transmit delays of shape (n_tx, n_elements). These are the times between
        t=0 and every element firing in seconds. (t=0 is when the first element fires.)
        t0_delays is always of shape (n_tx, n_el)."""
        return np.copy(self._t0_delays)

    @property
    def initial_times(self):
        """The time between t=0 and the first sample being recorded. (t=0 is when the
        first element fires.)"""
        return np.copy(self._initial_times)

    @property
    def sampling_frequency(self):
        """The sampling frequency in Hz."""
        return self._sampling_frequency

    @property
    def carrier_frequency(self):
        """The center frequency of the RF data in Hz."""
        return self._carrier_frequency

    @property
    def sound_speed(self):
        """The speed of sound in m/s."""
        return self._sound_speed

    @property
    def t_peak(self):
        """The time between t=0 and the peak of the waveform to beamform to.
        (t=0 is when the first element fires)"""
        return self._t_peak

    @property
    def f_number(self):
        """The f-number to use for the beamforming. The f-number is the ratio of the
        focal length to the aperture size. Elements that are more to the side of the
        current pixel than the f-number are not used in the beamforming. Default is 3.
        """
        return self._f_number

    @property
    def rx_apodization(self):
        """The apodization of the receive elements."""
        return np.copy(self._rx_apodization)

    @property
    def n_tx(self):
        """The number of transmits."""
        return self._t0_delays.shape[0]

    @property
    def n_el(self):
        """The number of elements."""
        return self._probe_geometry.shape[0]

    @property
    def shape(self):
        """The shape of the beamformed image."""
        return (self.n_z, self.n_x)

    @property
    def extent(self):
        """The extent of the beamformed image. (xmin, xmax, zmax, zmin) in meters or
        radians, depending on the grid type."""
        return (
            self._pixel_grid.collim[0],
            self._pixel_grid.collim[1],
            self._pixel_grid.rowlim[1],
            self._pixel_grid.rowlim[0],
        )

    @property
    def extent_mm(self):
        """The extent of the beamformed image in mm."""
        xlim = self._pixel_grid.xlim
        zlim = self._pixel_grid.zlim
        return (
            xlim[0] * 1e3,
            xlim[1] * 1e3,
            zlim[1] * 1e3,
            zlim[0] * 1e3,
        )

    def n_pixels(self):
        """The number of pixels in the beamforming grid."""
        return self.n_x * self.n_z

    @property
    def wavelength(self):
        """The wavelength of the center frequency in meters."""
        return self._sound_speed / self._carrier_frequency

    # ==================================================================================
    # Functions
    # ==================================================================================
    def configure(
        self,
        pixel_grid: PixelGrid,
        probe_geometry: np.ndarray,
        t0_delays: np.ndarray,
        initial_times: np.ndarray,
        sampling_frequency: Union[float, int],
        carrier_frequency: Union[float, int],
        sound_speed: Union[float, int],
        t_peak: Union[float, int],
        rx_apodization: np.ndarray,
        f_number: Union[float, int] = 3,
        iq_beamform: bool = False,
    ):

        sampling_frequency = check_frequency(sampling_frequency)
        carrier_frequency = check_frequency(carrier_frequency)
        sound_speed = check_posfloat(sound_speed, name="sound_speed")
        f_number = check_posfloat(f_number, name="f_number")
        probe_geometry = check_pos_array(
            probe_geometry, name="probe_geometry", ax_dim=1
        )
        t0_delays = check_t0_delays(t0_delays)
        initial_times = check_initial_times(initial_times)

        check_shapes_consistent(
            probe_geometry, initial_times, t0_delays, rx_apodization
        )
        n_tx = t0_delays.shape[0]

        if isinstance(t_peak, (int, float)):
            t_peak = np.ones(n_tx) * t_peak
        elif not isinstance(t_peak, np.ndarray):
            raise TypeError("t_peak must be an int, float, or a numpy array.")

        # ==============================================================================
        # Convert numpy arrays to jax arrays
        # ==============================================================================
        probe_geometry = jnp.array(probe_geometry, dtype=jnp.float32)
        t0_delays = jnp.array(t0_delays, dtype=jnp.float32)
        initial_times = jnp.array(initial_times, dtype=jnp.float32)
        rx_apodization = jnp.array(rx_apodization, dtype=jnp.float32)
        carrier_frequency = float(carrier_frequency)
        sampling_frequency = float(sampling_frequency)
        sound_speed = float(sound_speed)
        t_peak = jnp.array(t_peak, dtype=jnp.float32)
        pixel_positions_flat = jnp.array(
            pixel_grid.pixel_positions_flat, dtype=jnp.float32
        )

        # ==============================================================================
        # Assign the parameters to the class
        # ==============================================================================
        self._pixel_grid = pixel_grid
        self._probe_geometry = probe_geometry
        self._t0_delays = t0_delays
        self._initial_times = initial_times
        self._sampling_frequency = sampling_frequency
        self._carrier_frequency = carrier_frequency
        self._sound_speed = sound_speed
        self._t_peak = t_peak
        self._f_number = f_number
        self._rx_apodization = rx_apodization
        self._pixel_positions_flat = pixel_positions_flat
        self._iq_beamform = iq_beamform

    def beamform(self, rf_data: jnp.ndarray):
        """Beamforms RF data to a pixel grid using the parameters of the beamformer.

        ### Args:
            `rf_data` (`jnp.ndarray`): The RF data to beamform of shape
                (n_frames, n_tx, n_samples, n_elements, n_ch).

        ### Returns:
            `beamformed_images` (`jnp.ndarray`): The beamformed images of shape
                (n_frames, n_z, n_x).
        """

        beamformed = beamform_das(
            rf_data=rf_data,
            pixel_positions=self._pixel_positions_flat,
            probe_geometry=self._probe_geometry,
            t0_delays=self._t0_delays,
            initial_times=self._initial_times,
            sampling_frequency=self._sampling_frequency,
            carrier_frequency=self._carrier_frequency,
            sound_speed=self._sound_speed,
            t_peak=self._t_peak,
            rx_apodization=self._rx_apodization,
            f_number=self._f_number,
            iq_beamform=self._iq_beamform,
        )

        beamformed = jnp.reshape(
            beamformed,
            (beamformed.shape[0], self._pixel_grid.n_rows, self._pixel_grid.n_cols),
        )

        return beamformed
