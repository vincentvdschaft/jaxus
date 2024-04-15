"""Container classes for transmit waveforms."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from scipy.signal import butter, filtfilt

import jaxus.utils.log as log


class Waveform:
    """Conatiner class to store the parameters pertaining to the transmit waveform as
    transmitted by a single element."""

    def __init__(self):
        pass

    def get_waveform_function(self):
        pass

    def get_waveform_example(self):
        """Returns a time vector and a waveform example."""
        sampling_frequency = 150e6
        t0 = -5e-6
        t1 = -t0
        N = int((t1 - t0) * sampling_frequency) + 1
        t = np.linspace(t0, t1, N)

        waveform_function = self.get_waveform_function_array()

        y = waveform_function(t)

        return t, y

    def get_waveform_function_array(self):
        """Calls the get_waveform_function() function and returns a function that can
        handle arrays as input."""
        waveform_function = self.get_waveform_function()
        waveform_function = vmap(waveform_function)

        return waveform_function


class Pulse(Waveform):
    """Container class for classical pulse waveforms."""

    def __init__(self, carrier_frequency, pulse_width, chirp_rate, phase):
        self._validate_input(carrier_frequency, pulse_width, chirp_rate, phase)
        self._carrier_frequency = carrier_frequency
        self._pulse_width = pulse_width
        self._chirp_rate = chirp_rate
        self._phase = phase

    @staticmethod
    def _validate_input(carrier_frequency, pulse_width, chirp_rate, phase):
        test_carrier_frequency(carrier_frequency)
        test_pulse_width(pulse_width)

    @property
    def carrier_frequency(self):
        """The center frequency of the transmit pulse in Hz."""
        return self._carrier_frequency

    @property
    def pulse_width(self):
        """The pulse width of the transmit pulse in seconds."""
        return self._pulse_width

    @property
    def chirp_rate(self):
        """The chirp rate in Hz/s."""
        return self._chirp_rate

    @property
    def phase(self):
        """The phase of the transmit pulse in radians."""
        return self._phase

    @property
    def t_peak(self):
        """The time at which the pulse envelope peaks."""
        return self._pulse_width

    def __repr__(self) -> str:
        return (
            f"Pulse(carrier_frequency={self.carrier_frequency*1e-6:.1f}MHz, "
            f"pulse_width={self.pulse_width*1e6:.1f}us)"
        )

    def get_waveform_function(self):
        return get_pulse(
            self._carrier_frequency, self._pulse_width, self._chirp_rate, self._phase
        )


def test_carrier_frequency(carrier_frequency):
    """Checks if the center frequency is valid and warns for strange values."""
    # Test carrier_frequency
    # ------------------------------------------------------------------------------
    # Check if the input is a float or int
    if not isinstance(carrier_frequency, (float, int)):
        raise TypeError(
            "carrier_frequency must be a float. " f"Got {type(carrier_frequency)}"
        )

    # Warnings
    # ------------------------------------------------------------------------------
    # Warn if the center frequency is very low
    if carrier_frequency <= 1000:
        log.warning(f"Center frequency is very low: {carrier_frequency} Hz")


def test_pulse_width(pulse_width):
    """Checks if the pulse width is valid and warns for strange values."""
    # Check if the input is a float or int
    if not isinstance(pulse_width, (float, int)):
        raise TypeError(
            "pulse_width must be a float or int. " f"Got {type(pulse_width)}"
        )
    # Warnings
    # ------------------------------------------------------------------------------
    # Warn if the pulse width is very large
    if pulse_width >= 1e-5:
        log.warning(f"Pulse width is very large: {pulse_width*1e6} us")


def get_pulse(carrier_frequency, pulse_width, chirp_rate=0, phase=0):
    """Returns a function that computes a generalized waveform. The waveform can be
    a chirp by setting the chirp_rate to a nonzero value or a traditional pulse by
    setting the chirp_rate to zero.

    Parameters
    ----------
        carrier_frequency : float
            The carrier frequency.
        pulse_width : float
            The pulse width in seconds.
        chirp_rate : float
            The chirp rate in Hz/s.
        phase : float
            The phase of the waveform in radians.

    Returns
    -------
        function: A function that computes a pulse waveform.
    """

    if not isinstance(carrier_frequency, (float, int)):
        raise TypeError("carrier_frequency must be a float or an int")

    if not isinstance(pulse_width, (float, int)):
        raise TypeError("pulse_width must be a float or an int")

    if not isinstance(chirp_rate, (float, int)):
        raise TypeError("chirp_rate must be a float or an int")

    @jit
    def chirp(t):
        """Computes a pulse waveform.

        Parameters
        ----------
        t : jnp.array
            The time vector.

        Returns
        -------
        np.array
            The pulse waveform sampled at ``t``.
        """
        sigma = (0.5 * pulse_width) / jnp.sqrt(-np.log(0.1))
        t = t - pulse_width
        y = jnp.exp(-((t / sigma) ** 2))
        y *= jnp.sin(2 * jnp.pi * ((carrier_frequency + (chirp_rate * t)) * t) + phase)
        return y

    return chirp


def band_limit(signal, cutoff_low, cutoff_high, sampling_frequency):
    """Band limits a signal by applying a butterworth filter twice.

    Parameters
    ----------
    signal : ndarray
        The signal to be filtered.
    cutoff_low : float
        The lower cutoff frequency in Hz.
    cutoff_high : float
        The upper cutoff frequency in Hz.
    sampling_frequency :float
        The sampling frequency in Hz.

    Returns
    -------
    ndarray
        The filtered signal.
    """
    cutoff_low = cutoff_low / (0.5 * sampling_frequency)
    cutoff_high = cutoff_high / (0.5 * sampling_frequency)

    b, a = butter(N=3, Wn=[cutoff_low, cutoff_high], btype="bandpass")

    filtered = filtfilt(b, a, signal)

    return filtered
