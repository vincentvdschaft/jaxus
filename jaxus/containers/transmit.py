import numpy as np

import jaxus.utils.log as log
from jaxus.containers.waveform import Waveform


class Transmit:
    """Container class storing the parameters pertaining to the transmit.

    Contains
    --------
    - t0_delays : np.ndarray
        The t0 delays in seconds. [n_el,]
    - tx_apodization : np.ndarray
        The transmit apodization. [n_el,]
    - carrier_frequency : float
        The center frequency of the transmit pulse in Hz.
    - pulse_width : float
        The pulse width of the transmit pulse in seconds.
    """

    def __init__(
        self, t0_delays: np.ndarray, tx_apodization: np.ndarray, waveform: Waveform
    ):
        """Initializes the Transmit object.

        Parameters
        ----------
            t0_delays : np.ndarray
                The t0 delays in seconds. `(n_el,)`
            tx_apodization : np.ndarray
                The transmit apodization. `(n_el,)`
            carrier_frequency : float
                The center frequency of the transmit pulse in Hz.
            pulse_width : float
                The pulse width of the transmit pulse in seconds.
        """
        self._validate_input(t0_delays, tx_apodization, waveform)

        self._t0_delays = t0_delays.astype(np.float32)
        self._tx_apodization = tx_apodization.astype(np.float32)
        self._waveform = waveform

    @property
    def t0_delays(self):
        """The t0 delays in seconds. [n_el,]"""
        return self._t0_delays

    @property
    def tx_apodization(self):
        """The transmit apodization. [n_el,]"""
        return self._tx_apodization

    @property
    def waveform(self):
        """The transmit waveform."""
        return self._waveform

    @property
    def waveform_function(self):
        """The transmit waveform function."""
        return self._waveform.get_waveform_function()

    @property
    def carrier_frequency(self):
        """The center frequency of the transmit pulse in Hz."""
        return self._waveform._carrier_frequency

    @property
    def pulse_width(self):
        """The pulse width of the transmit pulse in seconds."""
        return self._waveform._pulse_width

    @property
    def chirp_rate(self):
        """The chirp rate in Hz/s."""
        try:
            return self._waveform._chirp_rate
        except AttributeError:
            return 0

    def __repr__(self) -> str:
        return f"Transmit({self._waveform})"

    @staticmethod
    def _validate_input(t0_delays, tx_apodization, waveform):
        """Checks if the input is valid.

        Parameters
        ----------
            t0_delays : array_like
                The input t0 delays.
            tx_apodization : array_like
                The input transmit apodization.
            carrier_frequency : float or int
                The input center frequency.
            pulse_width : float or int
                The input pulse width.
        """
        # ------------------------------------------------------------------------------
        # Test t0_delays
        # ------------------------------------------------------------------------------
        # Check if the input is a numpy array
        if not isinstance(t0_delays, np.ndarray):
            raise TypeError(
                "t0_delays must be a numpy array. " f"Got {type(t0_delays)}"
            )

        if t0_delays.ndim != 1:
            raise ValueError(
                "t0_delays must have shape (n_el,). " f"Got {t0_delays.shape}"
            )

        if not t0_delays.dtype in [np.float32, np.float64]:
            raise TypeError(
                "t0_delays must be float32 or float64. " f"Got {t0_delays.dtype}"
            )

        # ------------------------------------------------------------------------------
        # Test tx_apodization
        # ------------------------------------------------------------------------------
        # Check if the input is a numpy array
        if not isinstance(tx_apodization, np.ndarray):
            raise TypeError(
                "tx_apodization must be a numpy array. " f"Got {type(tx_apodization)}"
            )

        if tx_apodization.ndim != 1:
            raise ValueError(
                "tx_apodization must have shape (n_el,). " f"Got {tx_apodization.shape}"
            )

        if not tx_apodization.dtype in [np.float32, np.float64]:
            raise TypeError(
                "tx_apodization must be float32 or float64. "
                f"Got {tx_apodization.dtype}"
            )

        # ------------------------------------------------------------------------------
        # Further checks
        # ------------------------------------------------------------------------------
        # Test if inputs are compatible
        if tx_apodization.shape[0] != t0_delays.shape[0]:
            raise ValueError(
                "tx_apodization must have the same number of elements as t0_delays"
            )

        if np.any(t0_delays < 0):
            raise ValueError("t0_delays must be positive")

        if np.any(tx_apodization < 0):
            raise ValueError("tx_apodization must be positive")

        if np.min(t0_delays) != 0:
            raise ValueError("The smallest t0_delay must be 0")
