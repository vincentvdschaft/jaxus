import numpy as np

import jaxus.utils.log as log


class Receive:
    """Container class storing the parameters pertaining to the receive.

    Contains
    --------
    float : sampling_frequency
        The sampling frequency of the receive pulse in Hz.
    int : n_ax
        The number of axial samples in an rf line.
    """

    def __init__(self, sampling_frequency, n_ax, initial_time):
        """Initializes the Receive object.

        Parameters
        ----------
        sampling_frequency : float
            The sampling frequency of the receive pulse in Hz.
        n_ax : int
            The number of axial samples in an rf line.
        initial_time : float
            The time of recording the first sample in each rf line.
        """
        self._validate_input(sampling_frequency, n_ax, initial_time)

        self._sampling_frequency = float(sampling_frequency)
        self._n_ax = int(n_ax)
        self._initial_time = float(initial_time)

    @property
    def sampling_frequency(self):
        """The sampling frequency of the receive pulse in Hz."""
        return self._sampling_frequency

    @property
    def n_ax(self):
        """The number of axial samples in an rf line."""
        return self._n_ax

    @property
    def initial_time(self):
        """The time of recording the first sample in each rf line."""
        return self._initial_time

    def __repr__(self) -> str:
        return (
            f"Receive(sampling_frequency={self.sampling_frequency*1e-6:.1f}MHz, "
            f"n_ax={self.n_ax}, "
            f"initial_time={self.initial_time*1e6:.1f}us)"
        )

    @staticmethod
    def _validate_input(sampling_frequency, n_ax, initial_time):
        """Checks if the input is valid.

        Parameters
        ----------
        sampling_frequency : float
            The input sampling frequency.
        n_ax : int
            The input number of axial samples.

        Raises
        ------
            TypeError: If the sampling frequency is not a float.
            TypeError: If the n_ax is not an int.
        """

        # ==============================================================================
        # Check sampling frequency
        # ==============================================================================
        if not isinstance(sampling_frequency, float):
            raise TypeError(
                f"The sampling frequency must be a float. It was {type(sampling_frequency)}."
            )

        if sampling_frequency <= 0:
            raise ValueError(
                f"The sampling frequency must be positive. It was {sampling_frequency}."
            )

        # ==============================================================================
        # Check n_ax
        # ==============================================================================
        if not isinstance(n_ax, int):
            raise TypeError(f"The n_ax must be an int. It was {type(n_ax)}.")

        if n_ax <= 0:
            raise ValueError(f"The n_ax must be positive. It was {n_ax}.")

        # ==============================================================================
        # Check initial_time
        # ==============================================================================
        if not isinstance(initial_time, (float, int)):
            raise TypeError(
                f"The initial time must be a float or int. It was {type(initial_time)}."
            )

        if initial_time < 0:
            raise ValueError(
                f"The initial time must be positive. It was {initial_time}."
            )

        # ==============================================================================
        # Warnings
        # ==============================================================================
        if sampling_frequency < 1000:
            log.warning(
                "The sampling frequency is very low. "
                "This may cause aliasing in the received signal."
            )
