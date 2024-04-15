import numpy as np

import jaxus.utils.log as log


class Medium:
    """Container class storing the parameters pertaining to the medium.

    Contains
    --------
    - scatterer_positions : np.ndarray
        The scatterer positions in meters. (n_scatterers, 2)
    - scatterer_amplitudes : np.ndarray
        The scatterer amplitudes in meters.
        (n_scatterers,)
    - sound_speed : float
        The speed of sound in the medium in m/s.
    - n_z : int
        The number of grid points in the z direction. Can be optionally set with the
        set_grid_size method.
    - n_x : int
        The number of grid points in the x direction. Can be optionally set with the
        set_grid_size method.
    """

    def __init__(
        self,
        scatterer_positions,
        scatterer_amplitudes,
        sound_speed,
        attenuation_coefficient=0.0,
    ):
        """Initializes the Medium object.

        Parameters
        ----------
        scatterer_positions : np.ndarray
            The scatterer positions in meters. (2, n_scatterers)
        scatterer_amplitudes : np.ndarray
            The scatterer amplitudes in meters. (n_scatterers)
        sound_speed : float
            The speed of sound in the medium in m/s.
        """
        self._validate_input(
            scatterer_positions,
            scatterer_amplitudes,
            sound_speed,
            attenuation_coefficient,
        )

        self._scatterer_positions = scatterer_positions.astype(np.float32)
        self._scatterer_amplitudes = scatterer_amplitudes.astype(np.float32)
        self._sound_speed = float(sound_speed)
        self._n_z = None
        self._n_x = None
        self.attenuation_coefficient = float(attenuation_coefficient)

    @property
    def scatterer_positions(self):
        """The scatterer positions in meters. [2, n_scatterers]"""
        return self._scatterer_positions

    @property
    def scatterer_amplitudes(self):
        """The scatterer amplitudes in meters. [n_scatterers]"""
        return self._scatterer_amplitudes

    @property
    def sound_speed(self):
        """The speed of sound in the medium in m/s."""
        return self._sound_speed

    @property
    def n_z(self):
        """The number of grid points in the z direction."""
        return self._n_z

    @property
    def n_x(self):
        """The number of grid points in the x direction."""
        return self._n_x

    def __repr__(self) -> str:
        return (
            f"Medium(n_scatterers={self.scatterer_positions.shape[1]}, "
            f"sound_speed={self.sound_speed}m/s)"
        )

    def set_grid_size(self, n_z, n_x):
        """Sets the grid size.

        Parameters
        ----------
        n_z : int
            The number of grid points in the z direction.
        n_x : int
            The number of grid points in the x direction.
        """
        if not isinstance(n_z, int) or not isinstance(n_x, int):
            raise TypeError("n_z and n_x must be integers")

        if n_z <= 0 or n_x <= 0:
            raise ValueError("n_z and n_x must be positive")

        if not n_x * n_z == self.scatterer_positions.shape[1]:
            raise ValueError("n_x*n_z must be equal to the number of scatterers")

        if not self._n_x is None and not self._n_z is None:
            raise ValueError("n_x and n_z have already been set")

        self._n_z = n_z
        self._n_x = n_x

    @staticmethod
    def _validate_input(
        scatterer_positions, scatterer_amplitudes, sound_speed, attenuation_coefficient
    ):
        """Validates the input.

        Parameters
        ----------
            scatterer_positions : np.ndarray
                The input scatterer positions.
            scatterer_amplitudes : np.ndarray
                The input scatterer amplitudes.
            sound_speed : float or int
                The input sound speed.
            attenuation_coefficient : float or int
                The input attenuation coefficient in dB/m/MHz.
        """

        # ==============================================================================
        # Check scatterer positions
        # ==============================================================================
        # Check if the input is a numpy array
        if not isinstance(scatterer_positions, np.ndarray):
            raise TypeError(
                "scatterer_positions must be a numpy array. "
                f"Got {type(scatterer_positions)}"
            )

        if scatterer_positions.ndim != 2:
            raise ValueError(
                "scatterer_positions must have shape (2, n_scatterers). "
                f"Got {scatterer_positions.shape}"
            )

        if not scatterer_positions.dtype in [np.float32, np.float64]:
            raise TypeError(
                "scatterer_positions must be float32 or float64. "
                f"Got {scatterer_positions.dtype}"
            )

        if not scatterer_positions.shape[1] == 2:
            raise ValueError(
                "scatterer_positions must have shape (2, n_scatterers). "
                f"Got {scatterer_positions.shape}"
            )

        # ==============================================================================
        # Check scatterer amplitudes
        # ==============================================================================
        # Check if the input is a numpy array
        if not isinstance(scatterer_amplitudes, np.ndarray):
            raise TypeError(
                "scatterer_amplitudes must be a numpy array. "
                f"Got {type(scatterer_amplitudes)}"
            )

        if scatterer_amplitudes.ndim != 1:
            raise ValueError(
                "scatterer_amplitudes must have shape (n_scatterers,). "
                f"Got {scatterer_amplitudes.shape}"
            )

        if not scatterer_amplitudes.dtype in [np.float32, np.float64]:
            raise TypeError(
                "scatterer_amplitudes must be float32 or float64. "
                f"Got {scatterer_amplitudes.dtype}"
            )

        # ==============================================================================
        # Check sound speed
        # ==============================================================================
        # Check if the input is a float or int
        if not isinstance(sound_speed, (float, int)):
            raise TypeError("sound_speed must be a float. " f"Got {type(sound_speed)}")

        # ==============================================================================
        # Further checks
        # ==============================================================================
        # Test if inputs are compatible
        if scatterer_positions.shape[0] != scatterer_amplitudes.shape[0]:
            raise ValueError(
                "scatterer_positions and scatterer_amplitudes must have "
                "compatible shapes. "
                f"Got {scatterer_positions.shape} and {scatterer_amplitudes.shape}"
            )

        # Test if the sound speed is positive
        if sound_speed <= 0:
            raise ValueError("sound_speed must be positive")

        # ==============================================================================
        # Check attenuation coefficient
        # ==============================================================================
        # Check if the input is a float or int
        if not isinstance(attenuation_coefficient, (float, int)):
            raise TypeError(
                "The attenuation coefficient must be a float. "
                f"Got {type(attenuation_coefficient)}"
            )

        # Check if the attenuation coefficient is positive
        if attenuation_coefficient < 0:
            raise ValueError("The attenuation coefficient must be positive")

        # ==============================================================================
        # Warnings
        # ==============================================================================
        # Warn if the sound speed is very low
        if sound_speed <= 1000:
            log.warning(f"Sound speed is very low: {sound_speed} m/s")
