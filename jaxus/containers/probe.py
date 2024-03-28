from typing import Union

import numpy as np

import src.utils.log as log
from src.utils.utils import hash_list, hash_ndarray


class Probe:
    """Containter class storing the parameters pertaining to the probe.

    ## Contains:
    - probe_geometry (`np.ndarray`): The probe geometry in meters. [2, n_el]
    - center_frequency (`float`): The center frequency of the probe in Hz.
    - element_width (`float`): The width of each element in meters.
    """

    def __init__(
        self,
        probe_geometry: np.ndarray,
        center_frequency: Union[float, int],
        element_width: Union[float, int],
        bandwidth: Union[float, int],
    ):
        """Initializes the Probe object.

        ### Args:
            `probe_geometry` (`np.ndarray`): The probe geometry in meters. [2, n_el]
            `center_frequency` (`float`, `int`): The center frequency of the probe in Hz.
            `element_width` (`float`, `int`): The width of each element in meters.
            `bandwidth` (`float`, `int`): The bandwidth of the probe in Hz.
        """
        self._validate_input(probe_geometry, center_frequency, element_width, bandwidth)

        self._probe_geometry = probe_geometry.astype(np.float32)
        self._center_frequency = float(center_frequency)
        self._element_width = float(element_width)
        self._bandwidth = float(bandwidth)

    @property
    def n_el(self):
        """The number of elements in the probe."""
        return self._probe_geometry.shape[1]

    @property
    def center_frequency(self):
        """The center frequency of the probe in Hz."""
        return self._center_frequency

    @property
    def probe_geometry(self):
        """The probe geometry in meters. [2, n_el]"""
        return self._probe_geometry

    @property
    def element_width(self):
        """The width of each element in meters."""
        return self._element_width

    def __repr__(self):
        return (
            f"Probe({self.n_el} elements, "
            f"from {self._probe_geometry[0, 0]*1e3:.1f}mm to {self._probe_geometry[0, -1]*1e3:.1f}mm, "
            f"center_frequency={self.center_frequency*1e-6:.1f}MHz, "
            f"element_width={self.element_width*1e3:.1f}mm)"
        )

    @staticmethod
    def _validate_input(
        probe_geometry: np.ndarray,
        center_frequency: Union[float, int],
        element_width: Union[float, int],
        bandwidth: Union[float, int],
    ):
        """Checks if the input is valid.

        ### Args:
            `probe_geometry` (`np.ndarray`): The input probe geometry of shape
                `(2, n_el)`.
            `center_frequency` (`float`, `int`): The input center frequency.
            `element_width` (`float`, `int`): The input element width.
            `bandwidth` (`float`, `int`): The input bandwidth.

        Raises:
            TypeError: If the probe geometry is not a numpy array.
            ValueError: If the probe geometry does not have shape `(2, n_el)`.
            TypeError: If the probe geometry is not `float32` or `float64`.
            TypeError: If the center frequency is not a float.
        """

        # ==============================================================================
        # Check probe geometry
        # ==============================================================================
        if not isinstance(probe_geometry, np.ndarray):
            raise TypeError(
                "probe_geometry must be a numpy array. " f"Got {type(probe_geometry)}"
            )

        if probe_geometry.shape[0] != 2:
            raise ValueError(
                "probe_geometry must have shape (2, n_el). "
                f"Got {probe_geometry.shape}"
            )

        if not probe_geometry.dtype in [np.float32, np.float64]:
            raise TypeError(
                "probe_geometry must be float32 or float64. "
                f"Got {probe_geometry.dtype}"
            )

        # ==============================================================================
        # Check center frequency
        # ==============================================================================
        if not isinstance(center_frequency, (float, int)):
            raise TypeError("center_frequency must be a float or int")

        if center_frequency < 0:
            raise ValueError("center_frequency must be positive")

        # ==============================================================================
        # Check element width
        # ==============================================================================
        if not isinstance(element_width, (float, int)):
            raise TypeError("element_width must be a float or int")

        if element_width < 0:
            raise ValueError("element_width must be positive")

        # ==============================================================================
        # Check bandwidth
        # ==============================================================================
        if not isinstance(bandwidth, (float, int)):
            raise TypeError("bandwidth must be a float or int")

        if bandwidth < 0:
            raise ValueError("bandwidth must be positive")

        if bandwidth > center_frequency:
            raise ValueError("bandwidth must be less than center_frequency")

        # ==============================================================================
        # Warnings
        # ==============================================================================
        # Warn if the probe geometry is very large
        if np.any(probe_geometry > 0.1):
            log.warning(f"Probe geometry is very large: {probe_geometry.max()} m")

        # Warn if the center frequency is very low
        if center_frequency <= 1000:
            log.warning(f"Center frequency is very low: {center_frequency} Hz")

        # Warn if the element width is very large
        if element_width > 5e-3:
            log.warning(f"Element width is very large: {element_width*1e3} mm")

    def __hash__(self) -> int:
        """Hashes the Probe object."""
        list_of_hashes = [
            hash_ndarray(self._probe_geometry),
            self._center_frequency,
            self._element_width,
        ]
        return int(hash_list(list_of_hashes), base=16)
