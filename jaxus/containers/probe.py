from typing import Union

import numpy as np

import jaxus.utils.log as log
from jaxus.utils.checks import check_pos_array


class Probe:
    """Containter class storing the parameters pertaining to the probe.

    Contains:
    ----------
    - probe_geometry (`np.ndarray`): The probe geometry in meters. (2, n_el)
    - center_frequency (`float`): The center frequency of the probe in Hz.
    - element_width (`float`): The width of each element in meters.
    - bandwidth (`float`): The bandwidth of the probe in Hz.
    """

    def __init__(
        self,
        probe_geometry: np.ndarray,
        center_frequency: Union[float, int],
        element_width: Union[float, int],
        bandwidth: tuple[Union[float, int], Union[float, int]],
        lens_correction: float = 0.0,
    ):
        """Initializes the Probe object.

        Parameters
        ----------
        probe_geometry : np.ndarray
            The probe geometry in meters of shape (2, n_el).
        center_frequency : float or int
            The center frequency of the probe in Hz.
        element_width : float or int
            The width of each element in meters.
        bandwidth : tuple of float or int
            The start and end of the bandwidth in Hz.
        lens_correction : float, optional
            The extra path-length introduced by the lens in m, by default 0.0
        """
        self._validate_input(
            probe_geometry, center_frequency, element_width, bandwidth, lens_correction
        )

        self._probe_geometry = probe_geometry.astype(np.float32)
        self._center_frequency = float(center_frequency)
        self._element_width = float(element_width)
        self._bandwidth = (float(bandwidth[0]), float(bandwidth[1]))
        self._lens_correction = float(lens_correction)

    @property
    def n_el(self):
        """The number of elements in the probe."""
        return self._probe_geometry.shape[0]

    @property
    def center_frequency(self):
        """The center frequency of the probe in Hz."""
        return self._center_frequency

    @property
    def probe_geometry(self):
        """The probe geometry in meters. (n_el, 2)"""
        return self._probe_geometry

    @property
    def element_width(self):
        """The width of each element in meters."""
        return self._element_width

    @property
    def bandwidth(self):
        """The bandwidth of the probe in Hz."""
        return self._bandwidth

    @property
    def aperture(self):
        """The aperture of the probe in meters, computed as the distance between the
        first and last element."""
        return np.linalg.norm(self._probe_geometry[:, -1] - self._probe_geometry[:, 0])

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
        lens_correction: Union[float, int],
        verbose: bool = False,
    ):
        """Validates the input parameters.

        Parameters
        ----------
        probe_geometry : np.ndarray
            The probe geometry in meters of shape (2, n_el).
        center_frequency : Union[float, int]
            The center frequency of the probe in Hz.
        element_width : Union[float, int]
            The width of each element in meters.
        bandwidth : Union[float, int]
            The start and end of the bandwidth in Hz.
        lens_correction : Union[float, int]
            The extra path-length introduced by the lens in m.
        verbose : bool, optional
            Set to True to receive warning about uncommon values, by default False
        """

        # ==============================================================================
        # Check probe geometry
        # ==============================================================================
        check_pos_array(probe_geometry, "probe_geometry")

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
        try:
            bandwidth = tuple(bandwidth)
            if not all(isinstance(b, (float, int)) for b in bandwidth):
                raise TypeError("bandwidth must be tuple of floats or ints")
        except TypeError as e:
            raise TypeError("bandwidth must be tuple of floats or ints") from e

        if len(bandwidth) != 2 or bandwidth[0] > bandwidth[1] or bandwidth[0] < 0:
            raise ValueError(
                "bandwidth must be a tuple of 2 increasing positive floats"
            )

        if bandwidth[1] - bandwidth[0] > center_frequency * 2:
            raise ValueError("bandwidth must be less than twice center_frequency")

        # ==============================================================================
        # Check lens correction
        # ==============================================================================
        if not isinstance(lens_correction, (float, int)):
            raise TypeError("lens_correction must be a float or int")

        # ==============================================================================
        # Warnings
        # ==============================================================================
        if not verbose:
            return

        # Warn if the probe geometry is very large
        if np.any(probe_geometry > 0.1):
            log.warning(f"Probe geometry is very large: {probe_geometry.max()} m")

        # Warn if the center frequency is very low
        if center_frequency <= 1000:
            log.warning(f"Center frequency is very low: {center_frequency} Hz")

        # Warn if the element width is very large
        if element_width > 5e-3:
            log.warning(f"Element width is very large: {element_width*1e3} mm")
