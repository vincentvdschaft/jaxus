import numpy as np
import pytest


def _get_probe_geometry_linear(aperture, n_el):
    """Helper function to generate a linear probe geometry."""
    x_positions = np.linspace(-aperture / 2, aperture / 2, n_el)
    z_positions = np.zeros_like(x_positions)
    return np.stack((x_positions, z_positions), axis=-1)


@pytest.fixture
def fixture_probe_geometry_s51():
    """Fixture to return the probe geometry of the Philips S51 probe."""
    return _get_probe_geometry_linear(20e-3, 80)


@pytest.fixture
def fixture_probe_geometry_l11_4v():
    """Fixture to return the probe geometry of the Verasonics L11-4v probe."""
    return _get_probe_geometry_linear(40e-3, 128)
