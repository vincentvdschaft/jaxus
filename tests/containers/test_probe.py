import numpy as np
import pytest

from jaxus.containers.probe import Probe


def test_correct_initialization():
    """Tests that the probe is correctly initialized without errors."""

    n_el = 128

    # Define the probe geometry
    x_pos = np.linspace(-0.02, 0.02, n_el)
    z_pos = np.zeros_like(x_pos)
    probe_geometry = np.stack((x_pos, z_pos), axis=1)

    center_frequency = 4e6

    element_width = 1e-3

    bandwidth = (2e6, 6e6)

    probe = Probe(probe_geometry, center_frequency, element_width, bandwidth)

    # Check that all values ended up in the correct place
    assert probe.n_el == n_el
    assert probe.center_frequency == center_frequency
    assert np.array_equal(probe.probe_geometry, probe_geometry.astype(np.float32))
    assert probe.element_width == element_width


@pytest.mark.parametrize(
    "probe_geometry, center_frequency",
    [
        (np.ones((2, 6), dtype=np.int32), 1e6),
        (np.ones((2, 128)), "string"),
        (None, 1e6),
    ],
)
def test_wrong_datatypes(probe_geometry, center_frequency):
    """Tests that the probe is correctly initialized without errors."""

    with pytest.raises(TypeError):
        Probe(probe_geometry, center_frequency)


def test_wrong_shape():
    """Tests if the class raises an error if the probe geometry has the wrong shape."""

    # Define the probe geometry
    probe_geometry = np.ones((128, 3), dtype=np.float32)
    center_frequency = 4e6
    element_width = 1e-3
    bandwidth = (2e6, 6e6)

    with pytest.raises(ValueError):
        Probe(probe_geometry, center_frequency, element_width, bandwidth)


def test_negative_frequency():
    """Tests if the class raises an error if the center frequency is negative."""

    # Define the probe geometry
    probe_geometry = np.ones((128, 2), dtype=np.float32)
    center_frequency = -1e6
    element_width = 1e-3
    bandwidth = (2e6, 6e6)

    with pytest.raises(ValueError):
        Probe(probe_geometry, center_frequency, element_width, bandwidth)


def test_negative_element_width():
    """Tests if the class raises an error if the element width is negative."""

    # Define the probe geometry
    probe_geometry = np.ones((128, 2), dtype=np.float32)
    center_frequency = 4e6
    element_width = -1e-3
    bandwidth = (2e6, 6e6)

    with pytest.raises(ValueError):
        Probe(probe_geometry, center_frequency, element_width, bandwidth)
