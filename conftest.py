import numpy as np
import pytest
from jaxus.containers import Probe, Medium, Receive, Transmit, Pulse
from scipy.signal import hamming

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


@pytest.fixture
def get_test_containers():
    """Produces a `Probe`, `Medium`, `Receive`, and `Transmit` containers for testing
    purposes.

    Returns
    -------
    probe, medium, receive, transmit
        The containers.
    """
    n_el = 128
    probe = Probe(
        probe_geometry=np.stack(
            [np.linspace(-0.02, 0.02, n_el), np.zeros(n_el)], axis=1
        ),
        center_frequency=7.6e6,
        element_width=3e-4,
        bandwidth=(5e6, 11e6),
    )
    n_scat = 50
    medium = Medium(
        scatterer_positions=np.stack(
            [
                np.random.randn(n_scat) * 4e-3,
                np.abs(25e-3 + np.random.randn(n_scat) * 4e-3),
            ],
            axis=1,
        ),
        scatterer_amplitudes=np.ones(n_scat),
        sound_speed=1540,
    )
    receive = Receive(sampling_frequency=4 * 7.6e6, n_ax=1024 * 2, initial_time=0)
    waveform = Pulse(
        carrier_frequency=probe.center_frequency,
        pulse_width=300e-9,
        chirp_rate=0,
        phase=0,
    )

    transmit = Transmit(
        t0_delays=np.zeros(n_el), tx_apodization=hamming(n_el), waveform=waveform
    )

    return probe, medium, receive, transmit