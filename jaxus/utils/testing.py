import numpy as np
from scipy.signal.windows import hamming

from jaxus.containers import Medium, Probe, Pulse, Receive, Transmit


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
