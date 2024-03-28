import matplotlib.pyplot as plt
import numpy as np

from jaxus import simulate_to_usbmd
from jaxus.containers import Medium, Probe, Pulse, Receive, Transmit
from jaxus.plotting import plot_beamformed, plot_rf, plot_to_darkmode


def test_simulate_to_usbmd():
    probe = Probe(
        probe_geometry=np.stack(
            [np.linspace(-19e-3, 19e-3, 128), np.zeros(128)], axis=0
        ),
        center_frequency=5e6,
        element_width=0.5e-3,
        bandwidth=(0.5e6, 10e6),
    )
    transmit = Transmit(
        t0_delays=np.zeros(probe.n_el),
        tx_apodization=np.ones(probe.n_el),
        waveform=Pulse(
            carrier_frequency=probe.center_frequency,
            pulse_width=400e-9,
            chirp_rate=0.0,
            phase=0.0,
        ),
    )
    receive = Receive(
        sampling_frequency=4 * probe.center_frequency,
        n_ax=1024,
        initial_time=0.0,
    )
    medium = Medium(
        sound_speed=1540.0,
        attenuation_coefficient=0.5,
        scatterer_positions=np.array(
            [
                [
                    0.0,
                ],
                [30e-3],
            ]
        ),
        scatterer_amplitudes=np.array([1.0]),
    )
    result = simulate_to_usbmd(
        path="output.h5",
        probe=probe,
        transmit=transmit,
        receive=receive,
        medium=medium,
    )
    fig, ax = plt.subplots()
    plot_beamformed(ax, result[0], extent_m=[-19e-3, 19e-3, 0, 30e-3])
    plot_to_darkmode(fig, ax)
    plt.show()