from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jaxus import simulate_to_hdf5
from jaxus.containers import Medium, Probe, Pulse, Receive, Transmit
from jaxus.plotting import plot_beamformed, plot_rf, plot_to_darkmode, use_dark_style


def test_simulate_to_hdf5():

    n_el = 64
    probe_geometry = np.stack(
        [np.linspace(-9.5e-3, 9.5e-3, n_el), np.zeros(n_el)], axis=1
    )
    probe = Probe(
        probe_geometry=probe_geometry,
        center_frequency=5e6,
        element_width=probe_geometry[1, 0] - probe_geometry[0, 0],
        bandwidth=(2e6, 9e6),
    )
    transmit = Transmit(
        t0_delays=np.zeros(probe.n_el),
        tx_apodization=np.ones(probe.n_el),
        waveform=Pulse(
            carrier_frequency=probe.center_frequency,
            pulse_width=700e-9,
            chirp_rate=0.0,
            phase=0.0,
        ),
    )
    receive = Receive(
        sampling_frequency=4 * probe.center_frequency,
        n_ax=1024,
        initial_time=0.0,
    )
    n_scat = 10
    scat_x = np.concatenate([np.linspace(-5e-3, 5e-3, 5), np.zeros(5)])
    scat_y = np.concatenate([np.ones(5) * 15e-3, np.linspace(15e-3, 30e-3, 5)])

    positions = np.stack([scat_x, scat_y], axis=1)

    medium = Medium(
        scatterer_positions=positions,
        scatterer_amplitudes=np.ones(scat_x.shape[0]),
        sound_speed=1540,
    )
    output_path = Path(
        r"C:\Users\vince\Documents\3_resources\data\verasonics\hdf5\simu\output.h5"
    )
    # Remove the file if it exists
    if output_path.exists():
        output_path.unlink()

    result = simulate_to_hdf5(
        path=output_path,
        probe=probe,
        transmit=transmit,
        receive=receive,
        medium=medium,
    )
    use_dark_style()
    fig, ax = plt.subplots()
    depth = receive.n_ax / 2 / receive.sampling_frequency * medium.sound_speed
    plot_beamformed(
        ax,
        result[0],
        extent_m=[-19e-3, 19e-3, depth, -1e-3],
        probe_geometry=probe_geometry,
    )
    plt.show()
