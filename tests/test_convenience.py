from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jaxus import compute_t0_delays_from_origin_distance_angle, simulate_to_hdf5
from jaxus.containers import Medium, Probe, Pulse, Receive, Transmit
from jaxus.plotting import plot_beamformed, plot_rf, plot_to_darkmode, use_dark_style


def test_simulate_to_hdf5():

    n_el = 100
    probe_geometry = np.stack(
        [np.linspace(-12.5e-3, 12.5e-3, n_el), np.zeros(n_el)], axis=1
    )
    probe = Probe(
        probe_geometry=probe_geometry,
        center_frequency=5e6,
        element_width=probe_geometry[1, 0] - probe_geometry[0, 0],
        bandwidth=(2e6, 9e6),
    )
    focus_angle = 0 * np.pi / 180
    focus_distance = 25e-3
    transmit = Transmit(
        t0_delays=np.array(
            compute_t0_delays_from_origin_distance_angle(
                probe_geometry,
                origin=np.array([0, 0]),
                distance=focus_distance,
                angle=focus_angle,
                sound_speed=1540,
            )
        ),
        focus_angle=focus_angle,
        focus_distance=focus_distance,
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
        n_ax=1024 + 512,
        initial_time=0.0,
    )
    n_scat = 9
    xmax = 10e-3
    ymax = 50e-3
    scat_x = np.concatenate([np.linspace(-5e-3, 5e-3, 5), np.zeros(5)]) + 10e-3
    scat_y = np.concatenate([np.ones(5) * 15e-3, np.linspace(15e-3, 30e-3, 5)])

    scat_x, scat_y = np.meshgrid(
        np.linspace(-xmax, xmax, n_scat), np.linspace(5e-3, ymax, n_scat)
    )
    scat_x, scat_y = scat_x.flatten(), scat_y.flatten()

    positions = np.stack([scat_x, scat_y], axis=1)

    medium = Medium(
        scatterer_positions=positions,
        scatterer_amplitudes=np.ones(scat_x.shape[0]),
        sound_speed=1540,
    )
    output_path = Path("output.hdf5")
    # Remove the file if it exists
    if output_path.exists():
        output_path.unlink()

    images, extent = simulate_to_hdf5(
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
        images[0],
        extent_m=extent,
        probe_geometry=probe_geometry,
    )
    ax.plot(scat_x, scat_y, "x", color="red", markersize=0.8)
    colorcycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # ax.plot(vsource[0], vsource[1], ".", color=colorcycle[2])
    plt.savefig("output.png")
    plt.show()
