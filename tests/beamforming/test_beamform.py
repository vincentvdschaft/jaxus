import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import hamming

from jaxus.beamforming.beamform import CartesianPixelGrid, beamform, log_compress
from jaxus.containers.medium import Medium
from jaxus.containers.probe import Probe
from jaxus.containers.receive import Receive
from jaxus.containers.transmit import Transmit
from jaxus.containers.waveform import Pulse
from jaxus.data.usbmd_data_format import generate_usbmd_dataset
from jaxus.plotting import plot_beamformed, plot_rf, plot_to_darkmode
from jaxus.rf_simulator import simulate_rf_data


def test_beamform():
    n_el = 128 * 2
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
                np.abs(35e-3 + np.random.randn(n_scat) * 4e-3),
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

    wavelength = medium.sound_speed / probe.center_frequency

    t0_delays = np.linspace(-1e-6, 1e-6, n_el)

    rf = []

    factors = (0, 1)
    n_tx = len(factors)

    t0_stack = []
    for factor in factors:
        t0 = t0_delays * factor
        t0 -= np.min(t0)
        t0_stack.append(t0)

    t0_delays = jnp.stack(t0_stack, axis=0)

    for n in range(t0_delays.shape[0]):

        rf_data = simulate_rf_data(
            n_ax=receive.n_ax,
            scatterer_positions=medium.scatterer_positions,
            scatterer_amplitudes=medium.scatterer_amplitudes,
            t0_delays=t0_delays[n],
            probe_geometry=probe.probe_geometry,
            element_angles=np.zeros(n_el),
            tx_apodization=transmit.tx_apodization,
            initial_time=receive.initial_time,
            element_width_wl=probe.element_width / wavelength,
            sampling_frequency=receive.sampling_frequency,
            carrier_frequency=transmit.carrier_frequency,
            sound_speed=medium.sound_speed,
            attenuation_coefficient=0.7,
            tx_angle_sensitivity=True,
            rx_angle_sensitivity=True,
            wavefront_only=False,
        )
        rf.append(rf_data)
    rf_data = np.stack(rf, axis=0)

    n_z, n_x = 512, 512
    pixel_grid = CartesianPixelGrid(
        n_x=n_x, n_z=n_z, dx_wl=0.5, dz_wl=0.5, z0=0, wavelength=wavelength
    )

    bf_data = beamform(
        rf_data[None, :, :, :, None],
        pixel_grid.pixel_positions_flat,
        probe_geometry=probe.probe_geometry,
        t0_delays=np.array(transmit.t0_delays),
        initial_times=np.ones(n_tx) * receive.initial_time,
        sampling_frequency=receive.sampling_frequency,
        carrier_frequency=waveform.carrier_frequency,
        sound_speed=medium.sound_speed,
        t_peak=waveform.t_peak * jnp.ones(1),
        rx_apodization=hamming(n_el),
        f_number=1.5,
        iq_beamform=True,
        # pixel_chunk_size=2**15,
    )

    # bf_data = detect_envelope_beamformed(bf_data.reshape((512, 512))[None], dz_wl=1.0)[0]

    bf_data = log_compress(bf_data.reshape((n_z, n_x)), normalize=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_beamformed(
        axes[0],
        bf_data,
        pixel_grid.extent,
        probe_geometry=probe.probe_geometry,
    )
    plot_rf(axes[1], rf_data[0], cmap="cividis", axis_in_mm=True)
    plot_to_darkmode(fig, axes)
    plt.show()
