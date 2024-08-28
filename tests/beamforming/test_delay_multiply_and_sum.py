import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal.windows import hamming

from jaxus.beamforming import beamform_dmas, log_compress, get_pixel_grid
from jaxus.plotting import plot_beamformed, plot_rf, plot_to_darkmode
from jaxus.rf_simulator import simulate_rf_transmit
from jaxus.utils.testing import get_test_containers


def test_beamform():
    probe, medium, receive, transmit = get_test_containers()
    n_el = probe.n_el

    wavelength = medium.sound_speed / probe.center_frequency

    # ==================================================================================
    # Generate the RF data
    # ==================================================================================
    rf_data = simulate_rf_transmit(
        n_ax=receive.n_ax,
        scatterer_positions=medium.scatterer_positions,
        scatterer_amplitudes=medium.scatterer_amplitudes,
        t0_delays=transmit.t0_delays,
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
    )[None]

    # ==================================================================================
    # Define beamforming grid
    # ==================================================================================
    n_z, n_x = 512 + 256, 512
    dx_wl, dz_wl = 0.25, 0.25
    spacing = np.array([dx_wl, dz_wl]) * wavelength
    pixel_grid = get_pixel_grid(
        shape=(n_x, n_z),
        spacing=spacing,
        startpoints=(0, 1e-3),
        center=(True, False),
    )

    # ==================================================================================
    # Beamform the RF data
    # ==================================================================================
    bf_data = beamform_dmas(
        rf_data[None, :, :, :, None],
        pixel_grid.pixel_positions_flat,
        probe_geometry=probe.probe_geometry,
        t0_delays=np.array(transmit.t0_delays)[None],
        initial_times=np.ones(1) * receive.initial_time,
        sampling_frequency=receive.sampling_frequency,
        carrier_frequency=transmit.waveform.carrier_frequency,
        sound_speed=medium.sound_speed,
        sound_speed_lens=medium.sound_speed,
        lens_thickness=1e-3,
        t_peak=transmit.waveform.t_peak * jnp.ones(1),
        tx_apodizations=transmit.tx_apodization[None],
        rx_apodization=hamming(n_el),
        f_number=1.5,
        pixel_chunk_size=2**14,
    )

    bf_data = log_compress(bf_data.reshape(pixel_grid.shape_2d).T, normalize=True)

    # ==================================================================================
    # Plot the beamformed data
    # ==================================================================================

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_beamformed(
        axes[0],
        bf_data,
        pixel_grid.extent_m_2d_zflipped,
        probe_geometry=probe.probe_geometry,
    )
    plot_rf(axes[1], rf_data[0], cmap="cividis")
    plot_to_darkmode(fig, axes)
    axes[0].set_title("DMAS IQ beamformed data")
    plt.show()
