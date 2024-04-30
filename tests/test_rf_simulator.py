import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxus import plot_rf, use_dark_style, simulate_rf_transmit

# Set to True to plot the result
PLOT = True


def test_rf_simulator():
    """Runs the rf simulator and plots the result."""
    # The number of axial samples to simulate
    n_ax = 1024
    # The number of elements in the probe
    n_el = 128
    # The center frequency in Hz
    carrier_frequency = 7e6
    # Sampling frequency in Hz
    sampling_frequency = 4 * carrier_frequency
    # The speed of sound in m/s
    c = 1540
    # The width of the elements in wavelengths of the center frequency
    width_wl = 1.33
    # The time instant of the first sample in seconds
    initial_time = 0.0
    # The number of scatterers to process simultaneously. If it does not fit in memory
    # then lower this number.
    scatterer_chunk_size = 512
    # The number of axial samples to process simultaneously. If it does not fit in
    # memory then lower this number.
    ax_chunk_size = 1024
    # Set to True to simulate a single point scatterer. If False a region filled
    # with randomly placed scatterers is simulated.
    single_point_scatterer = True
    # The attenuation coefficient in dB/(MHz*cm)
    attenuation_coefficient = 0.9
    # Set to True to simulate the wavefront only. Instead of summing the wavefronts
    # from each transmit element, the wavefront from the transmit element with the
    # shortest time of flight is used. The reduces computation time, but is less
    # accurate.
    wavefront_only = False

    # Generate probe geometry
    probe_geometry = jnp.stack(
        [jnp.linspace(-19e-3, 19e-3, n_el), jnp.zeros(n_el)], axis=1
    )

    element_angles = 0 * jnp.ones(n_el) * jnp.pi / 2

    # Set all t0_delays to 0 to simulate a plane wave with angle 0
    t0_delays = jnp.zeros(n_el)

    # Set the tx apodization to 1
    tx_apodization = jnp.ones(n_el)

    # Alternative: Set hanning apodization
    # tx_apodization = jnp.hanning(n_el)

    if single_point_scatterer:
        scatterer_positions = jnp.array(
            [
                [0e-3, 0, 0],
                [10e-3, 20e-3, 30e-3],
            ],
        ).T
    else:
        n_scat = 10000
        x_pos = np.random.uniform(-1e-3, 5e-3, n_scat)
        z_pos = np.random.uniform(2e-3, 8e-3, n_scat)
        scatterer_positions = np.stack([x_pos, z_pos], axis=1)

    scatterer_amplitudes = np.ones((scatterer_positions.shape[0]))

    scatterer_positions = jnp.array(scatterer_positions)
    scatterer_amplitudes = jnp.array(scatterer_amplitudes)

    print("scatterer_positions.shape: ", scatterer_positions.shape)
    print("scatterer_amplitudes.shape: ", scatterer_amplitudes.shape)

    rf_data = simulate_rf_transmit(
        n_ax,
        scatterer_positions,
        scatterer_amplitudes,
        t0_delays,
        probe_geometry,
        element_angles,
        tx_apodization,
        initial_time,
        width_wl,
        sampling_frequency,
        carrier_frequency,
        c,
        attenuation_coefficient,
        wavefront_only,
        ax_chunk_size=ax_chunk_size,
        scatterer_chunk_size=scatterer_chunk_size,
        progress_bar=True,
    )

    assert not jnp.isnan(rf_data).any(), "rf_data contains NaNs"
    assert not jnp.isinf(rf_data).any(), "rf_data contains infs"

    if not PLOT:
        return

    use_dark_style()
    fig, ax = plt.subplots(1, 1)
    plot_rf(ax, rf_data, cmap="cividis")
    plt.show()
