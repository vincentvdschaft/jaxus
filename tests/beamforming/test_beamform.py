import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal.windows import hamming

from jaxus.beamforming import (
    CartesianPixelGrid,
    beamform_das,
    detect_envelope_beamformed,
    log_compress,
)
from jaxus.containers.medium import Medium
from jaxus.containers.probe import Probe
from jaxus.containers.receive import Receive
from jaxus.containers.transmit import Transmit
from jaxus.containers.waveform import Pulse
from jaxus.data.hdf5_data_format import generate_hdf5_dataset
from jaxus.plotting import plot_beamformed, plot_rf, plot_to_darkmode
from jaxus.rf_simulator import simulate_rf_transmit
from jaxus.utils.testing import get_test_containers


@pytest.mark.parametrize("iq_beamform", [True, False])
def test_beamform(iq_beamform):
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
    dz_wl = 0.25
    pixel_grid = CartesianPixelGrid(
        n_x=n_x, n_z=n_z, dx_wl=0.25, dz_wl=dz_wl, z0=1e-3, wavelength=wavelength
    )

    # ==================================================================================
    # Beamform the RF data
    # ==================================================================================
    bf_data = beamform_das(
        rf_data[None, :, :, :, None],
        pixel_grid.pixel_positions_flat,
        probe_geometry=probe.probe_geometry,
        t0_delays=np.array(transmit.t0_delays)[None],
        initial_times=np.ones(1) * receive.initial_time,
        sampling_frequency=receive.sampling_frequency,
        carrier_frequency=transmit.waveform.carrier_frequency,
        sound_speed=medium.sound_speed,
        sound_speed_lens=1540,
        lens_thickness=1e-3,
        t_peak=transmit.waveform.t_peak * jnp.ones(1),
        rx_apodization=hamming(n_el),
        tx_apodizations=transmit.tx_apodization[None],
        f_number=1.5,
        iq_beamform=iq_beamform,
        # pixel_chunk_size=2**15,
    )

    if not iq_beamform:
        # Perform envelope detection
        bf_data = detect_envelope_beamformed(
            bf_data.reshape((n_z, n_x))[None], dz_wl=pixel_grid.dz / wavelength
        )[0]
        # bf_data = bf_data.reshape((n_z, n_x))

    bf_data = log_compress(bf_data.reshape((n_z, n_x)), normalize=True)

    # ==================================================================================
    # Plot the beamformed data
    # ==================================================================================

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_beamformed(
        axes[0],
        bf_data,
        pixel_grid.extent,
        probe_geometry=probe.probe_geometry,
        # vmin=np.min(bf_data),
        # vmax=np.max(bf_data),
    )
    plot_rf(axes[1], rf_data[0], cmap="cividis")
    plot_to_darkmode(fig, axes)
    axes[0].set_title("RF beamformed data" if not iq_beamform else "IQ beamformed data")
    plt.show()
