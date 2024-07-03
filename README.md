# JAX UltraSound
Basic ultrasound simulation and reconstruction functionality implemented in JAX.

> **⚠️ Note**: The project is in an early stage of development and the API is subject to change. Be sure to pin the version of the package in your `requirements.txt` file.

## Installation
Install the package using pip:
```bash
pip install jaxus
```

The project also depends on JAX. To install JAX follow the [official instructions](https://jax.readthedocs.io/en/latest/installation.html). At the time of writing, the following command should work for CUDA 12:
```bash
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


## Usage
### Define simulation parameters
```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxus import (
    plot_rf,
    use_dark_style,
    simulate_rf_transmit,
    beamform_das,
    log_compress,
    plot_beamformed,
    CartesianPixelGrid,
)
# The number of axial samples to simulate
n_ax = 2048
# The number of elements in the probe
n_el = 128
# The center frequency in Hz
carrier_frequency = 7e6
# Sampling frequency in Hz
sampling_frequency = 4 * carrier_frequency
# The speed of sound in m/s
sound_speed = 1540
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
probe_geometry = np.stack([jnp.linspace(-19e-3, 19e-3, n_el), jnp.zeros(n_el)], axis=1)

# Define the element angles in radians
element_angles = 0 * np.ones(n_el) * jnp.pi / 2

# Set all t0_delays to 0 to simulate a plane wave with angle 0
t0_delays = np.zeros(n_el)

# Set the tx apodization to 1
tx_apodization = np.ones(n_el)

# Define the scatterer positions and amplitudes
n_scat = 30

t = np.linspace(0, 2 * np.pi, n_scat, endpoint=False)
scatterer_x = 16e-3 * np.sin(t) ** 3
scatterer_z = -(
    13e-3 * np.cos(t)
    - 5e-3 * np.cos(2 * t)
    - 2e-3 * np.cos(3 * t)
    - 1e-3 * np.cos(4 * t)
)
scatterer_z -= np.min(scatterer_z)
scatterer_z += 15e-3

scatterer_positions = np.stack([scatterer_x, scatterer_z], axis=1)

scatterer_amplitudes = np.ones((scatterer_positions.shape[0]))


# Simulate RF data
# -----------------

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
    sound_speed,
    attenuation_coefficient,
    wavefront_only,
    ax_chunk_size=ax_chunk_size,
    scatterer_chunk_size=scatterer_chunk_size,
    progress_bar=True,
)


# Beamform the RF data using delay-and-sum
# ----------------------------------------

# Define beamforming grid
pixel_grid = CartesianPixelGrid(
    n_x=512,
    n_z=512,
    dx_wl=0.5,
    dz_wl=0.5,
    z0=1e-4,
    wavelength=sound_speed / carrier_frequency,
)

# Perform DAS beamforming
beamformed_image = beamform_das(
    rf_data[None, None, :, :, None],
    pixel_positions=pixel_grid.pixel_positions_flat,
    probe_geometry=probe_geometry,
    t0_delays=t0_delays[None],
    initial_times=np.ones(1) * initial_time,
    sampling_frequency=sampling_frequency,
    carrier_frequency=carrier_frequency,
    sound_speed=sound_speed,
    sound_speed_lens=sound_speed,
    lens_thickness=0.0,
    f_number=1.5,
    rx_apodization=np.ones(n_el),
    tx_apodizations=tx_apodization[None],
    iq_beamform=True,
    t_peak=np.array([0.0]),
)

beamformed_image = log_compress(
    beamformed_image.reshape(pixel_grid.shape), normalize=True
)


# Plotting
# --------

# Plot the RF data and the beamformed image
use_dark_style()
fig, axes = plt.subplots(1, 2)
ax_rf, ax_bf = axes
plot_rf(ax_rf, rf_data)
plot_beamformed(
    ax_bf,
    beamformed_image,
    extent_m=pixel_grid.extent,
    probe_geometry=probe_geometry,
)
plt.tight_layout()
plt.show()
```