# JAX UltraSound
Basic ultrasound simulation and reconstruction functionality implemented in JAX.

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
### Simulating ultrasound data
```python
from jaxus import (
    Probe,
    Transmit,
    Receive,
    Pulse,
    Medium,
    simulate_to_hdf5,
    plot_beamformed,
)

# Set the number of elements in the probe
n_el = 64

# Define the positions of the elements in the probe
probe_geometry = np.stack([np.linspace(-9.5e-3, 9.5e-3, n_el), np.zeros(n_el)], axis=1)

# Create a probe object
probe = Probe(
    probe_geometry=probe_geometry,
    center_frequency=5e6,
    element_width=probe_geometry[1, 0] - probe_geometry[0, 0],
    bandwidth=(2e6, 9e6),
)

# Create a transmit object
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

# Create a receive object
receive = Receive(
    sampling_frequency=4 * probe.center_frequency,
    n_ax=1024,
    initial_time=0.0,
)

# Define the scatterer positions
scat_x = np.concatenate([np.linspace(-5e-3, 5e-3, 5), np.zeros(5)])
scat_y = np.concatenate([np.ones(5) * 15e-3, np.linspace(15e-3, 30e-3, 5)])

positions = np.stack([scat_x, scat_y], axis=1)

# Create a medium object containing the scatterer positions
medium = Medium(
    scatterer_positions=positions,
    scatterer_amplitudes=np.ones(scat_x.shape[0]),
    sound_speed=1540,
)
output_path = Path(r"./output.h5")

# Simulate the ultrasound data
result = simulate_to_hdf5(
    path=output_path,
    probe=probe,
    transmit=transmit,
    receive=receive,
    medium=medium,
)

# Plot the beamformed image
use_dark_style()
fig, ax = plt.subplots()
depth = receive.n_ax / 2 / receive.sampling_frequency * medium.sound_speed
plot_beamformed(
    ax, result[0], extent_m=[-19e-3, 19e-3, depth, 0], probe_geometry=probe_geometry
)
plt.show()
```