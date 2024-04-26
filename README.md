# jaxus
Ultrasound beamforming and processing in JAX.

## Installation
```bash
cd jaxus
pip install .
```

The project also depends on JAX. To install JAX follow the [official instructions](https://jax.readthedocs.io/en/latest/installation.html). At the time of writing, the following command should work for CUDA 12:
```bash
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```