import jax.numpy as jnp
import numpy as np


def log_compress(beamformed: jnp.ndarray, normalize: bool = False):
    """Log-compresses the beamformed image.

    Parameters
    ----------
    beamformed : jnp.ndarray
        The beamformed image to log-compress of any shape.
    normalize : bool, default=False
        Whether to normalize the beamformed image.

    Returns
    -------
    beamformed : jnp.ndarray
        The log-compressed image of the same shape as the input in dB.
    """
    if not isinstance(beamformed, (jnp.ndarray, np.ndarray)):
        raise TypeError(f"beamformed must be a ndarray. Got {type(beamformed)}.")
    if not isinstance(normalize, bool):
        raise TypeError(f"normalize must be a bool. Got {type(normalize)}.")

    beamformed = jnp.abs(beamformed)
    if normalize:
        beamformed = beamformed / jnp.clip(jnp.max(beamformed), 1e-12)
    beamformed = 20 * jnp.log10(beamformed + 1e-12)

    return beamformed
