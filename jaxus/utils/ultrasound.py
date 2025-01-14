import jax.numpy as jnp
import numpy as np
import jax
import optax


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


def deduce_vsource(probe_geometry, t0_delays, sound_speed):
    """Estimates the virtual source.

    Parameters
    ----------
    probe_geometry : jnp.array
        The probe geometry of shape (n_el, 2)
    t0_delays : jnp.array
        The t0_delays of shape (n_el,)
    sound_speed : float
        The sound speed in m/s.

    Returns
    -------
    vsource : jnp.array
        The virtual source of shape (2,)
    """

    derivative = jnp.diff(t0_delays)

    derivative = (derivative - jnp.min(derivative)) / (
        jnp.max(derivative) - jnp.min(derivative)
    )
    second_derivative = jnp.mean(jnp.diff(derivative))

    sign = -1.0 if second_derivative < 0.0 else 1.0

    def loss_fn(vsource):
        t0_delays_hat = (
            sign
            * jnp.linalg.norm(probe_geometry - vsource[None] * 1e-3, axis=1)
            / sound_speed
        )
        t0_delays_hat -= jnp.min(t0_delays_hat)
        return jnp.mean(jnp.square(1e6 * (t0_delays - t0_delays_hat))) + jnp.square(
            jnp.min(t0_delays_hat)
        )

    loss_grad = jax.grad(loss_fn, argnums=0)

    vsource = jnp.array([0.0, 1.0])

    n_steps = 10000
    init_value = 5e0
    scheduler = optax.schedules.linear_schedule(
        init_value=init_value, end_value=init_value * 1e-2, transition_steps=n_steps
    )

    @jax.jit
    def update(vsource):
        vsource_grad = loss_grad(vsource)
        vsource -= scheduler(n) * vsource_grad
        return vsource

    for n in range(n_steps):
        vsource = update(vsource)

    vsource = jnp.array([vsource[0], sign * jnp.abs(vsource[1])])

    return vsource * 1e-3


if __name__ == "__main__":

    vsource = jnp.array([15.0e-3, -13e-3])

    probe_geometry = jnp.stack(
        [jnp.linspace(-10e-3, 10e-3, 80), jnp.zeros(80)], axis=-1
    )
    sound_speed = 1540

    t0_delays = (
        jnp.sign(vsource[1])
        * jnp.linalg.norm(probe_geometry - vsource[None], axis=1)
        / sound_speed
    )

    print(f"t0: {jnp.min(vsource)}")
    t0_delays -= jnp.min(t0_delays)

    vsource_hat = deduce_vsource(probe_geometry, t0_delays, sound_speed)
    print(f"vsource: [{vsource_hat[0]*1e3:.2f}, {vsource_hat[1]*1e3:.2f}]mm")

    from jaxus import load_hdf5

    data = load_hdf5(
        "/home/vincent/3-data/verasonics/usbmd/2024-04-09/S5-1_cirs_scatterers_0000.hdf5",
        frames=0,
        transmits=90,
        reduce_probe_to_2d=True,
    )
    probe_geometry = data["probe_geometry"]
    t0_delays = data["t0_delays"][0]
    sound_speed = data["sound_speed"]

    vsource_hat = deduce_vsource(probe_geometry, t0_delays, sound_speed)
    print(f"vsource: [{vsource_hat[0]*1e3:.2f}, {vsource_hat[1]*1e3:.2f}]mm")
