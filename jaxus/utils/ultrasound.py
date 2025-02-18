import jax.numpy as jnp
import numpy as np
import jax
from jax import jit
import optax
from scipy.interpolate import RegularGridInterpolator
from jaxus import log
from functools import partial


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
        log.warning("Normalizing in log_compress function is deprecated.")
        beamformed = beamformed / jnp.clip(jnp.max(beamformed), 1e-12)
    beamformed = 20 * jnp.log10(beamformed + 1e-12)

    return beamformed


def deduce_vsource(probe_geometry, t0_delays, sound_speed, n_steps=10000, lr=5e-2):
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
    print(sign)

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

    init_value = lr
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

    vsource = jnp.array([vsource[0], -sign * jnp.abs(vsource[1])])

    return vsource * 1e-3


def scan_convert(
    polar_image: np.array, polar_extent, target_pixel_positions, fill_value=0.0
):
    """Performs scan conversion on a polar image. Transforms the cartesian pixel grid
    to polar coordinates and interpolates in the polar domain.

    Parameters
    ----------
    polar_image : np.array
        The polar image to scan convert of shape (n_th, n_r).
    polar_extent : tuple
        The extent of the polar image in the form (th_min, th_max, r_min, r_max).
    target_pixel_positions : np.array
        The target pixel grid to interpolate to of shape (..., 2).
    fill_value : float, default=0.0
        The fill value for the interpolation.

    Returns
    -------
    image : np.array
        The scan-converted image of shape (...,).
    """
    polar_image = np.asarray(polar_image)
    assert polar_image.ndim == 2, "polar_image must be 2D."
    assert len(polar_extent) == 4, "polar_extent must have 4 elements."

    n_th, n_r = polar_image.shape
    th_source = np.linspace(polar_extent[0], polar_extent[1], n_th)
    r_source = np.linspace(polar_extent[2], polar_extent[3], n_r)

    target_shape = target_pixel_positions.shape

    assert target_shape[-1] == 2, "target_pixel_positions must have shape (..., 2)."
    target_pixel_positions = np.reshape(target_pixel_positions, (-1, 2))

    target_x, target_z = target_pixel_positions[:, 0], target_pixel_positions[:, 1]

    r_target = np.linalg.norm(target_pixel_positions, axis=1)
    th_target = np.arctan2(target_z, target_x)

    interpolator = RegularGridInterpolator(
        (th_source, r_source), polar_image, bounds_error=False, fill_value=fill_value
    )

    image = interpolator((th_target, r_target)).reshape(target_shape[:-1])
    return image


def t0_delays_from_vsource(
    probe_geometry, vsource_angle, vsource_depth, sound_speed, shift_to_zero=True
):
    """Computes the t0 delays for a probe given a virtual source defined by an angle and depth.

    Parameters
    ----------
    probe_geometry : jnp.array
        The positions of the elements of shape (n_el, 2).
    vsource_angle : float
        The angle of the virtual source with respect to the origin in radians.
    vsource_depth : float
        The depth of the virtual source in meters. Set to negative values for diverging
        wave. Set to inf for plane wave.
    sound_speed : float
        The speed of sound in the medium.

    Returns
    -------
    t0_delays : jnp.array
        The t0 delays in seconds, where the smallest delay is 0.
    """
    virtual_source = vsource_pos(vsource_angle, vsource_depth).reshape((-1, 2))

    t0_delays_vsource = (
        -jnp.sign(vsource_depth)
        * jnp.linalg.norm(probe_geometry - virtual_source, axis=-1)
        / sound_speed
    )
    v = jnp.stack([jnp.sin(vsource_angle), jnp.cos(vsource_angle)], axis=-1)
    t0_delays_pw = v @ probe_geometry.T / sound_speed

    t0_delays = jnp.where(jnp.isinf(vsource_depth), t0_delays_pw, t0_delays_vsource)

    t0_delays -= jnp.where(shift_to_zero, jnp.min(t0_delays), 0.0)

    return t0_delays


def vsource_pos(angle, distance):
    """Computes the position of a virtual source given an angle and distance.

    Parameters
    ----------
    angle : float
        The angle of the virtual source in radians.
    distance : float
        The distance of the virtual source in meters.

    Returns
    -------
    position : jnp.array
        The position of the virtual source in meters.
    """
    return jnp.stack([jnp.sin(angle), jnp.cos(angle)], axis=-1) * distance


def vsource_angle(vsource_pos):
    """Computes the angle of a virtual source given a position.

    Parameters
    ----------
    vsource_pos : jnp.array
        The position of the virtual source in meters of shape (..., 2).

    Returns
    -------
    angle : jnp.array
        The angle of the virtual source in radians in the range [-pi/2, pi/2].
        of shape (...,).
    """
    return jnp.arctan2(vsource_pos[..., 0], vsource_pos[..., 1]) - jnp.where(
        vsource_pos[..., 1] < 0, jnp.pi, 0
    )


def vsource_depth(vsource_pos):
    """Computes the depth of a virtual source given a position.

    Parameters
    ----------
    vsource_pos : jnp.array
        The position of the virtual source in meters.

    Returns
    -------
    depth : float
        The depth of the virtual source in meters.
    """
    return jnp.linalg.norm(vsource_pos, axis=-1) * jnp.sign(vsource_pos[..., 1])


if __name__ == "__main__":

    vsource = jnp.array([-10.0e-3, -10e-3])
    angle, depth = vsource_angle(vsource), vsource_depth(vsource)

    print(f"vsource: [{vsource[0]*1e3:.2f}, {vsource[1]*1e3:.2f}]mm")
    print(f"angle: {angle*180/jnp.pi:.2f}°, depth: {depth*1e3:.2f}mm")

    probe_geometry = jnp.stack(
        [jnp.linspace(-10e-3, 10e-3, 80), jnp.zeros(80)], axis=-1
    )
    sound_speed = 1540

    t0_delays = t0_delays_from_vsource(probe_geometry, angle, depth, sound_speed)
    # print(t0_delays)

    print(f"angle: {angle*180/jnp.pi:.2f}°, depth: {depth*1e3:.2f}mm")

    vsource_hat = deduce_vsource(probe_geometry, t0_delays, sound_speed, n_steps=30000)
    print(f"vsource: [{vsource_hat[0]*1e3:.2f}, {vsource_hat[1]*1e3:.2f}]mm")

    # from jaxus import load_hdf5

    # data = load_hdf5(
    #     "/home/vincent/3-data/verasonics/usbmd/2024-04-09/S5-1_cirs_scatterers_0000.hdf5",
    #     frames=0,
    #     transmits=90,
    #     reduce_probe_to_2d=True,
    # )
    # probe_geometry = data["probe_geometry"]
    # t0_delays = data["t0_delays"][0]
    # sound_speed = data["sound_speed"]

    # vsource_hat = deduce_vsource(probe_geometry, t0_delays, sound_speed)
    # print(f"vsource: [{vsource_hat[0]*1e3:.2f}, {vsource_hat[1]*1e3:.2f}]mm")
