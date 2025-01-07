import jax.numpy as jnp


def compute_t0_delays_from_vsource(probe_geometry, vsource, sound_speed):

    dist = jnp.linalg.norm(probe_geometry - vsource, axis=1)

    if vsource[1] > 0:
        t0_delays = -dist / sound_speed
    else:
        t0_delays = dist / sound_speed

    t0_delays -= jnp.min(t0_delays)

    return t0_delays


def compute_t0_delays_from_origin_distance_angle(
    probe_geometry, origin, distance, angle, sound_speed
):
    """Compute the t0 delays from the origin, distance and angle.

    Parameters
    ----------
    probe_geometry : array
        The probe geometry of shape (n_el, 2)
    origin : array
        The origin of the source of shape (2,)
    distance : float
        The distance from the origin to the source
    angle : float
        The angle of the source in radians
    sound_speed : float
        The speed of sound

    Returns
    -------
    t0_delays : array
        The t0 delays of shape (n_el,)
    """

    vsource = origin_distance_angle_to_vsource(origin, distance, angle)

    t0_delays = compute_t0_delays_from_vsource(probe_geometry, vsource, sound_speed)

    return t0_delays


def origin_distance_angle_to_vsource(origin, distance, angle):
    """Compute the source position from the origin, distance and angle.

    Parameters
    ----------
    origin : array
        The origin of the source of shape (2,)
    distance : float
        The distance from the origin to the source
    angle : float
        The angle of the source in radians

    Returns
    -------
    vsource : array
        The source position of shape (2,)
    """

    vsource = origin + distance * jnp.array([jnp.sin(angle), jnp.cos(angle)])

    return vsource
