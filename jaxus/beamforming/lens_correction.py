"""This module contains functions to compute the travel time through an acoustic lens.
"""

import jax.numpy as jnp
from jax import grad, jit, vmap
import jax
from functools import partial


@partial(jit, static_argnums=(7,))
def compute_xl(xe, ze, xs, zs, lens_thickness, c_lens, c_medium, n_iter):
    """Computes the lateral point on the lens that the shortest path goes through based
    on Fermat's principle.

    Parameters
    ----------
    xe : float
        The x-coordinate of the element in meters.
    ze : float
        The z-coordinate of the element in meters.
    xs : float
        The x-coordinate of the pixel in meters.
    zs : float
        The z-coordinate of the pixel in meters.
    lens_thickness : float
        The thickness of the lens in meters.
    c_lens : float
        The speed of sound in the lens in m/s.
    c_medium : float
        The speed of sound in the medium in m/s.
    n_iter : int
        The number of iterations to run the Newton-Raphson method.

    Returns
    -------
    float
        The x-coordinate of the lateral point on the lens.
    """
    xl_init = lens_thickness * (xs - xe) / (zs - ze) + xe
    xl = xl_init
    for _ in range(n_iter):
        xl = xl + dxl(xe, ze, xl, xs, zs, lens_thickness, c_lens, c_medium)

        # Clip the lateral point to be in between the element and the pixel
        xl = jnp.clip(xl, jnp.min(jnp.array([xe, xs])), jnp.max(jnp.array([xe, xs])))

    return xl


@jit
def dxl(xe, ze, xl, xs, zs, zl, c_lens, c_medium):
    """Computes the update step for the lateral point on the lens that the shortest path
    using the Newton-Raphson method.

    Notes
    -----
    This result was derived by defining the total travel time through the lens and the
    medium as a function of the lateral point on the lens and then taking the
    derivative. We then have a function whose root is the lateral point on the lens that
    the shortest path goes through. We then compute the derivative and update the
    lateral point on the lens using the Newton-Raphson method:
    x_new = x - f(x) / f'(x).
    """

    eps = 1e-6

    numerator = -((xe - xl) / (c_lens * jnp.sqrt((xe - xl) ** 2 + (ze - zl) ** 2))) + (
        (xl - xs) / (c_medium * jnp.sqrt((xl - xs) ** 2 + (zl - zs) ** 2)) + eps
    )

    denominator = (
        -(
            (xe - xl) ** 2
            / (c_lens * ((xe - xl) ** 2 + (ze - zl) ** 2) ** (3 / 2) + eps)
        )
        + (1 / (c_lens * jnp.sqrt((xe - xl) ** 2 + (ze - zl) ** 2)))
        - (
            (xl - xs) ** 2
            / (c_medium * ((xl - xs) ** 2 + (zl - zs) ** 2) ** (3 / 2) + eps)
        )
        + (1 / (c_medium * jnp.sqrt((xl - xs) ** 2 + (zl - zs) ** 2) + eps))
    )

    result = -numerator / (denominator + eps)

    # Handle NaNs
    result = jnp.nan_to_num(result)

    # Clip the update step to prevent divergence
    # This value is chosen to be small enough to prevent divergence but large enough to
    # cover the distance accross a normal ultrasound aperture in a single step.
    result = jnp.clip(result, -10e-3, 10e-3)

    return result


def compute_travel_time(pos_a, pos_b, c):
    """Compute the travel time between two points."""
    return jnp.linalg.norm(pos_a - pos_b) / c


@partial(jit, static_argnums=(5,))
def compute_lensed_travel_time_2d(
    element_pos, pixel_pos, lens_thickness, c_lens, c_medium, n_iter=1
):
    """Compute the travel time through an acoustic lens.

    Parameters
    ----------
    element_pos : jnp.ndarray
        The position of the element in the lens of shape (2,).
    pixel_pos : jnp.ndarray
        The position of the pixel in the medium of shape (2,).
    lens_thickness : float
        The thickness of the lens in meters.
    c_lens : float
        The speed of sound in the lens in m/s.
    c_medium : float
        The speed of sound in the medium in m/s.
    n_iter : int
        The number of iterations to run the Newton-Raph

    Returns
    -------
    float
        The travel time in seconds.
    """
    xe, ze = element_pos
    xs, zs = pixel_pos

    # Compute the lateral point on the lens that the shortest path goes through
    xl = compute_xl(xe, ze, xs, zs, lens_thickness, c_lens, c_medium, n_iter)

    pos_lenscrossing = jnp.array([xl, lens_thickness])

    # Compute the travel time of the shortest path
    travel_time = compute_travel_time(
        element_pos, pos_lenscrossing, c_lens
    ) + compute_travel_time(pos_lenscrossing, pixel_pos, c_medium)
    return travel_time


def compute_lensed_travel_time_3d(
    element_pos, pixel_pos, lens_thickness, c_lens, c_medium, n_iter=1
):
    """Compute the travel time through an acoustic lens in 3D.

    Parameters
    ----------
    element_pos : jnp.ndarray
        The position of the element in the lens of shape (3,).
    pixel_pos : jnp.ndarray
        The position of the pixel in the medium of shape (3,).
    lens_thickness : float
        The thickness of the lens in meters.
    c_lens : float
        The speed of sound in the lens in m/s.
    c_medium : float
        The speed of sound in the medium in m/s.
    n_iter : int
        The number of iterations to run the Newton-Raphson method.

    Returns
    -------
    float
        The travel time in seconds.
    """

    element_pos_2d = jnp.array([0.0, element_pos[2]])
    pixel_pos_2d = jnp.array(
        [jnp.linalg.norm(pixel_pos[:2] - element_pos[:2]), pixel_pos[2]]
    )

    return compute_lensed_travel_time_2d(
        element_pos_2d, pixel_pos_2d, lens_thickness, c_lens, c_medium, n_iter
    )


def compute_lensed_travel_time(
    element_pos, pixel_pos, lens_thickness, c_lens, c_medium, n_iter
):
    """Compute the travel time through an acoustic lens.

    Parameters
    ----------
    element_pos : jnp.ndarray
        The position of the element in the lens of shape (2,) or (3,).
    pixel_pos : jnp.ndarray
        The position of the pixel in the medium of shape (2,) or (3,).
    lens_thickness : float
        The thickness of the lens in meters.
    c_lens : float
        The speed of sound in the lens in m/s.
    c_medium : float
        The speed of sound in the medium in m/s.
    n_iter : int
        The number of iterations to run the Newton-Raphson method.

    Returns
    -------
    float
        The travel time in seconds.
    """
    if element_pos.shape[0] == 2:
        return compute_lensed_travel_time_2d(
            element_pos, pixel_pos, lens_thickness, c_lens, c_medium, n_iter
        )
    elif element_pos.shape[0] == 3:
        return compute_lensed_travel_time_3d(
            element_pos, pixel_pos, lens_thickness, c_lens, c_medium, n_iter
        )
    else:
        raise ValueError("element_pos and pixel_pos must have shape (2,) or (3,).")
