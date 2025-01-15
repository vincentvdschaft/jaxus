from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from jaxus.containers import Image


def fwhm(curve, width, required_repeats=3, log_scale=False):
    """
    Compute the full width at half maximum of a curve.

    Parameters
    ----------
    curve : np.array
        The curve to compute the FWHM on.
    width : float
        The width of the curve.
    required_repeats : int
        The number of times the curve should cross the half maximum.

    Returns
    -------
    float
        The FWHM of the curve.
    """
    _, idx_left, idx_right = find_fwhm_indices(
        curve, required_repeats, log_scale=log_scale
    )

    dx = width / (curve.size - 1)
    return (idx_right - idx_left) * dx


def find_fwhm_indices(curve, required_repeats=3, log_scale=False):
    """Finds the peak and the left and right indices of the FWHM."""
    # Find the half maximum

    if log_scale:
        curve = curve - np.max(curve)
        half_max = np.max(curve) - 20 * np.log10(2)
    else:
        curve = curve / np.max(curve)
        half_max = np.max(curve) / 2

    idx_peak = np.argmax(curve)

    right = curve[idx_peak:]
    left = curve[: idx_peak + 1]
    idx_right = idx_peak + _find_crossing(right, half_max, required_repeats)
    idx_left = idx_peak - _find_crossing(left[::-1], half_max, required_repeats)

    return idx_peak, idx_left, idx_right


def _find_crossing(curve, value, required_repeats):
    """
    Find the crossings of a curve with a value.

    Parameters
    ----------
    curve : np.array
        The curve to find the crossings on.
    value : float
        The value to find the crossings with.
    required_repeats : int
        The number of consecutive samples that should be below the value.

    Returns
    -------
    crossings : np.array
        The indices of the crossings.
    """
    index = 0
    repeats = 0
    found = False
    for index, sample in enumerate(curve):
        if sample <= value:
            repeats += 1
            if repeats >= required_repeats:
                found = True
                break
        else:
            repeats = 0
        index += 1

    if found:
        return index - (repeats - 1)
    else:
        print("No crossing found")
        return curve.size - 1


def fwhm_image(
    image: Image, position, direction, max_offset, required_repeats=3, n_samples=256
):
    """Computes the FWHM of a line profile in the image.

    Parameters
    ----------
    image : Image
        The image to compute the FWHM on.
    position : np.array
        The position of the line profile of shape (2,).
    direction : np.array
        The direction of the line profile of shape (2,).
    max_offset : float
        The maximum offset from the position to sample the line profile. The line
        profile will have a length of 2 * max_offset.
    required_repeats : int
        The number of consecutive samples that should be below the value for it to
        count to reject noise.

    Returns
    -------
    fwhm_value : float
        The FWHM value of the line profile
    """

    curve, _ = _sample_line(
        image=image.data,
        extent=image.extent,
        position=position,
        direction=direction,
        max_offset=max_offset,
        n_samples=n_samples,
    )

    fwhm_value = fwhm(curve, max_offset * 2, required_repeats, log_scale=image.in_db)

    return fwhm_value


def correct_fwhm_point(image: Image, position, max_diff=0.6e-3):
    """Find the point with the maximum intensity within a certain distance of a given point.

    Parameters
    ----------
    image : Image
        The image to search in.
    position : np.array
        The position to search around.
    max_diff : float
        The maximum distance from the position to search.

    Returns
    -------
    new_position : np.array
        The corrected position which is at most `max_diff` away from the original
        position.
    """

    position = np.array(position)

    if max_diff == 0.0:
        return position

    distances = np.linalg.norm(image.flatgrid - position, axis=1)

    mask = distances <= max_diff
    candidate_intensities = image.data.flatten()[mask]
    candidate_points = image.flatgrid[mask]

    if candidate_intensities.size == 0:
        return position

    idx = np.argmax(candidate_intensities)

    return candidate_points[idx]


def _sample_line(image, extent, position, direction, max_offset, n_samples):
    """
    Sample a line in the image.

    Parameters
    ----------
    image : np.array
        The image to sample.
    position : tuple of floats
        The position of the line.
    direction : tuple of floats
        The direction of the line.
    max_offset : float
        The maximum offset from the position.
    n_samples : int
        The number of samples to take.

    Returns
    -------
    samples : np.array
        The samples along the line.
    positions : np.array
        The positions of the samples. Can be used to plot the line in the image.
    """

    image_x_vals = np.linspace(extent[0], extent[1], image.shape[0])
    image_y_vals = np.linspace(extent[2], extent[3], image.shape[1])

    interpolator = RegularGridInterpolator(
        (image_x_vals, image_y_vals), image, method="linear", bounds_error=False
    )

    positions = _get_positions(
        center=position, direction=direction, max_offset=max_offset, n_samples=n_samples
    )
    print(positions.shape)

    curve = interpolator(positions)

    minval = np.min(curve[np.logical_not(np.isnan(curve))])
    curve = np.nan_to_num(curve, nan=minval)
    return curve, positions


def _get_positions(center, direction, max_offset, n_samples):
    """Defines evenly spaced positions along a line of length 2 * max_offset centered
    at center in the direction of direction.

    Parameters
    ----------
    center : np.array
        The center of the line of shape (2,).
    direction : np.array
        The direction of the line of shape (2,).
    max_offset : float
        The maximum offset from the center.
    n_samples : int
        The number of points along the line.

    Returns
    -------
    positions : np.array
        The positions along the line of shape (n_samples, 2).
    """

    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)

    x = np.linspace(
        center[0] - max_offset * direction[0],
        center[0] + max_offset * direction[0],
        n_samples,
    )
    y = np.linspace(
        center[1] - max_offset * direction[1],
        center[1] + max_offset * direction[1],
        n_samples,
    )

    positions = np.stack([x, y], axis=0)

    return positions.T


def plot_fwhm(
    ax,
    position,
    direction,
    max_offset,
    color1="C0",
    color2="C1",
    linewidth=0.5,
    **kwargs
):
    """Plot a cross indicating the axial and lateral FWHM of a line profile.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    position : tuple
        The position of the line profile.
    direction : tuple
        The axial direction of the line profile.
    max_offset : float
        The maximum offset from the position to sample the line profile.
    color1 : str, default="C0"
        The color of the axial FWHM line.
    color2 : str, default="C1"
        The color of the lateral FWHM line.
    linewidth : float, default=0.5
        The width of the lines.
    **kwargs : dict
        Additional arguments to pass to the plot function.

    Returns
    -------
    line1 : matplotlib.lines.Line2D
        The axial FWHM line.
    line2 : matplotlib.lines.Line2D
        The lateral FWHM line.
    """

    positions = _get_positions(
        center=position, direction=direction, max_offset=max_offset, n_samples=2
    )
    direction_orth = np.array([-direction[1], direction[0]])

    (line1,) = ax.plot(
        positions[:, 0],
        positions[:, 1],
        color=color1,
        linewidth=linewidth,
        linestyle="--",
        **kwargs
    )

    positions_orth = _get_positions(
        center=position, direction=direction_orth, max_offset=max_offset, n_samples=2
    )

    (line2,) = ax.plot(
        positions_orth[:, 0],
        positions_orth[:, 1],
        color=color2,
        linewidth=linewidth,
        linestyle="--",
        **kwargs
    )

    return line1, line2
