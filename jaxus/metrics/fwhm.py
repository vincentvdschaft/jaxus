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
        vec=direction,
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


def _sample_line(image, extent, position, vec, max_offset, n_samples):
    """
    Sample a line in the image.

    Parameters
    ----------
    image : np.array
        The image to sample.
    position : tuple of floats
        The position of the line.
    vec : tuple of floats
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
    # Normalize the vector
    vec = np.array(vec)
    vec = vec / np.linalg.norm(vec)

    x = np.linspace(
        position[0] - max_offset * vec[0],
        position[0] + max_offset * vec[0],
        n_samples,
    )
    y = np.linspace(
        position[1] - max_offset * vec[1],
        position[1] + max_offset * vec[1],
        n_samples,
    )

    image_x_vals = np.linspace(extent[0], extent[1], image.shape[0])
    image_y_vals = np.linspace(extent[2], extent[3], image.shape[1])

    interpolator = RegularGridInterpolator(
        (image_x_vals, image_y_vals), image, method="linear", bounds_error=False
    )

    positions = np.stack([x, y], axis=0)
    curve = interpolator(positions.T)

    minval = np.min(curve[np.logical_not(np.isnan(curve))])
    curve = np.nan_to_num(curve, nan=minval)
    return curve, positions.T
