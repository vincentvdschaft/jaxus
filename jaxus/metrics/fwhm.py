from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from jaxus.containers import Image


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

    im_width = extent[1] - extent[0]
    im_height = extent[3] - extent[2]

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

    x_idx = np.round((x - extent[0]) / im_width * image.shape[0]).astype(int)
    y_idx = np.round((y - extent[2]) / im_height * image.shape[1]).astype(int)

    x_idx = np.clip(x_idx, 0, image.shape[0] - 1)
    y_idx = np.clip(y_idx, 0, image.shape[1] - 1)

    fwhm(image[x_idx, y_idx], 1.0, 3, log_scale=True)

    return image[x_idx, y_idx], np.stack((x, y), axis=1)


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
    # Find the half maximum
    if log_scale:
        half_max = np.max(curve) - 10
    else:
        half_max = np.max(curve) / 2

    idx_peak = np.argmax(curve)

    right = curve[idx_peak:]
    left = curve[:idx_peak]
    idx_right = idx_peak + _find_crossing(right, half_max, required_repeats)
    idx_left = _find_crossing(left[::-1], half_max, required_repeats)

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
        The number of times the curve should cross the value.

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
            if repeats == required_repeats:
                found = True
                break
        else:
            repeats = 0
        index += 1

    if found:
        return index - repeats
    else:
        print("No crossing found")
        return curve.size - 1


def fwhm_image(image: Image, position, direction, max_offset, required_repeats=3):

    curve, positions = _sample_line(
        image=image.data,
        extent=image.extent,
        position=position,
        vec=direction,
        max_offset=max_offset,
        n_samples=128,
    )

    fwhm_value = fwhm(
        curve, max_offset * 2, required_repeats, log_scale=image.log_compressed
    )

    return fwhm_value


def correct_fwhm_point(image: Image, position, max_diff=0.6e-3):

    # 2D interpolation over the entire image
    pixel_coords = (image.x_vals, image.y_vals)
    interpolator = RegularGridInterpolator(
        pixel_coords, image.data, bounds_error=False, fill_value=0
    )
    x_vals = np.linspace(position[0] - max_diff, position[0] + max_diff, 128)
    y_vals = np.linspace(position[1] - max_diff, position[1] + max_diff, 128)

    xx, yy = np.meshgrid(x_vals, y_vals, indexing="ij")

    im = interpolator((xx, yy))

    row, col = np.unravel_index(np.argmax(im), im.shape)

    x = x_vals[row]
    y = y_vals[col]
    return np.array([x, y])
