from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from jaxus import use_dark_style
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit


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

    return image[x_idx, y_idx]
