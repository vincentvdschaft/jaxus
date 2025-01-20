"""This modules contains functions to compute the Generalized Contrast-to-Noise Ratio
(GCNR) and to plot the regions used to compute the GCNR."""

import matplotlib.pyplot as plt
import numpy as np

from jaxus import log
from jaxus.containers import Image


def gcnr(region1: np.ndarray, region2: np.ndarray, bins: int = 100):
    """Computes the Generalized Contrast-to-Noise Ratio (GCNR) between two sets of pixel
    intensities. The two input arrays are flattened and the GCNR is computed based on
    the histogram of the pixel intensities with the specified number of bins.

    Parameters
    ----------
    region1 : np.ndarray
        The first set of pixel intensities.
    region2 : np.ndarray
        The second set of pixel intensities.
    bins : int
        The number of bins to use for the histogram.

    Returns
    -------
    float
        The GCNR value.
    """

    if bins != 100:
        log.warning(
            "The number of bins is not 100 as suggested by the authors. "
            "gCNR values may not be comparable to other values."
        )

    # Flatten arrays of pixels
    region1 = region1.flatten()
    region2 = region2.flatten()

    # Compute a histogram for the two regions together to find a good set of shared bins
    _, bins = np.histogram(np.concatenate((region1, region2)), bins=bins)

    # Compute the histograms for the two regions individually with the shared bins
    hist_region_1, _ = np.histogram(region1, bins=bins, density=True)
    hist_region_2, _ = np.histogram(region2, bins=bins, density=True)

    # Normalize the histograms to unit area
    hist_region_1 /= hist_region_1.sum()
    hist_region_2 /= hist_region_2.sum()

    # Compute and return the GCNR
    return 1 - np.sum(np.minimum(hist_region_1, hist_region_2))


def gcnr_disk_annulus(
    image: Image,
    disk_center: tuple,
    disk_radius: float,
    annulus_offset: float,
    annulus_width: float,
    num_bins: int = 100,
):
    """Computes the GCNR between a circle and a surrounding annulus.

    Parameters
    ----------
    image : np.ndarray
        The image to compute the GCNR on.
    extent : np.ndarray
        The extent of the image.
    disk_center : tuple
        The position of the disk.
    disk_radius : float
        The radius of the disk.
    annulus_offset : float
        The space between disk and annulus.
    annulus_width : float
        The width of the annulus.
    num_bins : int
        The number of bins to use for the histogram.

    Returns
    -------
    float
        The GCNR value.
    """

    # Create meshgrid of locations for the pixels
    x_grid, z_grid = image.grid

    # Compute the distance from the center of the circle
    r = np.sqrt((x_grid - disk_center[0]) ** 2 + (z_grid - disk_center[1]) ** 2)

    # Create a mask for the disk
    mask_disk = r < disk_radius

    annulus_r0, annulus_r1 = (
        disk_radius + annulus_offset,
        disk_radius + annulus_offset + annulus_width,
    )

    # Create a mask for the annulus
    mask_annulus = (r > annulus_r0) & (r < annulus_r1)

    # Extract the pixels from the two regions
    pixels_disk = image.data[mask_disk]
    pixels_annulus = image.data[mask_annulus]

    # Compute the GCNR
    gcnr_value = gcnr(pixels_disk, pixels_annulus, bins=num_bins)

    return gcnr_value


def gcnr_plot_disk_annulus(
    ax: plt.Axes,
    disk_center: tuple,
    disk_radius: float,
    annulus_offset: float,
    annulus_width: float,
    opacity: float = 1.0,
    linewidth: float = 0.5,
    color1: str = "C0",
    color2: str = "C1",
):
    """Plots the disk and annulus on top of the image.

    Parameters
    ----------
        ax : plt.Axes
            The axis to plot the disk and annulus on.
        disk_center : tuple
            The position of the disk in meters.
        disk_radius : float
            The inner radius of the disk in meters.
        annulus_offset : float
            The space between disk and annulus.
        annulus_width : float
            The width of the annulus.
        opacity : float
            The opacity of the disk and annulus. Should be between 0 and 1. Defaults to
            0.5.

    """

    # Plot the inner circle
    disk = plt.Circle(
        disk_center,
        disk_radius,
        color=color1,
        fill=False,
        linestyle="--",
        linewidth=linewidth,
        alpha=opacity,
    )
    ax.add_artist(disk)

    # Draw the annulus
    annul0 = plt.Circle(
        disk_center,
        disk_radius + annulus_offset,
        color=color2,
        fill=False,
        linestyle="--",
        linewidth=linewidth,
        alpha=opacity,
    )
    ax.add_artist(annul0)
    annul1 = plt.Circle(
        disk_center,
        disk_radius + annulus_offset + annulus_width,
        color=color2,
        fill=False,
        linestyle="--",
        linewidth=linewidth,
        alpha=opacity,
    )
    ax.add_artist(annul1)

    return disk, annul0, annul1
