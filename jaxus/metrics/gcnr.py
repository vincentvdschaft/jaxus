"""This modules contains functions to compute the Generalized Contrast-to-Noise Ratio
(GCNR) and to plot the regions used to compute the GCNR."""

import matplotlib.pyplot as plt
import numpy as np


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
    image: np.ndarray,
    extent: np.ndarray,
    disk_center: tuple,
    disk_r: float,
    annul_r0: float,
    annul_r1: float,
    num_bins: int = 128,
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
    disk_r : float
        The radius of the disk.
    annul_r0 : float
        The start radius of the annulus.
    annul_r1 : float
        The end radius of the annulus.
    num_bins : int
        The number of bins to use for the histogram.

    Returns
    -------
    float
        The GCNR value.
    """

    # Create meshgrid of locations for the pixels
    x = np.linspace(extent[0], extent[1], image.shape[0])
    z = np.linspace(extent[2], extent[3], image.shape[1])
    x_grid, z_grid = np.meshgrid(x, z, indexing="ij")

    # Compute the distance from the center of the circle
    r = np.sqrt((x_grid - disk_center[0]) ** 2 + (z_grid - disk_center[1]) ** 2)

    # Create a mask for the disk
    mask_disk = r < disk_r

    # Create a mask for the annulus
    mask_annulus = (r > annul_r0) & (r < annul_r1)

    # Extract the pixels from the two regions
    pixels_disk = image[mask_disk]
    pixels_annulus = image[mask_annulus]

    # Compute the GCNR
    gcnr_value = gcnr(pixels_disk, pixels_annulus, bins=num_bins)

    return gcnr_value


def gcnr_plot_disk_annulus(
    ax: plt.Axes,
    disk_center: tuple,
    disk_r: float,
    annul_r0: float,
    annul_r1: float,
    opacity: float = 0.5,
    linewidth: float = 0.5,
):
    """Plots the disk and annulus on top of the image.

    Parameters
    ----------
        ax : plt.Axes
            The axis to plot the disk and annulus on.
        disk_center : tuple
            The position of the disk in meters.
        disk_r : float
            The inner radius of the disk in meters.
        annul_r0 : float
            The start radius of the annulus in meters.
        annul_r1 : float
            The end radius of the annulus in meters.
        opacity : float
            The opacity of the disk and annulus. Should be between 0 and 1. Defaults to
            0.5.

    """

    # Get color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot the inner circle
    circle = plt.Circle(
        disk_center,
        disk_r,
        color=color_cycle[0],
        fill=False,
        linestyle="--",
        linewidth=linewidth,
        alpha=opacity,
    )
    ax.add_artist(circle)

    # Draw the annulus
    circle = plt.Circle(
        disk_center,
        annul_r0,
        color=color_cycle[1],
        fill=False,
        linestyle="--",
        linewidth=linewidth,
        alpha=opacity,
    )
    ax.add_artist(circle)
    circle = plt.Circle(
        disk_center,
        annul_r1,
        color=color_cycle[1],
        fill=False,
        linestyle="--",
        linewidth=linewidth,
        alpha=opacity,
    )
    ax.add_artist(circle)
