"""This modules contains functions to compute the Generalized Contrast-to-Noise Ratio
(GCNR) and to plot the regions used to compute the GCNR."""

import matplotlib.pyplot as plt
import numpy as np


def gcnr(region1: np.ndarray, region2: np.ndarray, bins: int = 256):
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


def gcnr_compute_disk(
    image: np.ndarray,
    xlims_m: tuple,
    zlims_m: tuple,
    disk_pos_m: tuple,
    inner_radius_m: float,
    outer_radius_start_m: float,
    outer_radius_end_m: float,
    num_bins: int = 256,
):
    """Computes the GCNR between a circle and a surrounding annulus.

    Parameters
    ----------
    image : np.ndarray
        The image to compute the GCNR on.
    xlims_m : tuple
        The limits of the image in the x-direction in meters.
    zlims_m : tuple
        The limits of the image in the z-direction in meters.
    disk_pos_m : tuple
        The position of the disk in meters.
    inner_radius_m : float
        The inner radius of the disk in meters.
    outer_radius_start_m : float
        The start radius of the annulus in meters.
    outer_radius_end_m : float
        The end radius of the annulus in meters.
    num_bins : int
        The number of bins to use for the histogram.

    Returns
    -------
    float
        The GCNR value.
    """

    # Create meshgrid of locations for the pixels
    x_m = np.linspace(xlims_m[0], xlims_m[1], image.shape[1])
    z_m = np.linspace(zlims_m[0], zlims_m[1], image.shape[0])
    X, Z = np.meshgrid(x_m, z_m)

    # Compute the distance from the center of the circle
    r = np.sqrt((X - disk_pos_m[0]) ** 2 + (Z - disk_pos_m[1]) ** 2)

    # Create a mask for the disk
    mask_disk = r < inner_radius_m

    # Create a mask for the annulus
    mask_annulus = (r > outer_radius_start_m) & (r < outer_radius_end_m)

    # Extract the pixels from the two regions
    pixels_disk = image[mask_disk]
    pixels_annulus = image[mask_annulus]

    # Compute the GCNR
    gcnr_value = gcnr(pixels_disk, pixels_annulus, bins=num_bins)

    return gcnr_value


def gcnr_plot_disk_annulus(
    ax: plt.Axes,
    pos_m: tuple,
    inner_radius_m: float,
    outer_radius_start_m: float,
    outer_radius_end_m: float,
    opacity: float = 0.5,
):
    """Plots the disk and annulus on top of the image.

    Parameters
    ----------
        ax : plt.Axes
            The axis to plot the disk and annulus on.
        disk_pos_m : tuple
            The position of the disk in meters.
        inner_radius_m : float
            The inner radius of the disk in meters.
        outer_radius_start_m : float
            The start radius of the annulus in meters.
        outer_radius_end_m : float
            The end radius of the annulus in meters.
        opacity : float
            The opacity of the disk and annulus. Should be between 0 and 1. Defaults to
            0.5.

    """

    # Get color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot the inner circle
    circle = plt.Circle(
        pos_m,
        inner_radius_m,
        color=color_cycle[0],
        fill=False,
        linestyle="--",
        linewidth=0.5,
        alpha=opacity,
    )
    ax.add_artist(circle)

    # Draw the annulus
    circle = plt.Circle(
        pos_m,
        outer_radius_start_m,
        color=color_cycle[1],
        fill=False,
        linestyle="--",
        linewidth=0.5,
        alpha=opacity,
    )
    ax.add_artist(circle)
    circle = plt.Circle(
        pos_m,
        outer_radius_end_m,
        color=color_cycle[1],
        fill=False,
        linestyle="--",
        linewidth=0.5,
        alpha=opacity,
    )
    ax.add_artist(circle)
