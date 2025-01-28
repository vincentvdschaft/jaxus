import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from jaxus import Image


def plot_rf(
    ax,
    rf_data,
    start_sample=0,
    cmap="viridis",
    vmin=None,
    vmax=None,
    aspect="auto",
    title=None,
    **kwargs,
):
    """Plots RF data to an axis.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot to.
    rf_data : np.ndarray
        The RF data to plot of shape (n_ax, n_ch).
    start_sample : int, optional
        The sample number to start plotting from. Defaults to 0.
    cmap : str, optional
        The colormap to use. Defaults to "viridis".
    vmin : float, optional
        The minimum value of the colormap. If None, the minimum value is set to the 0.5
        percentile of the data. Defaults to None.
    vmax : float, optional
        The maximum value of the colormap. If None, the maximum value is set to the 99.5
        percentile of the data. Defaults to None.
    aspect : str, optional
        The aspect ratio of the plot. Defaults to "auto".
    title : str, optional
        The title of the plot. Defaults to None.
    """
    formatter = FuncFormatter(lambda x, _: f"{int(x)}")
    kwargs = {"aspect": aspect, **kwargs}
    xlabel = "element [-]"
    zlabel = "sample [-]"
    extent = [0, rf_data.shape[1], start_sample + rf_data.shape[0], start_sample]

    if vmin is None and vmax is None:
        vmin = np.percentile(rf_data, 0.5)
        vmax = np.percentile(rf_data, 99.5)

        max_abs = max(abs(vmin), abs(vmax))
        vmin = -max_abs
        vmax = max_abs

    # Plot the RF data to the axis
    ax.imshow(
        rf_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        **kwargs,
    )

    # Set the formatter for the major ticker on both axes
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(zlabel)
    # Set the yticks to start at the start_sample
    n_ax = rf_data.shape[0]
    if n_ax >= 500:
        # Divide the axis into 4 parts and round to multiples of 100
        step = int(np.floor((n_ax / 4) / 100) * 100)
    else:
        # Divide the axis into 4 parts and round to multiples of 50
        step = int(np.floor((n_ax / 4) / 50) * 50)
    step = max(step, 10)
    ax.set_yticks(np.arange(0, n_ax, step) + start_sample)
    ax.set_xticks(np.linspace(0, rf_data.shape[1], 4))

    if title is not None:
        ax.set_title(title)


def plot_beamformed(
    ax,
    image,
    extent_m,
    vmin=-60,
    vmax=0,
    cmap="gray",
    axis_in_mm=True,
    probe_geometry=None,
    title=None,
    xlabel_override=None,
    zlabel_override=None,
    include_axes=True,
):
    """Plots a beamformed image to an axis.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot to.
    image : np.ndarray
        The image to plot of shape (n_x, n_z) (image should be in decibels).
    extent_m : list
        The extent of the plot in meters.
    vmin : float, optional
        The minimum value of the colormap. Defaults to -60.
    vmax : float, optional
        The maximum value of the colormap. Defaults to 0.
    cmap : str, optional
        The colormap to use. Defaults to "gray".
    axis_in_mm : bool, optional
        Whether to plot the x-axis in mm. Defaults to True.
    probe_geometry : np.ndarray, optional
        The probe geometry in meters of shape `(n_el, 2)`. If provided, the probe
        geometry is plotted on top of the image. Defaults to None.
    title : str, optional
        The title of the plot. Defaults to None.
    xlabel_override : str, optional
        The x-axis label to use. If None, the default label is used. Defaults to None.
    zlabel_override : str, optional
        The z-axis label to use. If None, the default label is used. Defaults to None.
    include_axes : bool, optional
        Whether to include axes, labels, and titles. Defaults to True.
    """

    if isinstance(image, Image):
        extent_m = image.extent
        image = image.data

    if axis_in_mm:
        xlabel = "x [mm]"
        zlabel = "z [mm]"
        formatter = FuncFormatter(lambda x, _: f"{round(1000 * x)}")
    else:
        xlabel = "x [m]"
        zlabel = "z [m]"
        formatter = FuncFormatter(lambda x, _: f"{x:.3f}")

    if xlabel_override is not None:
        xlabel = xlabel_override
    if zlabel_override is not None:
        zlabel = zlabel_override

    extent_m = [
        np.min(extent_m[:2]),
        np.max(extent_m[:2]),
        np.max(extent_m[-2:]),
        np.min(extent_m[-2:]),
    ]

    # Correct extent for pixel size
    pixel_size_x = (extent_m[1] - extent_m[0]) / image.shape[0]
    pixel_size_z = (extent_m[3] - extent_m[2]) / image.shape[1]
    extent_m = (
        extent_m[0] + pixel_size_x / 2,
        extent_m[1] - pixel_size_x / 2,
        extent_m[2] - pixel_size_z / 2,
        extent_m[3] + pixel_size_z / 2,
    )

    # Plot the image
    ax.imshow(
        image.T,
        extent=extent_m,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="equal",
        interpolation="none",
    )

    # Include axes, labels, and titles if include_axes is True
    if include_axes:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(zlabel)

        if probe_geometry is not None:
            ax.plot(
                [probe_geometry[0, 0], probe_geometry[-1, 0]],
                [probe_geometry[0, 1], probe_geometry[-1, 1]],
                "-|",
                markersize=6,
                color="#AA0000",
                linewidth=1,
            )

        if title is not None:
            ax.set_title(title)

    else:
        # Hide axes, ticks, and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

    # Turn the axes background black
    ax.set_facecolor("black")


def plot_beamformed_window(xlim, zlim, ax, *args, **kwargs):
    """Plots a beamformed image with a window applied to an axis.

    Parameters
    ----------
    xlim : list
        The x-limits of the window.
    zlim : list
        The z-limits of the window.
    args : list
        The arguments to pass to plot_beamformed.
    kwargs : dict
        The keyword arguments to pass to plot_beamformed.
    """
    # Plot the beamformed image
    plot_beamformed(ax, *args, **kwargs)

    # Set the x-limits and z-limits
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)


def plot_to_darkmode(fig, axes, grid=False):
    """Turns a plot into a dark plot with a black background and white text, ticks, and
    spines

    Parameters
    ----------
    fig : plt.fig
        The figure handle.
    axes : plt.axes or list/tuple of plt.axes
        The axes to change.
    grid : bool, default=False
        Whether to add a grid. Defaults to False.
    """
    assert isinstance(fig, plt.Figure), "fig must be a plt.Figure"

    # Turn the figure background black
    fig.patch.set_facecolor("black")

    color = "#BBBBBB"

    # Turn the figure text white
    if fig._suptitle:
        fig._suptitle.set_color(color)

    for ax in iterate_axes(axes):
        # Turn the axes background black
        ax.set_facecolor("black")
        # Turn the labels white
        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)
        # Turn the ticks white
        ax.tick_params(axis="x", colors=color)
        ax.tick_params(axis="y", colors=color)
        # Turn the spines white
        ax.spines["bottom"].set_color(color)
        ax.spines["left"].set_color(color)
        ax.spines["top"].set_color(color)
        ax.spines["right"].set_color(color)

        # Make the legend background transparent
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("black")
            legend.get_frame().set_edgecolor(color)
        # Make the legend text white
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(color)
        # Make the titles white
        ax.title.set_color(color)
        if grid:
            # Make the grid dark gray
            ax.grid(color=color, alpha=0.3)


def iterate_axes(axes):
    """Iterates over axes as returned by plt.subplots() works with single ax, 1d array
    of axes, and 2d array of axes.

    Parameters
    ----------
    axes : plt.Axes, np.ndarray
        The axes to iterate over.
    """
    if isinstance(axes, plt.Axes):
        yield axes
    elif isinstance(axes, np.ndarray):
        if axes.ndim == 1:
            for ax in axes:
                yield ax
        elif axes.ndim == 2:
            for row in axes:
                for ax in row:
                    yield ax
        else:
            raise ValueError("axes must be 1d or 2d array")
    else:
        raise TypeError("axes must be Axes or ndarray")


def use_light_style():
    """Sets the matplotlib style to the light style."""
    # ----------------------------------------------------------------------------------
    # Construct the path to the style sheet
    # ----------------------------------------------------------------------------------
    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)
    # Construct the path to the style sheet
    style_path = os.path.join(current_dir, "styles", "lightmode.mplstyle")

    # Use the style
    plt.style.use(style_path)


def use_dark_style():
    """Sets the matplotlib style to the dark style."""
    # ----------------------------------------------------------------------------------
    # Construct the path to the style sheet
    # ----------------------------------------------------------------------------------
    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)
    # Construct the path to the style sheet
    style_path = os.path.join(current_dir, "styles", "darkmode.mplstyle")

    # Use the style
    plt.style.use(style_path)


def symlog(data: np.ndarray, threshold: float = 1e-6):
    """Converts the data to a symmetric log scale."""
    positive_data = data >= threshold
    negative_data = data <= -threshold
    zero_data = np.logical_not(positive_data | negative_data)

    symlog_data = np.zeros_like(data)
    symlog_data[positive_data] = 20 * np.log10(data[positive_data])
    symlog_data[negative_data] = -20 * np.log10(-data[negative_data])
    symlog_data[zero_data] = 0.0

    return symlog_data
