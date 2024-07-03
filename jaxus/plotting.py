import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def plot_rf(
    ax,
    rf_data,
    start_sample=0,
    cmap="viridis",
    vmin=None,
    vmax=None,
    aspect="auto",
    title=None,
):
    """Plots RF data to an axis.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot to.
    rf_data : np.ndarray
        The RF data to plot.
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
    kwargs = {"aspect": aspect}
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
):
    """Plots a beamformed image to an axis.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot to.
    image : np.ndarray
        The image to plot (image should be in decibels).
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
    """

    if axis_in_mm:
        xlabel = "x [mm]"
        zlabel = "z [mm]"
        formatter = FuncFormatter(lambda x, _: f"{round(1000*x)}")
    else:
        xlabel = "x [m]"
        zlabel = "z [m]"
        formatter = FuncFormatter(lambda x, _: f"{x:.3f}")

    ax.imshow(
        image,
        extent=extent_m,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="equal",
        interpolation="none",
    )

    # Set the formatter for the major ticker on both axes
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

    # Turn the axes background black
    ax.set_facecolor("black")


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
