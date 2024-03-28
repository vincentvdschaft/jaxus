import matplotlib.pyplot as plt
import numpy as np


def plot_rf(
    ax,
    rf_data,
    start_sample=0,
    extent_m=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    aspect="auto",
    axis_in_mm=True,
):
    """Plots RF data to an axis.

    ### Parameters
        `ax` (`plt.Axes`): The axis to plot to.
        `rf_data` (`np.ndarray`): The RF data to plot.
        `start_sample` (`int`, optional): The sample number to start plotting from.
            Defaults to 0.
        `extent_m` (`list`, optional): The extent of the plot in meters. If None, the
            extent is set to the number of elements and samples. Defaults to None.
        `cmap` (`str`, optional): The colormap to use. Defaults to "viridis".
        `vmin` (`float`, optional): The minimum value of the colormap. If None, the
            minimum value is set to the 0.5 percentile of the data. Defaults to None.
        `vmax` (`float`, optional): The maximum value of the colormap. If None, the
            maximum value is set to the 99.5 percentile of the data. Defaults to None.
        `aspect` (`str`, optional): The aspect ratio of the plot. Defaults to "auto".
        `axis_in_mm` (`bool`, optional): Whether to plot the x-axis in mm. Defaults to
            True.
    """

    if extent_m is not None:
        if axis_in_mm:
            extent = np.array(extent_m) * 1e3
            xlabel = "x [mm]"
            zlabel = "z [mm]"
        else:
            extent = extent_m
            xlabel = "x [m]"
            zlabel = "z [m]"
        kwargs = {"extent": extent}

    else:
        kwargs = {"aspect": aspect}
        xlabel = "element [-]"
        zlabel = "sample [-]"

    if vmin is None and vmax is None:
        vmin = np.percentile(rf_data, 0.5)
        vmax = np.percentile(rf_data, 99.5)

    # Plot the RF data to the axis
    ax.imshow(
        rf_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(zlabel)
    # Set the yticks to start at the start_sample
    ax.set_yticks(np.arange(0, rf_data.shape[0], 50) + start_sample)


def plot_beamformed(
    ax, image, extent_m, vmin=-60, vmax=0, cmap="gray", axis_in_mm=True
):
    """Plots a beamformed image to an axis.

    ### Parameters
        `ax` (`plt.Axes`): The axis to plot to.
        `image` (`np.ndarray`): The image to plot.
        `extent_m` (`list`): The extent of the plot in meters.
        `vmin` (`float`, optional): The minimum value of the colormap. Defaults to -60.
        `vmax` (`float`, optional): The maximum value of the colormap. Defaults to 0.
        `cmap` (`str`, optional): The colormap to use. Defaults to "gray".
        `axis_in_mm` (`bool`, optional): Whether to plot the x-axis in mm. Defaults to
            True.
    """
    if axis_in_mm:
        extent = np.array(extent_m) * 1e3
        xlabel = "x [mm]"
        zlabel = "z [mm]"
    else:
        extent = extent_m
        xlabel = "x [m]"
        zlabel = "z [m]"

    ax.imshow(
        image,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect="equal",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(zlabel)


def plot_to_darkmode(fig, axes, grid=False):
    """Turns a plot into a dark plot with a black background and white text, ticks, and
    spines

    ### Args:
        `fig` (`plt.fig`): The figure handle.
        axes (plt.axes, list/tuple of plt.axews): The axes to change.
        grid (bool, optional): Whether to add a grid. Defaults to False.
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

    ### Args:
        axes (plt.Axes, np.ndarray): The axes to iterate over.
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
