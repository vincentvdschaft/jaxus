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
