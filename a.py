from jaxus import (
    Image,
    plot_beamformed,
    image_measure_gcnr_disk_annulus,
    image_measure_fwhm,
    gcnr_plot_disk_annulus,
    fwhm,
    correct_fwhm_point,
    fwhm_image,
)
import numpy as np
import matplotlib.pyplot as plt
from myplotlib import *
from jaxus.metrics.fwhm import _sample_line, find_fwhm_indices

use_style(STYLE_DARK)

extent = [
    -20,
    20,
    0,
    30,
]

disk_pos = np.array((0, 15)) * 1e-3
disk_radius = 7

scat_pos = np.array([4.062e-3, 7.286e-2])
vsource = np.array((0, -15)) * 1e-3

max_offset = 4e-3
n_samples = 100


image_loaded = Image.load("image_frame_0000.hdf5")


def update_plot(scat_pos):
    vec = scat_pos - vsource
    vec_orth = np.array((-vec[1], vec[0]))
    result_axial, positions_axial = _sample_line(
        image_loaded.data,
        image_loaded.extent,
        scat_pos,
        vec,
        max_offset,
        n_samples,
    )

    result_lateral, positions_lateral = _sample_line(
        image_loaded.data,
        image_loaded.extent,
        scat_pos,
        vec_orth,
        max_offset,
        n_samples,
    )
    plot_beamformed(ax_im, image_loaded.data, np.array(image_loaded.extent))
    gcnr_plot_disk_annulus(
        ax_im,
        disk_center=disk_pos,
        disk_r=disk_radius,
        annulus_offset=disk_radius + 1e-3,
        annulus_width=disk_radius + 3e-3,
    )
    # scat_indicator.set_data(scat_pos[0][None], scat_pos[1][None])
    scat_line_axial_indicator.set_data(positions_axial[:, 0], positions_axial[:, 1])
    scat_line_lateral_indicator.set_data(
        positions_lateral[:, 0], positions_lateral[:, 1]
    )
    line_axial.set_ydata(result_axial)
    line_lateral.set_ydata(result_lateral)

    fwhm_axial = fwhm_image(
        image=image_loaded, position=scat_pos, direction=vec, max_offset=max_offset
    )
    fwhm_lateral = fwhm_image(
        image=image_loaded, position=scat_pos, direction=vec_orth, max_offset=max_offset
    )
    print(f"{fwhm_axial*1e3:.1f} mm, {fwhm_lateral*1e3:.1f} mm")

    idx_peak, idx_left, idx_right = find_fwhm_indices(
        result_axial, required_repeats=3, log_scale=image_loaded.log_compressed
    )
    print(idx_peak, idx_left, idx_right)
    dist_vals = np.linspace(-max_offset, max_offset, n_samples)

    vline_left_axial.set_xdata([dist_vals[idx_left]])
    vline_right_axial.set_xdata([dist_vals[idx_right]])
    vline_peak_axial.set_xdata([dist_vals[idx_peak]])

    idx_peak, idx_left, idx_right = find_fwhm_indices(
        result_lateral, required_repeats=3, log_scale=image_loaded.log_compressed
    )
    vline_left_lateral.set_xdata([dist_vals[idx_left]])
    vline_right_lateral.set_xdata([dist_vals[idx_right]])
    vline_peak_lateral.set_xdata([dist_vals[idx_peak]])

    image_measure_fwhm(
        image_loaded,
        scat_pos,
        vec,
        max_offset=max_offset,
        correct_position=True,
        max_correction_distance=1.0e-3,
    )

    plt.draw()


def on_click(event):
    print("click")
    global scat_pos, disk_pos
    # Check if mouse click
    if event.xdata is None or event.ydata is None:
        return
    if event.button == 1:
        scat_pos = np.array([event.xdata, event.ydata])
        scat_pos = correct_fwhm_point(image_loaded, scat_pos, max_diff=1.0e-3)
    elif event.button == 3:
        disk_pos = np.array([event.xdata, event.ydata])
    else:
        return

    update_plot(scat_pos)


def on_key(event):
    step = 0.1e-3
    if event.key == "right":
        scat_pos[0] += step
    elif event.key == "left":
        scat_pos[0] -= step
    elif event.key == "up":
        scat_pos[1] -= step
    elif event.key == "down":
        scat_pos[1] += step
    else:
        return
    print(scat_pos)

    update_plot(scat_pos)


# ==============================================================================
# Initialize the figure
# ==============================================================================
# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
fig_w, fig_h = 8, 11
margin_left, margin_right = 1.0, 0.3
margin = margin_left + margin_right
spacing = 0.7
grid_spacing = 0.2

y_line = margin_right
lineplot_h = 1

# ------------------------------------------------------------------------------
# Create figure and axes
# ------------------------------------------------------------------------------
fig = MPLFigure(figsize=(fig_w, fig_h))
axes = fig.add_axes_grid(
    n_rows=2,
    n_cols=1,
    x=margin_left,
    y=margin_right,
    width=fig_w - margin,
    height=lineplot_h,
    spacing=grid_spacing,
)
ax_line_axial, ax_line_lateral = axes[:, 0]
remove_internal_ticks_labels(axes)

ax_line_lateral.set_xlabel("Lateral distance [mm]")
for ax in [ax_line_axial, ax_line_lateral]:
    ax.set_ylabel("[dB]")
ax_line_axial.set_title("Profile")

ax_im = fig.add_ax(
    x=margin_left,
    y=margin_right + 2 * lineplot_h + spacing + grid_spacing,
    width=fig_w - margin,
    aspect=image_loaded.extent,
)


# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
(line_axial,) = ax_line_axial.plot(
    np.linspace(-max_offset, max_offset, n_samples), np.zeros(n_samples)
)
(line_lateral,) = ax_line_lateral.plot(
    np.linspace(-max_offset, max_offset, n_samples),
    np.zeros(n_samples),
    "C1",
)
ax_line_axial.set_ylim(-60, 0)
ax_line_lateral.set_ylim(-60, 0)
mm_formatter_ax(ax_line_axial, x=True, y=False)
mm_formatter_ax(ax_line_lateral, x=True, y=False)


(vline_left_axial,) = ax_line_axial.plot([0, 0], [-60, 0], color="gray", linestyle="--")
(vline_peak_axial,) = ax_line_axial.plot([0, 0], [-60, 0], color="gray", linestyle="--")
(vline_right_axial,) = ax_line_axial.plot(
    [0, 0], [-60, 0], color="gray", linestyle="--"
)

(vline_left_lateral,) = ax_line_lateral.plot(
    [0, 0], [-60, 0], color="gray", linestyle="--"
)
(vline_peak_lateral,) = ax_line_lateral.plot(
    [0, 0], [-60, 0], color="gray", linestyle="--"
)
(vline_right_lateral,) = ax_line_lateral.plot(
    [0, 0], [-60, 0], color="gray", linestyle="--"
)


plot_beamformed(ax_im, image_loaded.data, np.array(image_loaded.extent))

(scat_line_axial_indicator,) = ax_im.plot(
    np.zeros(n_samples), np.zeros(n_samples), "C0--", linewidth=0.5
)
(scat_line_lateral_indicator,) = ax_im.plot(
    np.zeros(n_samples), np.zeros(n_samples), "C1--", linewidth=0.5
)

update_plot(scat_pos)

cid = fig.fig.canvas.mpl_connect("button_release_event", on_click)
cid_key = fig.fig.canvas.mpl_connect("key_press_event", on_key)


# plt.tight_layout()
plt.savefig("image.png", bbox_inches="tight")
plt.show()

image_loaded.save("image_frame_0000_measured.hdf5")
