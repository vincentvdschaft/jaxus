from jaxus import (
    Image,
    plot_beamformed,
    image_measure_gcnr_disk_annulus,
    gcnr_disk_annulus,
    image_measure_fwhm,
    gcnr_plot_disk_annulus,
    fwhm,
    correct_fwhm_point,
    fwhm_image,
)
from jaxus.utils import log
import numpy as np
import matplotlib.pyplot as plt
from myplotlib import *
from jaxus.metrics.fwhm import _sample_line, find_fwhm_indices
from matplotlib.widgets import Button
import argparse
from dataclasses import dataclass

parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path, default="image_frame_0000.hdf5", nargs="?")
args = parser.parse_args()

path = Path(args.file)
image_loaded = Image.load(path)
use_style(STYLE_DARK)


def _arrow_key_to_vector(key, step=0.1e-3):
    if key == "right":
        return np.array([1, 0]) * step
    elif key == "left":
        return np.array([-1, 0]) * step
    elif key == "up":
        return np.array([0, -1]) * step
    elif key == "down":
        return np.array([0, 1]) * step
    else:
        return None


@dataclass
class FWHMCurve:
    positions: np.ndarray
    values: np.ndarray


@dataclass
class FWHMLinePlot:
    ax: matplotlib.axes.Axes
    line: matplotlib.lines.Line2D
    vline_left: matplotlib.lines.Line2D
    vline_peak: matplotlib.lines.Line2D
    vline_right: matplotlib.lines.Line2D


class FWHMMarker:

    def __init__(self, ax, position, direction, max_offset, n_samples, active=True):
        self.ax = ax
        self.position = position
        self.direction = direction
        self.max_offset = max_offset
        self.n_samples = n_samples
        self.active = active

        self.positions_axial = None
        self.positions_lateral = None

        (self.scat_line_axial_indicator,) = self.ax.plot(
            np.zeros(n_samples), np.zeros(n_samples), "C0--", linewidth=0.5
        )
        (self.scat_line_lateral_indicator,) = self.ax.plot(
            np.zeros(n_samples), np.zeros(n_samples), "C1--", linewidth=0.5
        )

        self.update(
            self.position, self.direction, self.positions_axial, self.positions_lateral
        )

    def draw(self):
        if self.positions_axial is None:
            return
        self.scat_line_axial_indicator.set_data(
            self.positions_axial[:, 0],
            self.positions_axial[:, 1],
        )
        self.scat_line_lateral_indicator.set_data(
            self.positions_lateral[:, 0],
            self.positions_lateral[:, 1],
        )

        if self.active:
            # Set the color of the line
            self.scat_line_axial_indicator.set_color("C0")
            self.scat_line_lateral_indicator.set_color("C1")
        else:
            self.scat_line_axial_indicator.set_color("white")
            self.scat_line_lateral_indicator.set_color("white")

        plt.draw()

    def update(self, position, direction, positions_axial, positions_lateral):
        self.position = position
        self.direction = direction
        self.positions_axial = positions_axial
        self.positions_lateral = positions_lateral
        self.draw()

    def deactivate(self):
        self.active = False
        self.draw()


class GCNRDiskAnnulusMarker:
    pass


class EvalTool:
    TARGET_FWHM, TARGET_GCNR = 0, 1

    def __init__(self, image):
        self.image = image
        self.spacing = 0.7
        self.margin_left = 1.0
        self.margin_right = self.spacing
        self.margin_top = 0.3
        self.margin_bottom = 0.6
        self.im_width = 8
        self.margin = 0.3
        self.grid_spacing = 0.2
        self.button_width = 2
        self.arrow_target = EvalTool.TARGET_FWHM

        self.y_line = self.margin_top
        self.lineplot_h = 1

        self.fig_w = (
            self.margin_left
            + self.im_width
            + self.spacing
            + self.button_width
            + self.margin_right
        )
        im_height = image.aspect * self.im_width
        self.fig_h = (
            self.margin_top
            + 2 * self.lineplot_h
            + self.spacing
            + self.grid_spacing
            + im_height
            + self.margin_bottom
        )

        self.fig = MPLFigure(figsize=(self.fig_w, self.fig_h))

        self.max_offset = 4e-3
        self.n_samples = 100

        # Axes
        self.fwhm_axes_grid = None
        self.fwhm_lineplot_axial = None
        self.fwhm_lineplot_lateral = None
        self.ax_im = None

        self.button_save_fwhm = None

        self.active_fwhm_marker = None
        self.fwhm_curve_axial = None
        self.fwhm_curve_lateral = None
        self.vsource = np.array((0, -15)) * 1e-3

        self.fwhm_axes_grid = self.fig.add_axes_grid(
            n_rows=2,
            n_cols=1,
            x=self.margin_left,
            y=self.margin_top,
            width=self.im_width,
            height=self.lineplot_h,
            spacing=self.grid_spacing,
        )
        self.ax_line_axial, self.ax_line_lateral = self.fwhm_axes_grid[:, 0]
        remove_internal_ticks_labels(self.fwhm_axes_grid)

        self.ax_line_lateral.set_xlabel("Lateral distance [mm]")
        for ax in [self.ax_line_axial, self.ax_line_lateral]:
            ax.set_ylabel("[dB]")
        self.ax_line_axial.set_title("Profile")

        self.ax_im = self.fig.add_ax(
            x=self.margin_left,
            y=self.margin_top + 2 * self.lineplot_h + self.spacing + self.grid_spacing,
            width=self.im_width,
            aspect=self.image.extent,
        )

        self.fwhm_lineplot_axial = FWHMLinePlot(
            ax=self.ax_line_axial,
            line=self.ax_line_axial.plot(
                np.linspace(-self.max_offset, self.max_offset, self.n_samples),
                np.zeros(self.n_samples),
            )[0],
            vline_left=self.ax_line_axial.plot(
                [0, 0], [-60, 0], color="gray", linestyle="--"
            )[0],
            vline_peak=self.ax_line_axial.plot(
                [0, 0], [-60, 0], color="gray", linestyle="--"
            )[0],
            vline_right=self.ax_line_axial.plot(
                [0, 0], [-60, 0], color="gray", linestyle="--"
            )[0],
        )

        self.fwhm_lineplot_lateral = FWHMLinePlot(
            ax=self.ax_line_lateral,
            line=self.ax_line_lateral.plot(
                np.linspace(-self.max_offset, self.max_offset, self.n_samples),
                np.zeros(self.n_samples),
                "C1",
            )[0],
            vline_left=self.ax_line_lateral.plot(
                [0, 0], [-60, 0], color="gray", linestyle="--"
            )[0],
            vline_peak=self.ax_line_lateral.plot(
                [0, 0], [-60, 0], color="gray", linestyle="--"
            )[0],
            vline_right=self.ax_line_lateral.plot(
                [0, 0], [-60, 0], color="gray", linestyle="--"
            )[0],
        )

        self.save_fwhm_button = self.fig.add_button(
            "Save FWHM",
            x=self.margin_left + self.im_width + self.spacing,
            y=self.margin_top,
            width=self.button_width,
            height=0.5,
            color="black",
            hovercolor="#444444",
        )
        # Attach events
        self.save_fwhm_button.on_clicked(lambda x: self.save_fwhm())

        self.active_fwhm_marker = None

        plot_beamformed(self.ax_im, image_loaded.data, np.array(image_loaded.extent))

        # Attach events
        self.cid_click = self.fig.fig.canvas.mpl_connect(
            "button_release_event", self.on_click
        )
        self.cid_key = self.fig.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.update_plot()

    def on_click(self, event):

        if not event.inaxes == self.ax_im:
            return

        # Check if mouse click
        if event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            print("click")
            position = np.array([event.xdata, event.ydata])
            position = correct_fwhm_point(image_loaded, position, max_diff=1.0e-3)
            self.update_fwhm(position)

            self.arrow_target = EvalTool.TARGET_FWHM
        # elif event.button == 3:
        #     disk_pos = np.array([event.xdata, event.ydata])
        #     arrow_target = TARGET_GCNR

        else:
            return

        self.update_plot()

    def on_key(self, event):
        if event.key == "d":
            print("Adding FWHM measurement")
            self.active_fwhm_marker.deactivate()
            self.active_fwhm_marker = None
            return

    def update_fwhm(self, position):

        direction = position
        if self.vsource is None:
            direction -= self.vsource

        result, positions_axial = _sample_line(
            image=self.image.data,
            extent=self.image.extent,
            position=position,
            vec=direction,
            max_offset=self.max_offset,
            n_samples=self.n_samples,
        )
        offsets = np.linspace(-self.max_offset, self.max_offset, self.n_samples)
        self.curve_axial = FWHMCurve(positions_axial, result)

        vec_orth = np.array((-direction[1], direction[0]))
        result, positions_lateral = _sample_line(
            image=self.image.data,
            extent=self.image.extent,
            position=position,
            vec=vec_orth,
            max_offset=self.max_offset,
            n_samples=self.n_samples,
        )
        self.curve_lateral = FWHMCurve(positions_lateral, result)

        if self.active_fwhm_marker is None:
            self.active_fwhm_marker = FWHMMarker(
                ax=self.ax_im,
                position=position,
                direction=direction,
                max_offset=self.max_offset,
                n_samples=self.n_samples,
            )
        self.active_fwhm_marker.update(
            position, direction, positions_axial, positions_lateral
        )

        self.fwhm_lineplot_axial.line.set_ydata(self.curve_axial.values)
        self.fwhm_lineplot_lateral.line.set_ydata(self.curve_lateral.values)

        idx_peak, idx_left, idx_right = find_fwhm_indices(
            self.curve_axial.values, required_repeats=3, log_scale=self.image.in_db
        )

        self.fwhm_lineplot_axial.vline_left.set_xdata([offsets[idx_left]])
        self.fwhm_lineplot_axial.vline_right.set_xdata([offsets[idx_right]])
        self.fwhm_lineplot_axial.vline_peak.set_xdata([offsets[idx_peak]])

        idx_peak, idx_left, idx_right = find_fwhm_indices(
            self.curve_lateral.values, required_repeats=3, log_scale=self.image.in_db
        )

        self.fwhm_lineplot_lateral.vline_left.set_xdata([offsets[idx_left]])
        self.fwhm_lineplot_lateral.vline_right.set_xdata([offsets[idx_right]])
        self.fwhm_lineplot_lateral.vline_peak.set_xdata([offsets[idx_peak]])

    def save_fwhm(self):
        if self.active_fwhm_marker is None:
            return

        self.image = image_measure_fwhm(
            self.image,
            self.active_fwhm_marker.position,
            self.active_fwhm_marker.direction,
            max_offset=self.max_offset,
        )

        self.active_fwhm_marker.deactivate()
        self.active_fwhm_marker = None

    def update_plot(self):
        # self.active_fwhm_marker.update()
        # plt.draw()
        pass

        # ------------------------------------------------------------------------------
        # Plotting
        # ------------------------------------------------------------------------------
        # (line_axial,) = ax_line_axial.plot(
        #     np.linspace(-max_offset, max_offset, n_samples), np.zeros(n_samples)
        # )
        # (line_lateral,) = ax_line_lateral.plot(
        #     np.linspace(-max_offset, max_offset, n_samples),
        #     np.zeros(n_samples),
        #     "C1",
        # )
        # ax_line_axial.set_ylim(-60, 0)
        # ax_line_lateral.set_ylim(-60, 0)
        # mm_formatter_ax(ax_line_axial, x=True, y=False)
        # mm_formatter_ax(ax_line_lateral, x=True, y=False)

        # (vline_left_axial,) = ax_line_axial.plot(
        #     [0, 0], [-60, 0], color="gray", linestyle="--"
        # )
        # (vline_peak_axial,) = ax_line_axial.plot(
        #     [0, 0], [-60, 0], color="gray", linestyle="--"
        # )
        # (vline_right_axial,) = ax_line_axial.plot(
        #     [0, 0], [-60, 0], color="gray", linestyle="--"
        # )

        # (vline_left_lateral,) = ax_line_lateral.plot(
        #     [0, 0], [-60, 0], color="gray", linestyle="--"
        # )
        # (vline_peak_lateral,) = ax_line_lateral.plot(
        #     [0, 0], [-60, 0], color="gray", linestyle="--"
        # )
        # (vline_right_lateral,) = ax_line_lateral.plot(
        #     [0, 0], [-60, 0], color="gray", linestyle="--"
        # )

        # disk, annul0, annul1 = gcnr_plot_disk_annulus(
        #     ax_im,
        #     disk_center=disk_pos,
        #     disk_r=disk_radius,
        #     annulus_offset=1e-3,
        #     annulus_width=3e-3,
        #     opacity=1.0,
        # )


# disk_pos = np.array((0, 15)) * 1e-3
# disk_radius = 4e-3
# annulus_offset = 1e-3
# annulus_width = 3e-3

# scat_pos = np.array([4.062e-3, 7.286e-2])
# vsource = np.array((0, -15)) * 1e-3

# max_offset = 4e-3
# n_samples = 100


# def update_plot():
#     # global scat_pos, disk_pos, disk_radius, annulus_offset, annulus_width
#     vec = scat_pos - vsource
#     vec_orth = np.array((-vec[1], vec[0]))
#     result_axial, positions_axial = _sample_line(
#         image_loaded.data,
#         image_loaded.extent,
#         scat_pos,
#         vec,
#         max_offset,
#         n_samples,
#     )

#     result_lateral, positions_lateral = _sample_line(
#         image_loaded.data,
#         image_loaded.extent,
#         scat_pos,
#         vec_orth,
#         max_offset,
#         n_samples,
#     )
#     plot_beamformed(ax_im, image_loaded.data, np.array(image_loaded.extent))
#     # scat_indicator.set_data(scat_pos[0][None], scat_pos[1][None])
#     scat_line_axial_indicator.set_data(positions_axial[:, 0], positions_axial[:, 1])
#     scat_line_lateral_indicator.set_data(
#         positions_lateral[:, 0], positions_lateral[:, 1]
#     )
#     line_axial.set_ydata(result_axial)
#     line_lateral.set_ydata(result_lateral)

#     fwhm_axial = fwhm_image(
#         image=image_loaded, position=scat_pos, direction=vec, max_offset=max_offset
#     )
#     fwhm_lateral = fwhm_image(
#         image=image_loaded, position=scat_pos, direction=vec_orth, max_offset=max_offset
#     )
#     print(f"FWHM: {fwhm_axial*1e3:.1f} mm, {fwhm_lateral*1e3:.1f} mm")

#     idx_peak, idx_left, idx_right = find_fwhm_indices(
#         result_axial, required_repeats=3, log_scale=image_loaded.in_db
#     )
#     dist_vals = np.linspace(-max_offset, max_offset, n_samples)

#     vline_left_axial.set_xdata([dist_vals[idx_left]])
#     vline_right_axial.set_xdata([dist_vals[idx_right]])
#     vline_peak_axial.set_xdata([dist_vals[idx_peak]])

#     idx_peak, idx_left, idx_right = find_fwhm_indices(
#         result_lateral, required_repeats=3, log_scale=image_loaded.in_db
#     )
#     vline_left_lateral.set_xdata([dist_vals[idx_left]])
#     vline_right_lateral.set_xdata([dist_vals[idx_right]])
#     vline_peak_lateral.set_xdata([dist_vals[idx_peak]])

#     disk.center = disk_pos
#     annul0.center = disk_pos
#     annul1.center = disk_pos

#     disk.radius = disk_radius
#     annul0.radius = disk_radius + annulus_offset
#     annul1.radius = disk_radius + annulus_offset + annulus_width

#     gcnr_value = gcnr_disk_annulus(
#         image_loaded,
#         disk_center=disk_pos,
#         disk_r=disk_radius,
#         annulus_offset=annulus_offset,
#         annulus_width=annulus_width,
#     )
#     print(f"gCNR: {gcnr_value:.2f}")

#     plt.draw()


# def on_click(event):

#     global scat_pos, disk_pos, arrow_target
#     # Check if mouse click
#     if event.xdata is None or event.ydata is None:
#         return
#     if event.button == 1:
#         scat_pos = np.array([event.xdata, event.ydata])
#         scat_pos = correct_fwhm_point(image_loaded, scat_pos, max_diff=1.0e-3)
#         arrow_target = TARGET_FWHM
#     elif event.button == 3:
#         disk_pos = np.array([event.xdata, event.ydata])
#         arrow_target = TARGET_GCNR

#     else:
#         return

#     update_plot()


# def on_key(event):
#     global scat_pos, disk_pos, disk_radius, annulus_offset, annulus_width
#     step = 0.1e-3
#     delta = np.zeros(2)
#     if event.key == "right":
#         delta = np.array([1, 0]) * step
#     elif event.key == "left":
#         delta = np.array([-1, 0]) * step
#     elif event.key == "up":
#         delta = np.array([0, -1]) * step
#     elif event.key == "down":
#         delta = np.array([0, 1]) * step
#     elif event.key == "a":
#         log.info("Adding gCNR measurement")
#         image_measure_gcnr_disk_annulus(
#             image_loaded, disk_pos, disk_radius, annulus_offset, annulus_width
#         )
#         return
#     elif event.key == "d":
#         log.info("Adding FWHM measurement")
#         image_measure_fwhm(
#             image_loaded, scat_pos, scat_pos - vsource, max_offset=max_offset
#         )
#         return
#     elif event.key == "o":
#         annulus_offset = np.clip(annulus_offset - step, 2e-4, None)
#     elif event.key == "p":
#         annulus_offset = np.clip(annulus_offset + step, 2e-4, None)
#     elif event.key == "u":
#         annulus_width = np.clip(annulus_width - step, 2e-4, None)
#     elif event.key == "i":
#         annulus_width = np.clip(annulus_width + step, 2e-4, None)
#     else:
#         return

#     if arrow_target == TARGET_FWHM:
#         scat_pos += delta
#     elif arrow_target == TARGET_GCNR:
#         disk_pos += delta

#     update_plot()


# def on_scroll(event):
#     global disk_radius
#     step = 0.1e-3
#     if event.button == "up":
#         disk_radius += step
#     elif event.button == "down":
#         disk_radius -= step
#     else:
#         return

#     update_plot()


tool = EvalTool(image_loaded)

# ==============================================================================
# Initialize the figure
# ==============================================================================
# ------------------------------------------------------------------------------
# Parameters
# # ------------------------------------------------------------------------------
# fig_w, fig_h = 8, 11
# margin_left, margin_right = 1.0, 0.3
# margin = margin_left + margin_right
# spacing = 0.7
# grid_spacing = 0.2

# y_line = margin_right
# lineplot_h = 1

# # ------------------------------------------------------------------------------
# # Create figure and axes
# # ------------------------------------------------------------------------------
# fig = MPLFigure(figsize=(fig_w, fig_h))
# axes = fig.add_axes_grid(
#     n_rows=2,
#     n_cols=1,
#     x=margin_left,
#     y=margin_right,
#     width=fig_w - margin,
#     height=lineplot_h,
#     spacing=grid_spacing,
# )
# ax_line_axial, ax_line_lateral = axes[:, 0]
# remove_internal_ticks_labels(axes)

# ax_line_lateral.set_xlabel("Lateral distance [mm]")
# for ax in [ax_line_axial, ax_line_lateral]:
#     ax.set_ylabel("[dB]")
# ax_line_axial.set_title("Profile")

# ax_im = fig.add_ax(
#     x=margin_left,
#     y=margin_right + 2 * lineplot_h + spacing + grid_spacing,
#     width=fig_w - margin,
#     aspect=image_loaded.extent,
# )


# # ------------------------------------------------------------------------------
# # Plotting
# # ------------------------------------------------------------------------------
# (line_axial,) = ax_line_axial.plot(
#     np.linspace(-max_offset, max_offset, n_samples), np.zeros(n_samples)
# )
# (line_lateral,) = ax_line_lateral.plot(
#     np.linspace(-max_offset, max_offset, n_samples),
#     np.zeros(n_samples),
#     "C1",
# )
# ax_line_axial.set_ylim(-60, 0)
# ax_line_lateral.set_ylim(-60, 0)
# mm_formatter_ax(ax_line_axial, x=True, y=False)
# mm_formatter_ax(ax_line_lateral, x=True, y=False)


# (vline_left_axial,) = ax_line_axial.plot([0, 0], [-60, 0], color="gray", linestyle="--")
# (vline_peak_axial,) = ax_line_axial.plot([0, 0], [-60, 0], color="gray", linestyle="--")
# (vline_right_axial,) = ax_line_axial.plot(
#     [0, 0], [-60, 0], color="gray", linestyle="--"
# )

# (vline_left_lateral,) = ax_line_lateral.plot(
#     [0, 0], [-60, 0], color="gray", linestyle="--"
# )
# (vline_peak_lateral,) = ax_line_lateral.plot(
#     [0, 0], [-60, 0], color="gray", linestyle="--"
# )
# (vline_right_lateral,) = ax_line_lateral.plot(
#     [0, 0], [-60, 0], color="gray", linestyle="--"
# )

# disk, annul0, annul1 = gcnr_plot_disk_annulus(
#     ax_im,
#     disk_center=disk_pos,
#     disk_r=disk_radius,
#     annulus_offset=1e-3,
#     annulus_width=3e-3,
#     opacity=1.0,
# )


# plot_beamformed(ax_im, image_loaded.data, np.array(image_loaded.extent))

# (scat_line_axial_indicator,) = ax_im.plot(
#     np.zeros(n_samples), np.zeros(n_samples), "C0--", linewidth=0.5
# )
# (scat_line_lateral_indicator,) = ax_im.plot(
#     np.zeros(n_samples), np.zeros(n_samples), "C1--", linewidth=0.5
# )

# update_plot()


# cid = fig.fig.canvas.mpl_connect("button_release_event", on_click)
# cid_key = fig.fig.canvas.mpl_connect("key_press_event", on_key)
# cid_scroll = fig.fig.canvas.mpl_connect("scroll_event", on_scroll)


# plt.tight_layout()
plt.savefig("image.png", bbox_inches="tight")
plt.show()

image_loaded.save("image_frame_0000_measured.hdf5")
