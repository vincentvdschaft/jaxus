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
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path, default="image_frame_0000.hdf5", nargs="?")
parser.add_argument("metrics_file", type=Path, default=None, nargs="?")
parser.add_argument("--clear-metadata", action="store_true")
args = parser.parse_args()

path = Path(args.file)
image_loaded = Image.load(path)
image_loaded_metrics = Image.load(args.metrics_file) if args.metrics_file else None
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


def _replace_none(value, replacement):
    if value is None:
        return replacement
    return value


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

    def as_dict(self):
        return {
            "position": [float(self.position[0]), float(self.position[1])],
            "direction": [float(self.direction[0]), float(self.direction[1])],
            "max_offset": float(self.max_offset),
            "n_samples": int(self.n_samples),
        }

    @classmethod
    def from_dict(cls, data, ax):
        return cls(ax, **data)

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

    def update(
        self,
        position=None,
        direction=None,
        positions_axial=None,
        positions_lateral=None,
    ):
        position = _replace_none(position, self.position)
        direction = _replace_none(direction, self.direction)
        positions_axial = _replace_none(positions_axial, self.positions_axial)
        positions_lateral = _replace_none(positions_lateral, self.positions_lateral)

        self.position = position
        self.direction = direction
        self.positions_axial = positions_axial
        self.positions_lateral = positions_lateral
        self.draw()

    def deactivate(self):
        self.active = False
        self.draw()


class GCNRDiskAnnulusMarker:
    def __init__(
        self, ax, disk_pos, disk_radius, annulus_offset, annulus_width, active=True
    ):
        self.ax = ax
        self.disk_pos = disk_pos
        self.disk_radius = disk_radius
        self.annulus_offset = annulus_offset
        self.annulus_width = annulus_width
        self.active = active

        self.disk, self.annul0, self.annul1 = gcnr_plot_disk_annulus(
            ax,
            disk_center=disk_pos,
            disk_radius=disk_radius,
            annulus_offset=annulus_offset,
            annulus_width=annulus_width,
            opacity=1.0,
        )
        self.update()

    def as_dict(self):
        return {
            "disk_pos": [float(self.disk_pos[0]), float(self.disk_pos[1])],
            "disk_radius": float(self.disk_radius),
            "annulus_offset": float(self.annulus_offset),
            "annulus_width": float(self.annulus_width),
        }

    @classmethod
    def from_dict(cls, data, ax):
        return cls(ax, **data)

    def update(
        self, disk_pos=None, disk_radius=None, annulus_offset=None, annulus_width=None
    ):

        disk_pos = _replace_none(disk_pos, self.disk_pos)
        disk_radius = _replace_none(disk_radius, self.disk_radius)
        annulus_offset = _replace_none(annulus_offset, self.annulus_offset)
        annulus_width = _replace_none(annulus_width, self.annulus_width)

        self.disk_pos = disk_pos
        self.disk_radius = disk_radius
        self.annulus_offset = annulus_offset
        self.annulus_width = annulus_width

        self.disk.center = disk_pos
        self.annul0.center = disk_pos
        self.annul1.center = disk_pos

        self.disk.radius = disk_radius
        self.annul0.radius = disk_radius + annulus_offset
        self.annul1.radius = disk_radius + annulus_offset + annulus_width

        if self.active:
            self.disk.set_edgecolor("C0")
            self.annul0.set_edgecolor("C1")
            self.annul1.set_edgecolor("C1")
        else:
            self.disk.set_edgecolor("white")
            self.annul0.set_edgecolor("white")
            self.annul1.set_edgecolor("white")

        plt.draw()

    def deactivate(self):
        self.active = False
        self.update()


class IncrementWidget:
    def __init__(
        self,
        fig,
        x,
        y,
        width,
        height,
        value,
        label,
        step=0.1e-3,
        minval=0.0,
        maxval=1.0,
        callback=None,
    ):
        self.fig = fig
        self.step = step
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.button_height = height / 2
        self.button_y = y + height / 2
        self.value = value
        self.minval = minval
        self.maxval = maxval
        self.callback = callback

        self.button_width = self.width / 4
        self.button_decrement = fig.add_button(
            x=x + self.button_width,
            y=self.button_y,
            width=self.button_width,
            height=self.button_height,
            label="<",
            color="black",
            hovercolor="#444444",
        )
        self.button_decrement_large = fig.add_button(
            x=x,
            y=self.button_y,
            width=self.button_width,
            height=self.button_height,
            label="<<",
            color="black",
            hovercolor="#444444",
        )
        self.button_increment = fig.add_button(
            x=x + self.width - 2 * self.button_width,
            y=self.button_y,
            width=self.button_width,
            height=self.button_height,
            label=">",
            color="black",
            hovercolor="#444444",
        )
        self.button_increment_large = fig.add_button(
            x=x + self.width - self.button_width,
            y=self.button_y,
            width=self.button_width,
            height=self.button_height,
            label=">>",
            color="black",
            hovercolor="#444444",
        )
        self.button_decrement.on_clicked(self.decrement)
        self.button_increment.on_clicked(self.increment)
        self.button_decrement_large.on_clicked(self.decrement_large)
        self.button_increment_large.on_clicked(self.increment_large)

        self.label = self.fig.add_text(
            x=x + self.width / 2,
            y=y,
            text=label,
            va="top",
            ha="center",
        )

    def increment(self, event):
        self.value = min(self.value + self.step, self.maxval)
        if self.callback is not None:
            self.callback(self.value)

    def decrement(self, event):
        self.value = max(self.value - self.step, self.minval)
        if self.callback is not None:
            self.callback(self.value)

    def increment_large(self, event):
        self.value = min(self.value + self.step * 10, self.maxval)
        if self.callback is not None:
            self.callback(self.value)

    def decrement_large(self, event):
        self.value = max(self.value - self.step * 10, self.minval)
        if self.callback is not None:
            self.callback(self.value)


class EvalTool:
    TARGET_FWHM, TARGET_GCNR = 0, 1

    def __init__(self, image, metrics_image=None):
        self.image = image
        self.spacing = 0.7
        self.button_spacing = 0.1
        self.margin_left = 1.0
        self.margin_right = self.spacing
        self.margin_top = 0.3
        self.margin_bottom = 0.6
        self.im_width = 8
        self.margin = 0.3
        self.grid_spacing = 0.2
        self.button_width = 2
        self.button_height = 0.5
        self.arrow_target = EvalTool.TARGET_FWHM

        self.active_fwhm_marker = None
        self.active_gcnr_marker = None
        self.frozen_fwhm_markers = []
        self.frozen_gcnr_markers = []

        self.y_line = self.margin_top
        self.lineplot_h = 1

        self.fig_w = (
            self.margin_left
            + self.im_width
            + self.spacing
            + self.button_width
            + self.margin_right
        )
        im_height = self.im_width / image.aspect
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

        self.freeze_fwhm_button = self.fig.add_button(
            "Freeze FWHM",
            x=self.margin_left + self.im_width + self.spacing,
            y=self.margin_top,
            width=self.button_width,
            height=0.5,
            color="black",
            hovercolor="#444444",
        )
        # Attach events
        self.freeze_fwhm_button.on_clicked(lambda x: self.freeze_fwhm())

        self.freeze_gcnr_button = self.fig.add_button(
            "Freeze gCNR",
            x=self.margin_left + self.im_width + self.spacing,
            y=self.margin_top + self.button_height + self.button_spacing,
            width=self.button_width,
            height=self.button_height,
            color="black",
            hovercolor="#444444",
        )
        self.freeze_gcnr_button.on_clicked(lambda x: self.freeze_gcnr())

        self.save_image_button = self.fig.add_button(
            "Save Image",
            x=self.margin_left + self.im_width + self.spacing,
            y=self.margin_top + 2 * self.button_height + 2 * self.button_spacing,
            width=self.button_width,
            height=self.button_height,
            color="black",
            hovercolor="#444444",
        )
        self.save_image_button.on_clicked(lambda x: self.save_image())

        plot_beamformed(self.ax_im, image_loaded.data, np.array(image_loaded.extent))

        # Attach events
        self.cid_click = self.fig.fig.canvas.mpl_connect(
            "button_release_event", self.on_click
        )
        self.cid_key = self.fig.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.radius_widget = IncrementWidget(
            fig=self.fig,
            x=self.margin_left + self.im_width + self.spacing,
            y=self.margin_top + 3 * self.button_height + 3 * self.button_spacing,
            width=self.button_width,
            height=self.button_height,
            value=5e-3,
            step=0.1e-3,
            minval=0.0,
            maxval=100e-3,
            label="Disk radius",
            callback=lambda x: self.update_gcnr(disk_radius=x),
        )

        self.offset_widget = IncrementWidget(
            fig=self.fig,
            x=self.margin_left + self.im_width + self.spacing,
            y=self.margin_top + 4 * self.button_height + 4 * self.button_spacing,
            width=self.button_width,
            height=self.button_height,
            value=1e-3,
            step=0.1e-3,
            minval=0.0,
            maxval=100e-3,
            label="Annulus offset",
            callback=lambda x: self.update_gcnr(annulus_offset=x),
        )

        self.width_widget = IncrementWidget(
            fig=self.fig,
            x=self.margin_left + self.im_width + self.spacing,
            y=self.margin_top + 5 * self.button_height + 5 * self.button_spacing,
            width=self.button_width,
            height=self.button_height,
            value=1e-3,
            step=0.1e-3,
            minval=0.0,
            maxval=100e-3,
            label="Annulus width",
            callback=lambda x: self.update_gcnr(annulus_width=x),
        )

        self.vsource_textbox = self.fig.add_textbox(
            x=self.margin_left + self.im_width + self.spacing,
            y=self.margin_top + 6 * self.button_height + 6 * self.button_spacing,
            width=self.button_width,
            height=self.button_height,
            label="Vsource",
            color="black",
            hovercolor="#444444",
        )
        self.vsource_textbox.on_submit(self.update_vsource)

        self.save_to_yaml_button = self.fig.add_button(
            "Save Metrics",
            x=self.margin_left + self.im_width + self.spacing,
            y=self.margin_top + 7 * self.button_height + 7 * self.button_spacing,
            width=self.button_width,
            height=self.button_height,
            color="black",
            hovercolor="#444444",
        )

        self.save_to_yaml_button.on_clicked(lambda event: self.save_metrics_to_yaml())

        if metrics_image is not None:
            self.load_measurements_from_image(metrics_image)
        else:
            self.load_measurements_from_image(image)
        plt.show()

    def save_metrics_to_yaml(self):
        yaml_dict = {
            "fwhm": [fwhm_marker.as_dict() for fwhm_marker in self.frozen_fwhm_markers],
            "gcnr": [gcnr_marker.as_dict() for gcnr_marker in self.frozen_gcnr_markers],
        }
        with open("metrics.yaml", "w") as f:
            yaml.dump(yaml_dict, f)

    def update_vsource(self, text):
        if "none" in text.lower():
            self.vsource = None
            return

        text = text.replace(" ", ",")
        text_parts = text.split(",")
        if len(text_parts) != 2:
            return
        x, y = text_parts
        self.vsource = np.array([float(x), float(y)]) * 1e-3
        if self.active_fwhm_marker is not None:
            self.update_fwhm(position=self.active_fwhm_marker.position)

        print(f"Vsource: {self.vsource}")

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
        elif event.button == 3:
            disk_pos = np.array([event.xdata, event.ydata])
            self.arrow_target = EvalTool.TARGET_GCNR
            self.update_gcnr(disk_pos)

        else:
            return

    def on_key(self, event):
        vector = _arrow_key_to_vector(event.key)
        if vector is None:
            return

        if (
            self.arrow_target == EvalTool.TARGET_FWHM
            and self.active_fwhm_marker is not None
        ):
            self.active_fwhm_marker.position += vector
            self.update_fwhm(self.active_fwhm_marker.position)

        elif (
            self.arrow_target == EvalTool.TARGET_GCNR
            and self.active_gcnr_marker is not None
        ):
            self.active_gcnr_marker.disk_pos += vector
            self.update_gcnr(self.active_gcnr_marker.disk_pos)

    def update_fwhm(self, position):

        direction = np.copy(position)
        if self.vsource is not None:
            direction -= self.vsource

        print(self.vsource)

        print(f"Position: {position}")

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

        plt.draw()

    def update_gcnr(
        self, disk_pos=None, disk_radius=None, annulus_offset=None, annulus_width=None
    ):
        if self.active_gcnr_marker is None:
            self.active_gcnr_marker = GCNRDiskAnnulusMarker(
                ax=self.ax_im,
                disk_pos=_replace_none(disk_pos, np.array([0, 15e-3])),
                disk_radius=_replace_none(disk_radius, 4e-3),
                annulus_offset=_replace_none(annulus_offset, 1e-3),
                annulus_width=_replace_none(annulus_width, 1e-3),
            )

        self.active_gcnr_marker.update(
            disk_pos=disk_pos,
            disk_radius=disk_radius,
            annulus_offset=annulus_offset,
            annulus_width=annulus_width,
        )

    def freeze_fwhm(self):
        if self.active_fwhm_marker is None:
            return

        measured_image = image_measure_fwhm(
            self.image,
            self.active_fwhm_marker.position,
            self.active_fwhm_marker.direction,
            max_offset=self.max_offset,
            return_copy=True,
        )

        fwhm_axial = measured_image.metadata["fwhm"][-1]["fwhm_value_axial"]
        fwhm_lateral = measured_image.metadata["fwhm"][-1]["fwhm_value_lateral"]

        print(f"FWHM axial: {fwhm_axial*1e-3:.2f}mm")
        print(f"FWHM lateral: {fwhm_lateral*1e-3:.2f}mm")

        self.active_fwhm_marker.deactivate()
        self.frozen_fwhm_markers.append(self.active_fwhm_marker)
        self.active_fwhm_marker = None

    def freeze_gcnr(self):
        if self.active_gcnr_marker is None:
            return

        measured_image = image_measure_gcnr_disk_annulus(
            self.image,
            self.active_gcnr_marker.disk_pos,
            self.active_gcnr_marker.disk_radius,
            self.active_gcnr_marker.annulus_offset,
            self.active_gcnr_marker.annulus_width,
            return_copy=True,
        )

        gcnr = measured_image.metadata["gcnr"][-1]["gcnr_value"]
        print(f"gCNR: {gcnr}")

        print(
            f"disk_center={self.active_gcnr_marker.disk_pos}, disk_radius={self.active_gcnr_marker.disk_radius}, annulus_offset={self.active_gcnr_marker.annulus_offset}, annulus_width={self.active_gcnr_marker.annulus_width}"
        )

        self.active_gcnr_marker.deactivate()
        self.frozen_gcnr_markers.append(self.active_gcnr_marker)
        self.active_gcnr_marker = None

    def save_image(self):
        """Performs all measurements and saves the image."""
        if self.active_fwhm_marker is not None:
            self.freeze_fwhm()

        if self.active_gcnr_marker is not None:
            self.freeze_gcnr()

        for n, fwhm_marker in enumerate(self.frozen_fwhm_markers):
            print(f"Measuring FWHM {n}")
            self.image = image_measure_fwhm(
                image=self.image,
                position=fwhm_marker.position,
                axial_direction=fwhm_marker.direction,
                max_offset=self.max_offset,
                correct_position=False,
            )

        for n, gcnr_marker in enumerate(self.frozen_gcnr_markers):
            print(f"Measuring gCNR {n}")
            self.image = image_measure_gcnr_disk_annulus(
                image=self.image,
                disk_center=gcnr_marker.disk_pos,
                disk_radius=gcnr_marker.disk_radius,
                annulus_offset=gcnr_marker.annulus_offset,
                annulus_width=gcnr_marker.annulus_width,
            )

        self.image.save("image_frame_0000_measured.hdf5")

    def load_measurements_from_image(self, image):
        self.image = image
        metadata = image.metadata

        if "fwhm" in metadata:
            for fwhm_data in metadata["fwhm"]:
                print("Loading FWHM")
                position = fwhm_data["position"]
                direction = fwhm_data["axial_direction"]
                self.active_fwhm_marker = FWHMMarker(
                    self.ax_im,
                    position,
                    direction,
                    max_offset=self.max_offset,
                    n_samples=self.n_samples,
                    active=True,
                )
                self.update_fwhm(position)
                self.freeze_fwhm()

        if "gcnr" in metadata:
            for gcnr_data in metadata["gcnr"]:
                print("Loading gCNR")
                disk_pos = gcnr_data["disk_center"]
                disk_radius = gcnr_data["disk_radius"]
                annulus_offset = gcnr_data["annulus_offset"]
                annulus_width = gcnr_data["annulus_width"]
                self.frozen_gcnr_markers.append(
                    GCNRDiskAnnulusMarker(
                        self.ax_im,
                        disk_pos,
                        disk_radius,
                        annulus_offset,
                        annulus_width,
                        active=False,
                    )
                )

        plt.draw()


if args.clear_metadata:
    image_loaded.clear_metadata()

tool = EvalTool(image_loaded, image_loaded_metrics)
