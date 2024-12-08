import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jaxus import (
    beamform_hdf5,
    get_pixel_grid_from_extent,
    log_compress,
    plot_beamformed,
    save_hdf5_image,
)
from myplotlib import use_dark_style

parser = argparse.ArgumentParser()
parser.add_argument("path", type=Path)
parser.add_argument("--transmits", default=-1, type=int, nargs="*")
parser.add_argument("--frame", type=int, default=0, nargs="?")
parser.add_argument("--extent", type=int, default=[-20, 20, 1, 60], nargs="+")
parser.add_argument("--save-path", type=Path, default=None)
args = parser.parse_args()

print(args.extent)
path = Path(args.path)
spacing = 0.125e-3
extent = np.array(args.extent) * 1e-3
pixel_grid = get_pixel_grid_from_extent(extent=extent, pixel_size=spacing)

print(args.transmits)

image = beamform_hdf5(
    path, transmits=args.transmits, frames=args.frame, pixel_grid=pixel_grid
)[0][0]
image = log_compress(image, normalize=True)

use_dark_style()
fig, ax = plt.subplots()

plot_beamformed(ax, image, extent_m=pixel_grid.extent_m, vmin=-70)
if args.save_path is not None:
    path = Path(args.save_path)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    save_hdf5_image(path.with_suffix(".hdf5"), image, extent=pixel_grid.extent_m)
plt.show()
