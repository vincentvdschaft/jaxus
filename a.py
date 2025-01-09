from jaxus import (
    Image,
    plot_beamformed,
    image_measure_gcnr_disk_annulus,
    gcnr_plot_disk_annulus,
)
import numpy as np
import matplotlib.pyplot as plt
from myplotlib import *

use_style(STYLE_DARK)

extent = [
    -20,
    20,
    0,
    30,
]

disk_pos = (0, 15)
disk_radius = 7

x_vals = np.linspace(extent[0], extent[1], 400)
z_vals = np.linspace(extent[2], extent[3], 300)
X, Z = np.meshgrid(x_vals, z_vals, indexing="ij")
R = np.sqrt((X - disk_pos[0]) ** 2 + (Z - disk_pos[1]) ** 2)
data = np.where(R < disk_radius, 1.0, 0.0)

image = Image(data=data, extent=extent, log_compressed=False, metadata={})

image_measure_gcnr_disk_annulus(
    image=image,
    disk_center=(0, 18),
    disk_radius=disk_radius,
    annulus_radius0=disk_radius + 1,
    annulus_radius1=disk_radius + 3,
)
image.save("test_image.hdf5")

image_loaded = Image.load("test_image.hdf5")


print(image_loaded.metadata)
fig, ax = plt.subplots()
plot_beamformed(ax, image_loaded.data, np.array(image_loaded.extent), vmin=0, vmax=1)
gcnr_plot_disk_annulus(
    ax,
    disk_center=(0, 18),
    disk_r=disk_radius - 1,
    annul_r0=disk_radius + 1,
    annul_r1=disk_radius + 3,
)

plt.tight_layout()
plt.savefig("image.png", bbox_inches="tight")
