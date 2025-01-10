from jaxus.metrics import *
from jaxus.containers import Image
from jaxus import plot_beamformed
import matplotlib.pyplot as plt
from myplotlib import *

image = Image.load("image_frame_0000.hdf5")
scat_pos = np.array([-0.0255, 0.0425])

new_pos = correct_fwhm_point(image, scat_pos, max_diff=1.5e-3)


fig, ax = plt.subplots()
plot_beamformed(ax, image.data, image.extent)
ax.plot(*scat_pos, "rx", markersize=1.2)
ax.plot(*new_pos, "bx", markersize=1.2)
plt.show()
