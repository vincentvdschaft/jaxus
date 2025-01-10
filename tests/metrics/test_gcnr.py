import matplotlib.pyplot as plt
import numpy as np

from jaxus import gcnr, gcnr_plot_disk_annulus, gcnr_disk_annulus
from jaxus.containers.image import Image


def test_gcnr():
    region1 = np.random.randn(100, 100)
    region2 = np.random.randn(100, 100)
    bins = 256

    gcnr_value = gcnr(region1=region1, region2=region2, bins=bins)
    assert 0 <= gcnr_value <= 1, "GCNR value should be between 0 and 1."


def test_gcnr_compute_disk():
    extent = np.array([-20, 20, -20, 20]) * 1e-3
    image = Image(data=np.random.randn(100, 100), extent=extent, log_compressed=False)

    gcnr_disk_annulus(
        image=image,
        disk_center=(0, 0),
        disk_r=10e-3,
        annulus_offset=1e-3,
        annulus_width=2e-3,
    )


def test_gcnr_plot_disk_annulus():
    fig, ax = plt.subplots()
    pos_m = (0, 0)
    disk_r = 10e-3
    annulus_offset = 1e-3
    annulus_width = 2e-3
    opacity = 0.5

    gcnr_plot_disk_annulus(
        ax=ax,
        disk_center=pos_m,
        disk_r=disk_r,
        annulus_offset=annulus_offset,
        annulus_width=annulus_width,
        opacity=opacity,
    )

    plt.close(fig)
