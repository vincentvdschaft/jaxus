import matplotlib.pyplot as plt
import numpy as np

from jaxus import gcnr, gcnr_compute_disk, gcnr_plot_disk_annulus


def test_gcnr():
    region1 = np.random.randn(100, 100)
    region2 = np.random.randn(100, 100)
    bins = 256

    gcnr_value = gcnr(region1=region1, region2=region2, bins=bins)
    assert 0 <= gcnr_value <= 1, "GCNR value should be between 0 and 1."


def test_gcnr_compute_disk():
    image = np.random.randn(100, 100)
    xlims_m = (-0.1, 0.1)
    zlims_m = (-0.1, 0.1)
    disk_pos_m = (0, 0)
    inner_radius_m = 0.01
    outer_radius_start_m = 0.02
    outer_radius_end_m = 0.03
    num_bins = 256

    gcnr_value = gcnr_compute_disk(
        image=image,
        xlims_m=xlims_m,
        zlims_m=zlims_m,
        disk_pos_m=disk_pos_m,
        inner_radius_m=inner_radius_m,
        outer_radius_start_m=outer_radius_start_m,
        outer_radius_end_m=outer_radius_end_m,
        num_bins=num_bins,
    )


def test_gcnr_plot_disk_annulus():
    fig, ax = plt.subplots()
    pos_m = (0, 0)
    inner_radius_m = 0.01
    outer_radius_start_m = 0.02
    outer_radius_end_m = 0.03
    opacity = 0.5

    gcnr_plot_disk_annulus(
        ax=ax,
        pos_m=pos_m,
        inner_radius_m=inner_radius_m,
        outer_radius_start_m=outer_radius_start_m,
        outer_radius_end_m=outer_radius_end_m,
        opacity=opacity,
    )
    plt.close(fig)
