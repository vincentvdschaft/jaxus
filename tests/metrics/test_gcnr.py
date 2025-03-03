import matplotlib.pyplot as plt
import numpy as np
from pytest import fixture
from imagelib import Image

from jaxus import gcnr, gcnr_plot_disk_annulus, gcnr_disk_annulus


@fixture
def fixture_circle_center():
    """Fixture to return the center of a circle."""
    return np.array((-5e-3, -25e-3))


@fixture
def fixture_circle_radius():
    """Fixture to return the radius of a circle."""
    return 2e-3


@fixture
def fixture_circle_image(fixture_circle_center, fixture_circle_radius):
    """Fixture to return an image of a white circle on a black background."""
    image = Image(data=np.zeros((256, 128)), extent=(-10e-3, 10e-3, -30e-3, -20e-3))
    grid = image.grid
    x_grid, z_grid = grid[:, :, 0], grid[:, :, 1]
    data = image.data
    data[
        (x_grid - fixture_circle_center[0]) ** 2
        + (z_grid - fixture_circle_center[1]) ** 2
        < fixture_circle_radius**2
    ] = 1
    image.data = data
    return image


def test_gcnr():
    """Tests if the gcnr function runs and returns a value between 0 and 1."""
    region1 = np.random.randn(100, 100)
    region2 = np.random.randn(100, 100)
    bins = 256

    gcnr_value = gcnr(region1=region1, region2=region2, bins=bins)
    assert 0 <= gcnr_value <= 1, "GCNR value should be between 0 and 1."


def test_gcnr_compute_disk(
    fixture_circle_image, fixture_circle_center, fixture_circle_radius
):
    """Tests if the gcnr_disk_annulus function works correctly."""

    gcnr_value_inside_disk = gcnr_disk_annulus(
        image=fixture_circle_image,
        disk_center=fixture_circle_center,
        disk_radius=fixture_circle_radius,
        annulus_offset=0.2e-3,
        annulus_width=2e-3,
    )
    gcnr_value_outside_disk = gcnr_disk_annulus(
        image=fixture_circle_image,
        disk_center=fixture_circle_center + np.array((10e-3, 0)),
        disk_radius=fixture_circle_radius,
        annulus_offset=0.2e-3,
        annulus_width=2e-3,
    )
    assert (
        gcnr_value_inside_disk == 1.0
    ), "GCNR value should be 1.0 for perfect overlap with circle."
    assert gcnr_value_outside_disk == 0.0, "GCNR value should be 0.0 for no overlap."


def test_gcnr_plot_disk_annulus():
    """Tests if the gcnr_plot_disk_annulus function runs without errors."""
    fig, ax = plt.subplots()
    pos_m = (0, 0)
    disk_radius = 10e-3
    annulus_offset = 1e-3
    annulus_width = 2e-3
    opacity = 0.5

    gcnr_plot_disk_annulus(
        ax=ax,
        disk_center=pos_m,
        disk_radius=disk_radius,
        annulus_offset=annulus_offset,
        annulus_width=annulus_width,
        opacity=opacity,
    )

    plt.close(fig)
