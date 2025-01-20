from jaxus.utils import *
from jaxus.testing import *
import matplotlib.pyplot as plt
import pytest


@pytest.mark.parametrize(
    "pos, angle",
    [
        ([0, 10e-3], 0.0),
        ([0, -10e-3], 0.0),
        ([10e-3, 10e-3], np.pi / 4),
        ([10e-3, -10e-3], -np.pi / 4),
    ],
)
def test_vsource_angle(pos, angle):
    vsource_pos = np.array(pos)
    computed_angle = vsource_angle(vsource_pos)
    assert np.isclose(computed_angle, angle, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize(
    "angle, depth",
    [
        (0.0, 0.0),
        (0.0, 10e-3),
        (np.pi / 4, 10e-3),
        (-np.pi / 4, 10e-3),
    ],
)
def test_t0_delays(angle, depth):
    n_el = 80
    probe_geometry = get_test_probe_geometry(n_el=n_el)

    t0_delays = t0_delays_from_vsource(
        probe_geometry, vsource_angle=angle, vsource_depth=depth, sound_speed=1540
    )

    assert t0_delays.shape == (n_el,)
    assert np.all(t0_delays >= 0)
    assert np.min(t0_delays) == 0.0


def test_get_vsource():
    probe_geometry = get_test_probe_geometry()

    t0_delays_focus = t0_delays_from_vsource(
        probe_geometry,
        vsource_angle=10 * np.pi / 180,
        vsource_depth=20e-3,
        sound_speed=1540,
    )
    t0_delays_diverging = t0_delays_from_vsource(
        probe_geometry,
        vsource_angle=10 * np.pi / 180,
        vsource_depth=-20e-3,
        sound_speed=1540,
    )
    t0_delays_plane = t0_delays_from_vsource(
        probe_geometry,
        vsource_angle=60 * np.pi / 180,
        vsource_depth=np.inf,
        sound_speed=1540,
    )

    _, ax = plt.subplots()
    ax.plot(t0_delays_focus * 1e6, label="Focus")
    ax.plot(t0_delays_diverging * 1e6, label="Diverging")
    ax.plot(t0_delays_plane * 1e6, label="Plane")
    ax.set_xlabel("Element index")
    ax.set_ylabel("T0 delays [us]")
    ax.legend()
    plt.show()


def test_deduce():
    probe_geometry = get_test_probe_geometry()

    angle, depth = 10 * np.pi / 180, 20e-3
    vsource_pos_true = vsource_pos(angle, depth)

    t0_delays = t0_delays_from_vsource(
        probe_geometry, vsource_angle=angle, vsource_depth=depth, sound_speed=1540
    )
    vsource = deduce_vsource(probe_geometry, t0_delays, sound_speed=1540)

    assert np.allclose(vsource, vsource_pos_true, rtol=1e-2, atol=1e-3)

    print(vsource)
