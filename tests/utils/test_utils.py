from jaxus.utils import *
from jaxus.testing import *
import matplotlib.pyplot as plt


def test_get_vsource():
    probe_geometry = get_test_probe_geometry()

    t0_delays_focus = t0_delays_from_vsource(
        probe_geometry, vsource_angle=10 * np.pi / 180, vsource_depth=20e-3, c=1540
    )
    t0_delays_diverging = t0_delays_from_vsource(
        probe_geometry, vsource_angle=10 * np.pi / 180, vsource_depth=-20e-3, c=1540
    )
    t0_delays_plane = t0_delays_from_vsource(
        probe_geometry, vsource_angle=60 * np.pi / 180, vsource_depth=np.inf, c=1540
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
