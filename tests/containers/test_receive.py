import numpy as np
import pytest

from jaxus.containers.receive import Receive


def test_correct_initialization():
    """Tests that the receive is correctly initialized without errors."""

    sampling_frequency = 40e6
    n_ax = 128
    initial_time = 1e-5

    receive = Receive(sampling_frequency, n_ax, initial_time)

    # Check that all values ended up in the correct place
    assert receive.sampling_frequency == sampling_frequency
    assert receive.n_ax == n_ax


@pytest.mark.parametrize(
    "sampling_frequency, n_ax, initial_time",
    [
        (np.ones((6,), dtype=np.int32), 128, None),
        (40e6, "string", None),
        (None, 128, None),
        (None, None, None),
        (40e6, 128, "string"),
    ],
)
def test_wrong_datatypes(sampling_frequency, n_ax, initial_time):
    """Tests that the receive is correctly initialized without errors."""

    with pytest.raises(TypeError):
        Receive(sampling_frequency, n_ax, initial_time)


@pytest.mark.parametrize(
    "sampling_frequency, n_ax, initial_time",
    [
        (-40e6, 128, 1e-5),
        (40e6, -128, 1e-5),
        (40e6, 128, -1e-5),
        (-40e6, -128, -1e-5),
    ],
)
def test_negative_values(sampling_frequency, n_ax, initial_time):
    """Tests that the receive is correctly initialized without errors."""

    with pytest.raises(ValueError):
        Receive(sampling_frequency, n_ax, initial_time)
