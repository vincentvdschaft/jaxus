import numpy as np
import pytest

from jaxus.containers.medium import Medium


def test_correct_initialization():
    """Tests that the medium is correctly initialized without errors."""

    # Define the scatterer positions
    scatterer_positions = np.ones((128, 2), dtype=np.float32)
    # Define the scatterer amplitudes
    scatterer_amplitudes = np.ones((128,), dtype=np.float32)
    sound_speed = 1480

    medium = Medium(scatterer_positions, scatterer_amplitudes, sound_speed)

    # Check that all values ended up in the correct place
    assert medium.sound_speed == sound_speed
    assert np.array_equal(
        medium.scatterer_positions, scatterer_positions.astype(np.float32)
    )
    assert np.array_equal(
        medium.scatterer_amplitudes, scatterer_amplitudes.astype(np.float32)
    )


@pytest.mark.parametrize(
    "scatterer_positions, scatterer_amplitudes, sound_speed",
    [
        (np.ones((2, 6), dtype=np.int32), np.ones((6,), dtype=np.int32), 1480),
        (np.ones((128, 2)), np.ones((128,)), "string"),
        (None, np.ones((128,)), 1480),
    ],
)
def test_wrong_datatypes(scatterer_positions, scatterer_amplitudes, sound_speed):
    """Tests that the medium is correctly initialized without errors."""

    with pytest.raises(TypeError):
        Medium(scatterer_positions, scatterer_amplitudes, sound_speed)


@pytest.mark.parametrize(
    "scatterer_positions, scatterer_amplitudes, sound_speed",
    [
        [np.ones((128, 2, 2)), np.ones((128,)), 1480],
        [np.ones((128, 2)), np.ones((128, 2)), 1480],
        [np.ones((128, 2, 2)), np.ones((128, 2)), 1480],
        [np.ones(()), np.ones((128,)), 1480],
    ],
)
def test_wrong_shape(scatterer_positions, scatterer_amplitudes, sound_speed):
    """Tests that the medium is correctly initialized without errors."""

    with pytest.raises(ValueError):
        Medium(scatterer_positions, scatterer_amplitudes, sound_speed)
