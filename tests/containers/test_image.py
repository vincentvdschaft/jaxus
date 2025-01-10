from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jaxus import load_hdf5_image, log_compress, save_hdf5_image, Image


# Fixture for sample data
@pytest.fixture
def sample_image_data():
    """Generate sample image data."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    extent = [0, 1, 0, 1]
    log_compressed = False
    metadata = {}
    return data, extent, log_compressed, metadata


def test_image_initialization(sample_image_data):
    data, extent, log_compressed, metadata = sample_image_data
    image = Image(data, extent, log_compressed, metadata)
    assert np.array_equal(image.data, data)
    assert np.array_equal(image.extent, extent)
    assert image.log_compressed == log_compressed


def test_image_shape_property(sample_image_data):
    data, extent, log_compressed, metadata = sample_image_data
    image = Image(data, extent, log_compressed, metadata)
    assert image.shape == data.shape


def test_image_data_setter_valid(sample_image_data):
    data, extent, log_compressed, metadata = sample_image_data
    image = Image(data, extent, log_compressed, metadata)
    new_data = [[5, 6], [7, 8]]
    image.data = new_data
    assert np.array_equal(image.data, np.array(new_data, dtype=float))


def test_image_data_setter_invalid(sample_image_data):
    data, extent, log_compressed, metadata = sample_image_data
    image = Image(data, extent, log_compressed, metadata)
    with pytest.raises(ValueError, match="Data must be 2D."):
        image.data = [1, 2, 3]  # Not 2D


def test_image_extent_setter_valid():
    data = [[1, 2], [3, 4]]
    extent = [0, 1, 0, 1]
    image = Image(data, extent, False)
    new_extent = [-1, 2, 3, -2]
    image.extent = new_extent
    assert np.array_equal(image.extent, [-1, 2, -2, 3])  # Sorted extent


def test_image_save(sample_image_data, tmp_path):
    data, extent, log_compressed, metadata = sample_image_data
    image = Image(data, extent, log_compressed, metadata)
    image.save(tmp_path / "test_image.hdf5")


def test_image_load(sample_image_data, tmp_path):
    data, extent, log_compressed, metadata = sample_image_data
    save_hdf5_image(
        path=tmp_path / "test_image.hdf5",
        image=data,
        extent=extent,
        log_compressed=log_compressed,
        metadata=metadata,
    )
    loaded_image = Image.load(tmp_path / "test_image.hdf5")
    assert np.array_equal(loaded_image.data, data)
    assert np.array_equal(loaded_image.extent, extent)
    assert loaded_image.log_compressed == log_compressed


def test_image_log_compress(sample_image_data):
    data, extent, log_compressed, metadata = sample_image_data
    image = Image(data, extent, log_compressed, metadata)
    image.log_compress()
    assert image.log_compressed


def test_image_normalize(sample_image_data):
    data, extent, log_compressed, metadata = sample_image_data
    image = Image(data, extent, log_compressed, metadata)
    image.normalize()
    assert np.allclose(image.data, data / np.max(data))


def test_image_normalize_log_compressed():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    extent = [0, 1, 0, 1]
    image = Image(data, extent, log_compressed=True)
    image.normalize()
    assert np.allclose(image.data, data - np.max(data))


def test_image_repr(sample_image_data):
    data, extent, log_compressed, metadata = sample_image_data
    image = Image(data, extent, log_compressed, metadata)
    expected_repr = "Image(((2, 2)), extent=[0 1 0 1])"
    assert repr(image) == expected_repr
