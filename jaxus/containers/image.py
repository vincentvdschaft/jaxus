"""Container for image data."""

import numpy as np

from jaxus.data import load_hdf5_image, save_hdf5_image
from jaxus.utils import log, log_compress, fix_extent


class Image:
    """Container for image data. Contains a 2D numpy array and metadata."""

    def __init__(self, data, extent, log_compressed, metadata=None):
        """Initialilze Image object.

        Parameters
        ----------
        data : array_like
            2D numpy array containing image data (n_x, n_y).
        extent : array_like
            4-element array containing the extent of the image.
        log_compressed : bool
            Whether the image data is log-compressed.
        """
        self.data = data
        self.extent = extent
        self.log_compressed = log_compressed
        self._metadata = {}
        if metadata is not None:
            self.update_metadata(metadata)

    @property
    def shape(self):
        """Return shape of image data."""
        return self.data.shape

    @property
    def data(self):
        """Return image data."""
        return self._data

    @data.setter
    def data(self, value):
        """Set image data."""
        data = np.array(value, dtype=float)
        if data.ndim != 2:
            raise ValueError("Data must be 2D.")

        self._data = data

    @property
    def extent(self):
        """Return extent of image."""
        return self._extent

    @extent.setter
    def extent(self, value):
        """Set extent of image."""
        extent = np.array([value[0], value[1], value[2], value[3]])
        self._extent = fix_extent(extent)

    @property
    def log_compressed(self):
        """Return whether image data is log-compressed."""
        return self._log_compressed

    @log_compressed.setter
    def log_compressed(self, value):
        """Set whether image data is log-compressed."""
        self._log_compressed = bool(value)

    def save(self, path):
        """Save image to HDF5 file."""
        save_hdf5_image(
            path=path,
            image=self.data,
            extent=self.extent,
            log_compressed=self.log_compressed,
            metadata=self.metadata,
        )

    @classmethod
    def load(cls, path):
        """Load image from HDF5 file."""
        data, extent, log_compressed, metadata = load_hdf5_image(path)
        return cls(data, extent, log_compressed, metadata)

    def log_compress(self):
        """Log-compress image data."""
        if self.log_compressed:
            log.warning("Image data is already log-compressed. Skipping.")
            return

        self.data = log_compress(self.data)
        self.log_compressed = True

    def normalize(self):
        """Normalize image data to max 1 when not log-compressed or 0 when log-compressed."""

        if self.log_compressed:
            self.data -= self.data.max()
        else:
            self.data /= self.data.max()

    def __repr__(self):
        """Return string representation of Image object."""
        shape = self.shape
        log_compressed_str = ", log-compressed" if self.log_compressed else ""
        return (
            f"Image(({shape[0], shape[1]}), extent={self.extent}{log_compressed_str})"
        )

    @property
    def metadata(self):
        """Return metadata of image."""
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        """Set metadata of image."""
        assert isinstance(value, dict), "Metadata must be a dictionary."
        self._metadata = value

    def add_metadata(self, key, value):
        """Add metadata to image."""
        self.metadata[key] = value

    def update_metadata(self, metadata):
        """Update metadata of image."""
        self.metadata.update(metadata)

    def append_metadata(self, key, value):
        """Add metadata assuming the key is a list."""

        if key not in self.metadata:
            self.metadata[key] = []
        elif not isinstance(self.metadata[key], list):
            raise ValueError(f"Metadata key {key} is not a list.")

        self.metadata[key].append(value)

    def clear_metadata(self):
        """Clear metadata of image."""
        self.metadata = {}
