"""Container for image data."""

import numpy as np
from pathlib import Path
import h5py
from jaxus.utils import log, log_compress, fix_extent
from skimage.exposure import match_histograms


SCALE_LINEAR = 0
SCALE_DB = 1


class Image:
    """Container for image data. Contains a 2D numpy array and metadata."""

    def __init__(self, data, extent, scale=SCALE_LINEAR, metadata=None):
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
        self.scale = scale

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
        if np.iscomplexobj(value):
            log.warning("Image data is complex. Taking magnitude.")
            value = np.abs(value)
        data = np.array(value, dtype=np.float32)
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
    def scale(self):
        """Return whether image data is in dB or linear."""
        return self._scale

    @scale.setter
    def scale(self, value):
        """Set whether image data is in dB or linear."""
        self._scale = Image._parse_scale(value)

    @property
    def in_db(self):
        """Return whether image data is log-compressed."""
        return self.scale == SCALE_DB

    def save(self, path):
        """Save image to HDF5 file."""
        save_hdf5_image(
            path=path,
            image=self.data,
            extent=self.extent,
            scale=self.scale,
            metadata=self.metadata,
        )

    @classmethod
    def load(cls, path):
        """Load image from HDF5 file."""
        return load_hdf5_image(path)

    def log_compress(self):
        """Log-compress image data."""
        if self.scale == SCALE_DB:
            log.warning("Image data is already log-compressed. Skipping.")
            return self

        self.data = log_compress(self.data)
        self.scale = SCALE_DB

        return self

    def normalize(self):
        """Normalize image data to max 1 when not log-compressed or 0 when log-compressed."""

        if self.scale == SCALE_DB:
            self.data -= self.data.max()
        else:
            self.data /= self.data.max()

        return self

    def __repr__(self):
        """Return string representation of Image object."""
        shape = self.shape
        log_compressed_str = ", in dB" if self.in_db() else ""
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

    @property
    def grid(self):
        """Return grid of image."""

        x_grid, y_grid = np.meshgrid(self.x_vals, self.y_vals, indexing="ij")
        return x_grid, y_grid

    @property
    def flatgrid(self):
        """Return flat grid of image."""
        return np.stack(self.grid, axis=-1).reshape(-1, 2)

    @property
    def x_vals(self):
        """Return x values of image."""
        return np.linspace(self.extent[0], self.extent[1], self.shape[0])

    @property
    def y_vals(self):
        """Return y values of image."""
        return np.linspace(self.extent[2], self.extent[3], self.shape[1])

    @property
    def aspect(self):
        """Return aspect ratio of image."""
        return (self.extent[1] - self.extent[0]) / (self.extent[3] - self.extent[2])

    def match_histogram(self, other):
        """Match the histogram of the image to another image."""

        self.data = match_histograms(self.data, other.data)

    @staticmethod
    def _parse_scale(val):
        """Parse scale value."""
        if val == SCALE_DB or val == SCALE_LINEAR:
            return val

        if isinstance(val, str):
            val = val.lower()
            if "lin" in val:
                return SCALE_LINEAR
            else:
                return SCALE_DB

        val = bool(val)
        return SCALE_DB if val else SCALE_LINEAR


def save_hdf5_image(path, image, extent, scale=SCALE_LINEAR, metadata=None):
    """
    Saves an image to an hdf5 file.

    Parameters
    ----------
    path : str
        The path to the hdf5 file.
    image : np.ndarray
        The image to save.
    extent : list
        The extent of the image (x0, x1, z0, z1).
    log_compressed : bool
        Whether the image is log compressed.
    metadata : dict
        Additional metadata to save.
    """

    extent = fix_extent(extent)

    path = Path(path)

    if path.exists():
        log.warning(f"Overwriting existing file {path}.")
        path.unlink()
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    if image.ndim > 2:
        image = np.squeeze(image)
        if image.ndim > 2:
            raise ValueError(
                f"Image must be 2D, but has shape {image.shape}. "
                f"Try using np.squeeze to remove extra dimensions."
            )

    with h5py.File(path, "w") as dataset:
        dataset.create_dataset("image", data=image)
        dataset["image"].attrs["extent"] = extent
        dataset["image"].attrs["scale"] = "linear" if scale == SCALE_LINEAR else "dB"
        if metadata is not None:
            save_dict_to_hdf5(dataset, metadata)


def load_hdf5_image(path):
    """
    Loads an image from an hdf5 file.

    Parameters
    ----------
    path : str
        The path to the hdf5 file.

    Returns
    -------
    image : np.ndarray
        The image.
    extent : np.ndarray
        The extent of the image (x0, x1, z0, z1).
    scale : int
        The scale of the image (SCALE_LINEAR or SCALE_DB).
    """

    with h5py.File(path, "r") as dataset:
        image = dataset["image"][()]
        extent = dataset["image"].attrs["extent"]
        scale = dataset["image"].attrs["scale"]
        metadata = load_hdf5_to_dict(dataset)
        metadata.pop("image", None)

    return Image(data=image, extent=extent, scale=scale, metadata=metadata)


def save_dict_to_hdf5(hdf5_file, data_dict, parent_group="/"):
    """
    Recursively saves a nested dictionary to an HDF5 file.

    Parameters
    ----------
    hdf5_file : h5py.File
        Opened h5py.File object.
    data_dict : dict
        (Nested) dictionary to save.
    parent_group : h5py.Group
        Current group path in HDF5 file (default is root "/").
    """
    data_dict = lists_to_numbered_dict(data_dict)
    for key, value in data_dict.items():
        group_path = f"{parent_group}/{key}"
        if isinstance(value, dict):
            # Create a new group for nested dictionary
            hdf5_file.require_group(group_path)
            save_dict_to_hdf5(hdf5_file, value, parent_group=group_path)
        else:
            # Convert leaf items into datasets
            hdf5_file[group_path] = value


def lists_to_numbered_dict(data_dict):
    """Transforms all lists in a dictionary to dictionaries with numbered keys."""
    for key, value in data_dict.items():
        if isinstance(value, list):
            data_dict[key] = {str(i): v for i, v in enumerate(value)}
        elif isinstance(value, dict):
            data_dict[key] = lists_to_numbered_dict(value)
    return data_dict


def numbered_dicts_to_list(data_dict):
    """Transforms all dictionaries with numbered keys to lists."""
    for key, value in data_dict.items():
        if isinstance(value, dict):
            if all(k.isdigit() for k in value.keys()):
                data_dict[key] = [value[k] for k in sorted(value.keys(), key=int)]
            else:
                data_dict[key] = numbered_dicts_to_list(value)
    return data_dict


def load_hdf5_to_dict(hdf5_file, parent_group="/"):
    """
    Recursively reads an HDF5 file into a nested dictionary.

    Args:
        hdf5_file: Opened h5py.File object.
        parent_group: Current group path in HDF5 file (default is root "/").

    Returns:
        Nested dictionary representing the HDF5 file structure.
    """
    data_dict = {}
    for key in hdf5_file[parent_group]:
        item_path = f"{parent_group}/{key}"
        if isinstance(hdf5_file[item_path], h5py.Group):
            data_dict[key] = load_hdf5_to_dict(hdf5_file, parent_group=item_path)
        else:
            data_dict[key] = hdf5_file[item_path][()]

    return numbered_dicts_to_list(data_dict)
