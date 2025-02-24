"""Container for image data."""

from copy import deepcopy
from pathlib import Path
from PIL import Image as PILImage
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

from jaxus import log
from jaxus.utils import log_compress

SCALE_LINEAR = 0
SCALE_DB = 1


class Image:
    """Container for image data. Contains a 2D numpy array and metadata."""

    def __init__(self, data, extent, scale=SCALE_LINEAR, polar=False, metadata=None):
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
        self.extent = Extent(extent)
        self.scale = scale
        self.polar = polar

        self._metadata = {}
        if metadata is not None:
            self.update_metadata(metadata)

    def imshow(self, ax, *args, **kwargs):
        """Display image using matplotlib imshow."""
        extent = self.extent_imshow
        return ax.imshow(self.data.T, extent=extent, origin="lower", *args, **kwargs)

    @property
    def shape(self):
        """Return shape of image data."""
        return self.data.shape

    @property
    def data(self):
        """Return image data."""
        return np.copy(self._data)

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
    def pixel_w(self):
        """The width of a pixel in the image."""
        return self.extent.width / (self.shape[0] - 1)

    @property
    def pixel_h(self):
        """The height of a pixel in the image."""
        return self.extent.height / (self.shape[1] - 1)

    @property
    def pixel_size(self):
        """The width and height of a pixel in the image."""
        return self.pixel_w, self.pixel_h

    @property
    def extent(self):
        """Return extent of image."""
        return self._extent

    @extent.setter
    def extent(self, value):
        """Set extent of image."""
        self._extent = Extent(value).sort()

    @property
    def scale(self):
        """Return whether image data is in dB or linear."""
        return self._scale

    @scale.setter
    def scale(self, value):
        """Set whether image data is in dB or linear."""
        self._scale = Image._parse_scale(value)

    def set_scale(self, value):
        """Returns a copy of the image with a new scale."""
        return Image(self.data, self.extent, scale=value, metadata=self.metadata)

    def set_data(self, data):
        """Returns a copy of the image with new data."""
        return Image(data, self.extent, scale=self.scale, metadata=self.metadata)

    def set_extent(self, extent):
        """Returns a copy of the image with new extent."""
        return Image(self.data, extent, scale=self.scale, metadata=self.metadata)

    def set_metadata(self, metadata):
        """Returns a copy of the image with new metadata."""
        return Image(self.data, self.extent, scale=self.scale, metadata=metadata)

    def __getitem__(self, idx):
        assert isinstance(idx, tuple) and len(idx) == 2
        # Index with integers
        if isinstance(idx[0], int) and isinstance(idx[1], int):
            return self.data[idx]

        # Index with slices
        if isinstance(idx[0], slice) and isinstance(idx[1], slice):
            slice0, slice1 = idx[0], idx[1]
            slice0 = slice(
                slice0.start if slice0.start is not None else 0,
                slice0.stop if slice0.stop is not None else self.shape[0],
                slice0.step,
            )
            slice1 = slice(
                slice1.start if slice1.start is not None else 0,
                slice1.stop if slice1.stop is not None else self.shape[1],
                slice1.step,
            )

            data = self.data[idx]
            extent = [
                self.extent[0] + slice0.start * self.extent.width / self.shape[0],
                self.extent[0] + slice0.stop * self.extent.width / self.shape[0],
                self.extent[2] + slice1.start * self.extent.height / self.shape[1],
                self.extent[2] + slice1.stop * self.extent.height / self.shape[1],
            ]
            return Image(data, extent=extent, scale=self.scale, metadata=self.metadata)

        # Index with data coordinates
        if isinstance(idx[0], float) and isinstance(idx[1], float):
            x_idx = int((idx[0] - self.extent[0]) / self.extent.width * self.shape[0])
            y_idx = int((idx[1] - self.extent[2]) / self.extent.height * self.shape[1])
            return self[x_idx, y_idx]

        raise ValueError(
            "Invalid index. Must be integers, slices, or data coordinates."
        )

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
        return self

    @staticmethod
    def load(path):
        """Load image from HDF5 file."""
        return load_hdf5_image(path)

    @staticmethod
    def from_png(path):
        """Load image from PNG file."""

        # Load grayscale image
        pil_image = PILImage.open(path).convert("L")
        data = np.array(pil_image).T
        extent = [0, pil_image.width, 0, pil_image.height]
        return Image(data, extent=extent)

    def log_compress(self):
        """Log-compress image data."""
        if self.scale == SCALE_DB:
            log.warning("Image data is already log-compressed. Skipping.")
            return self

        data = log_compress(self.data)
        scale = SCALE_DB

        return Image(data, extent=self.extent, scale=scale, metadata=self.metadata)

    def normalize(self, normval=None):
        """Normalize image data to max 1 when not log-compressed or 0 when log-compressed."""

        if normval is None:
            normval = self.data.max()

        data = self.data
        if self.scale == SCALE_DB:
            data -= normval
        else:
            data /= normval

        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

    def normalize_percentile(self, percentile=99):
        """Normalize image data to the given percentile value."""
        normval = np.percentile(self.data, percentile)
        return self.normalize(normval)

    def __repr__(self):
        """Return string representation of Image object."""
        shape = self.shape
        log_compressed_str = ", in dB" if self.in_db else ""
        return (
            f"Image(({shape[0], shape[1]}), extent={self.extent}{log_compressed_str})"
        )

    @property
    def metadata(self):
        """Return metadata of image."""
        return deepcopy(self._metadata)

    @metadata.setter
    def metadata(self, value):
        """Set metadata of image."""
        assert isinstance(value, dict), "Metadata must be a dictionary."
        self._metadata = value

    def add_metadata(self, key, value):
        """Add metadata to image."""
        self._metadata[key] = value
        return self

    def update_metadata(self, metadata):
        """Update metadata of image."""
        self._metadata.update(metadata)
        return self

    def append_metadata(self, key, value):
        """Add metadata assuming the key is a list."""

        if key not in self.metadata:
            self._metadata[key] = []
        elif not isinstance(self.metadata[key], list):
            raise ValueError(f"Metadata key {key} is not a list.")

        self._metadata[key].append(value)
        return self

    def clear_metadata(self):
        """Clear metadata of image."""
        self.metadata = {}

    @property
    def x_vals(self):
        """Return x values of image."""
        return np.linspace(self.extent[0], self.extent[1], self.shape[0])

    @property
    def y_vals(self):
        """Return y values of image."""
        return np.linspace(self.extent[2], self.extent[3], self.shape[1])

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
    def aspect(self):
        """Return aspect ratio of image."""
        return self.extent.aspect

    def match_histogram(self, other):
        """Match the histogram of the image to another image."""

        data = match_histograms(self.data, other.data)
        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

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

    def max(self):
        """Find the maximum value in the image data."""
        return np.max(self.data)

    def min(self):
        """Find the minimum value in the image data."""
        return np.min(self.data)

    def clip(self, minval=None, maxval=None):
        """Clip the image data to a range."""
        data = np.clip(self.data, minval, maxval)
        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

    def apply_fn(self, fn):
        """Apply a function to the image data."""
        self.data = fn(self.data)
        return self

    def map_range(self, minval, maxval, old_min=None, old_max=None):
        """Map the image data to a new range."""
        if old_min is None:
            old_min = self.min()
        if old_max is None:
            old_max = self.max()

        data = (self.data - old_min) / (old_max - old_min) * (maxval - minval) + minval
        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

    def to_pixels(self):
        """Converts the image to linear pixel values [0, 1]."""
        return self.map_range(0.0, 1.0).set_scale(SCALE_LINEAR)

    def __add__(self, other):
        """Add two images together."""
        if isinstance(other, (int, float, np.number)):
            data = self.data + other
            return Image(
                data, extent=self.extent, scale=self.scale, metadata=self.metadata
            )

        if isinstance(other, Image):
            assert all([e1 == e2 for e1, e2 in zip(self.extent, other.extent)])
            assert self.scale == other.scale
            data = self.data + other.data
            return Image(
                data, extent=self.extent, scale=self.scale, metadata=self.metadata
            )

        other = np.array(other)
        data = self.data + other
        return Image(data, extent=self.extent, scale=self.scale, metadata=self.metadata)

        return TypeError(f"Unsupported type {type(other)} for addition.")

    def __mul__(self, other):
        if isinstance(other, (int, float, np.number)):
            data = self.data * other
            return Image(
                data, extent=self.extent, scale=self.scale, metadata=self.metadata
            )

    def __sub__(self, other):
        """Subtract two images."""
        return self + (other * -1)

    def resample(self, shape, extent=None):
        """Resample image to a new shape."""

        if extent is None:
            extent = self.extent
        else:
            extent = Extent(extent)

        interpolator = RegularGridInterpolator(
            (self.x_vals, self.y_vals),
            self.data,
            bounds_error=False,
            fill_value=0 if self.scale == SCALE_LINEAR else -240,
        )
        new_xvals = np.linspace(extent[0], extent[1], shape[0])
        new_yvals = np.linspace(extent[2], extent[3], shape[1])

        x_grid, y_grid = np.meshgrid(new_xvals, new_yvals, indexing="ij")
        new_data = interpolator((x_grid, y_grid))

        return Image(
            new_data,
            extent=(new_xvals[0], new_xvals[-1], new_yvals[0], new_yvals[-1]),
            scale=self.scale,
            metadata=self.metadata,
        )

    def transpose(self):
        """Transpose image data."""
        data = self.data.T
        extent = [self.extent[2], self.extent[3], self.extent[0], self.extent[1]]
        return Image(data, extent=extent, scale=self.scale, metadata=self.metadata)

    def xflip(self):
        """Flip image data along x-axis."""
        data = np.flip(self.data, axis=0)
        extent = self.extent.xflip()
        return Image(data, extent=extent, scale=self.scale, metadata=self.metadata)

    def yflip(self):
        """Flip image data along y-axis."""
        data = np.flip(self.data, axis=1)
        extent = self.extent.yflip()
        return Image(data, extent=extent, scale=self.scale, metadata=self.metadata)

    @property
    def extent_imshow(self):
        """Returns an extent corrected for how imshow works. It ensures that the
        gridpoints represent the center of the pixels."""
        return _correct_imshow_extent(self.extent, self.shape)


def _correct_imshow_extent(extent, shape):
    """Corrects the extent of an image to ensure the min and max
    extent are in the centers of the outer pixels.

    Parameters
    ----------
    extent :  Extent
        The extent of the image (x0, x1, y0, y1).
    shape : tuple of int
        The shape of the image (width, height).

    Returns
    -------
    extent : tuple of float
        The corrected extent of the image.
    """
    extent = Extent(extent).sort()
    width, height = extent.size
    pixel_w = width / (shape[0] - 1)
    pixel_h = height / (shape[1] - 1)

    offset = (
        -pixel_w / 2,
        pixel_w / 2,
        -pixel_h / 2,
        pixel_h / 2,
    )
    return Extent([ext + off for ext, off in zip(extent, offset)])


class ImageSequence:
    """Container class to hold a sequence of images."""

    def __init__(self, images):
        self.images = images

    @property
    def images(self):
        """Get the list of images."""
        return self._images

    @property
    def data(self):
        """Get the data cube of all images."""
        return np.stack([im.data for im in self.images], axis=0)

    @images.setter
    def images(self, value):
        assert all(isinstance(obj, Image) for obj in value)
        self._images = list(value)

    @property
    def extent(self):
        """Get the extent."""
        return self.images[0].extent

    @property
    def scale(self):
        """Get the scale (SCALE_LINEAR or SCALE_DB)."""
        return self.images[0].scale

    def __getitem__(self, idx):
        """Get image from list."""
        if isinstance(idx, slice):
            return ImageSequence(self.images[idx])

        return self.images[idx]

    def append(self, im):
        """Append image to list."""
        assert isinstance(im, Image)
        self.images.append(im)

    def save(self, directory, name):
        # Remove file extension if it exists
        name = str(Path(name).with_suffix(""))
        for n, im in enumerate(self.images):
            im.save(Path(directory) / f"{name}_{str(n).zfill(5)}.hdf5")

        return self

    @staticmethod
    def load(paths):
        """Load image sequence from disk.

        Parameters
        ----------
        paths : Path or [Path]
            The path of a directory to load all images from or a list of specific paths.

        """
        if isinstance(paths, (Path, str)):
            paths = Path(paths)
            paths = list(paths.glob("*.hdf5"))
            paths.sort()
            paths = [Path(p) for p in paths]

        assert isinstance(paths, list)

        images = []
        for path in paths:
            images.append(Image.load(path))

        return ImageSequence(images)

    @staticmethod
    def from_numpy(data, extent, scale=SCALE_LINEAR):
        assert data.ndim == 3
        n_frames = data.shape[0]
        images = []
        for n in range(n_frames):
            images.append(Image(data[n], extent=extent, scale=scale))

        return ImageSequence(images)

    def log_compress(self):
        return ImageSequence([im.log_compress() for im in self.images])

    def normalize(self, normval=None):
        if normval is None:
            normval = self.max()
        return ImageSequence([im.normalize(normval) for im in self.images])

    def normalize_percentile(self, percentile=99):
        return ImageSequence(
            [im.normalize_percentile(percentile) for im in self.images]
        )

    def match_histogram(self, other):
        return ImageSequence([im.match_histogram(other) for im in self.images])

    def clip(self, minval=None, maxval=None):
        return ImageSequence([im.clip(minval, maxval) for im in self.images])

    def max(self):
        return max([im.max() for im in self.images])

    def min(self):
        return min([im.min() for im in self.images])

    def transpose(self):
        return ImageSequence([im.transpose() for im in self.images])

    def xflip(self):
        return ImageSequence([im.xflip() for im in self.images])

    def yflip(self):
        return ImageSequence([im.yflip() for im in self.images])

    def __iter__(self):
        return iter(self.images)

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"ImageSequence({len(self.images)} images)"

    def map(self, func):
        """Apply a function to each image in the sequence."""
        list(map(lambda im: Image.apply_fn(im, func), self.images))
        return self


class Extent(tuple):
    """Wrapper class for extent data."""

    def __new__(cls, *args, **kwargs):
        if kwargs:
            initializer = [kwargs["x0"], kwargs["x1"], kwargs["y0"], kwargs["y1"]]
        elif len(args) == 4:
            initializer = [
                float(args[0]),
                float(args[1]),
                float(args[2]),
                float(args[3]),
            ]
        # Reduce numpy arrays and such to list of floats
        elif len(args) == 1:
            initializer = [float(x) for x in args[0]]
            assert len(initializer) == 4
        else:
            raise ValueError("Extent must have 4 elements.")

        return super(Extent, cls).__new__(cls, initializer)

    def yflip(self):
        return Extent(self[0], self[1], self[3], self[2])

    def xflip(self):
        return Extent(self[1], self[0], self[2], self[3])

    def ycenter(self):

        offset = (self[3] - self[2]) / 2
        return Extent(self[0], self[1], self[2] - offset, self[3] - offset)

    def xcenter(self):

        offset = (self[1] - self[0]) / 2
        return Extent(self[0] - offset, self[1] - offset, self[2], self[3])

    def sort(self):
        x0 = min(self[0], self[1])
        x1 = max(self[0], self[1])
        y1 = max(self[2], self[3])
        y0 = min(self[2], self[3])
        return Extent(x0, x1, y0, y1)

    @property
    def x0(self):
        return self[0]

    @property
    def x1(self):
        return self[1]

    @property
    def y0(self):
        return self[2]

    @property
    def y1(self):
        return self[3]

    @property
    def width(self):
        self_sorted = self.sort()
        return self_sorted[1] - self_sorted[0]

    @property
    def height(self):
        self_sorted = self.sort()
        return self_sorted[3] - self_sorted[2]

    @property
    def size(self):
        return self.width, self.height

    @property
    def aspect(self):
        """Returns the aspect ratio (width/height)."""
        return self.width / self.height

    @property
    def xlims(self):
        return self[0], self[1]

    @property
    def xlims_flipped(self):
        return self[1], self[0]

    @property
    def ylims(self):
        return self[2], self[3]

    @property
    def ylims_flipped(self):
        return self[3], self[2]

    def __mul__(self, value):
        return Extent(
            self[0] * value, self[1] * value, self[2] * value, self[3] * value
        )

    def __add__(self, value):
        return Extent(
            self[0] + value, self[1] + value, self[2] + value, self[3] + value
        )

    def __sub__(self, value):
        return self + (-value)

    def __hash__(self):
        return hash(tuple(self))

    def setx0(self, value):
        return Extent(value, self[1], self[2], self[3])

    def setx1(self, value):
        return Extent(self[0], value, self[2], self[3])

    def sety0(self, value):
        return Extent(self[0], self[1], value, self[3])

    def sety1(self, value):
        return Extent(self[0], self[1], self[2], value)


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

    extent = Extent(extent).sort()

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
            data_dict[key] = {str(i).zfill(3): v for i, v in enumerate(value)}
        elif isinstance(value, dict):
            data_dict[key] = lists_to_numbered_dict(value)
    return data_dict


def _is_numbered_dict(data_dict):
    keys = data_dict.keys()
    try:
        keys = [int(k) for k in keys]
    except ValueError:
        return False
    return set(keys) == set(range(len(keys)))


def numbered_dicts_to_list(data_dict):
    """Transforms all dictionaries with numbered keys to lists."""
    for key, value in data_dict.items():
        if isinstance(value, dict):
            if _is_numbered_dict(value):
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


def sample_points_from_image(image: Image, num_samples: int) -> np.ndarray:
    """
    Samples points from a space defined by an extent, using the grayscale image
    as the probability density function.

    Parameters
    ----------
    image : Image
        Grayscale image to sample points from.
    num_samples : int
        Number of points to sample.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples, 2) containing the sampled points.

    Raises
    ------
    ValueError
        If the input image is not a 2D grayscale image.
    """
    image = image.map_range(0, 1)
    extent = image.extent
    image = image.data
    pixel_width = (extent[1] - extent[0]) / (image.shape[0] - 1)
    pixel_height = (extent[3] - extent[2]) / (image.shape[1] - 1)

    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    x0, x1, y0, y1 = extent

    # Normalize the image to use it as a probability density function
    pdf = image.astype(np.float64)
    pdf /= pdf.sum()

    # Flatten the PDF for easier sampling
    flat_pdf = np.ravel(pdf)

    # Generate the indices of the samples based on the PDF
    sampled_indices = np.random.choice(len(flat_pdf), size=num_samples, p=flat_pdf)

    # Convert flat indices back to 2D indices
    x_indices, y_indices = np.unravel_index(sampled_indices, shape=image.shape)

    # Map indices to coordinates in the given extent
    x_coords = x0 + (x_indices / (image.shape[0] - 1)) * (x1 - x0)
    y_coords = y0 + (y_indices / (image.shape[1] - 1)) * (y1 - y0)

    offset = np.random.uniform(-0.5, 0.5, size=(num_samples, 2)) * np.array(
        [[pixel_width, pixel_height]]
    )

    # Combine x and y coordinates
    sampled_points = np.stack((x_coords, y_coords), axis=-1) + offset

    return sampled_points
