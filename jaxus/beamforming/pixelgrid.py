from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class PixelGrid:
    """Container class for a pixel grid of 2 or 3 dimensions."""

    def __init__(self, extent_m, pixel_positions_flat, shape):
        self._extent_m = np.array([float(val) for val in extent_m])
        self._pixel_positions_flat = pixel_positions_flat
        self._shape = np.array([int(val) for val in shape])

        # Sort the extent to ensure that x0 < x1, y0 < y1, z0 < z1, ...
        self._extent_m = sort_extent(self._extent_m)

        assert len(extent_m) == 2 * len(shape)
        assert pixel_positions_flat.shape[0] == np.prod(shape)

    def plot(
        self,
        output_path=None,
    ):
        pixel_positions = self.pixel_positions_flat
        extent_m = self.extent_m_2d
        num_points = pixel_positions.shape[0]

        if self.n_dims == 2:
            plt.figure(figsize=(7, 7))
            plt.imshow(
                np.zeros(self.shape_2d),
                extent=[extent_m[0], extent_m[1], extent_m[2], extent_m[3]],
                origin="lower",
                cmap="gray",
                alpha=0.1,
            )

            # Dynamically adjust the size of the scatter points based on the number of points
            base_size = 2000  # Base size to scale point size
            s = base_size / num_points  # Scale size inversely with the number of points

            plt.scatter(
                pixel_positions[:, 0],
                pixel_positions[:, 1],
                c="lightblue",
                s=s,  # Dynamically computed size
                alpha=0.75,
            )

            plt.title("2D Pixel Grid", fontsize=16)
            plt.xlabel("X (mm)", fontsize=12)
            plt.ylabel("Z (mm)", fontsize=12)

            plt.xlim(extent_m[0], extent_m[1])
            plt.ylim(extent_m[2], extent_m[3])

            plt.grid(True, color="white", linestyle="--", linewidth=0.5)
            plt.text(
                0.05,
                0.95,
                f"dz={self.dz}",
                transform=plt.gca().transAxes,
                fontsize=12,
                color="black",
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    edgecolor="gray",
                    facecolor="lightyellow",
                    alpha=0.75,
                ),
            )

            # Set default output path if not provided
            if output_path is None:
                output_path = Path(
                    f"./figures/pixelgrid_plot_{int(self._shape[0])}_{int(self._shape[1])}.png"
                )

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"✅ Pixel grid plotted and saved to {output_path}")
        else:
            raise NotImplementedError("PixelGrid plotting is only supported for 2D")

    def uniformly_sample_points_within_extent(self):
        points = np.empty((self.n_points, self.n_dims))

        for i in range(self.n_dims):
            min_val = self.extent_m[2 * i]
            max_val = self.extent_m[2 * i + 1]
            points[:, i] = np.random.uniform(min_val, max_val, self.n_points)

        return points

    @property
    def extent_m(self):
        """Returns the extent of the pixel grid in m.

        Returns
        -------
        extent_m : np.ndarray
            The extent of the pixel grid in m (x0, x1, z1, z0)
        """
        return self._extent_m

    @property
    def extent_m_zflipped(self):
        """Returns the extent of the pixel grid in m with the z-axis flipped."""

        extent_zflipped = self._extent_m.copy()
        extent_zflipped[-2], extent_zflipped[-1] = (
            extent_zflipped[-1],
            extent_zflipped[-2],
        )
        return extent_zflipped

    @property
    def extent_mm(self):
        """Returns the extent of the pixel grid in mm.

        Returns
        -------
        extent_mm : np.ndarray
            The extent of the pixel grid in mm (x0, x1, z1, z0)
        """
        return self._extent_m * 1e3

    @property
    def pixel_positions_flat(self):
        """Returns the pixel positions in a flat array of shape (n_points, n_dims)."""
        return self._pixel_positions_flat

    @property
    def pixel_positions(self):
        """Returns the pixel positions in a grid of shape (x, y) or (x, y, z)."""
        return self._pixel_positions_flat.reshape(self.shape_ndims)

    @property
    def shape(self):
        """Returns the shape of the pixel grid.

        Note: This does not include the number of dimensions that is present in the
        pixel_positions array.

        Returns
        -------
        shape : tuple
            The shape of the pixel grid (x, y) or (x, y, z).
        """
        return np.copy(self._shape)

    @property
    def shape_ndims(self):
        """Returns the shape of the pixel grid including the number of dimensions."""
        return np.concatenate([self._shape, [self._shape.shape[0]]])

    @property
    def n_dims(self):
        """Returns the number of dimensions of the pixel grid."""
        return len(self._shape)

    @property
    def n_points(self):
        """Returns the number of points in the pixel grid."""
        return np.prod(self._shape)

    @property
    def xlim(self):
        """Returns the x limits of the pixel grid."""
        return float(self._extent_m[0]), float(self._extent_m[1])

    @property
    def ylim(self):
        """Returns the y limits of the pixel grid."""
        if self.n_dims == 3:
            return float(self._extent_m[2]), float(self._extent_m[3])
        else:
            raise ValueError("The grid is 2D and does not have a y dimension.")

    @property
    def zlim(self):
        """Returns the z limits of the pixel grid."""
        if self.n_dims == 3:
            return float(self._extent_m[4]), float(self._extent_m[5])
        elif self.n_dims == 2:
            return float(self._extent_m[2]), float(self._extent_m[3])

    def get_2d_dims(self):
        """Returns the indices of the 2D dimensions in the pixel grid."""
        shape = self.shape
        indices = np.where(shape > 1)[0]

        if indices.size == 2:
            return indices
        elif indices.size > 2:
            raise ValueError(
                "Cannot determine the 2D dimensions of a grid with more than 2 "
                "dimensions of size > 1."
            )
        elif indices.size == 1:
            if indices[0] == 0:
                return np.array([indices[0], indices[0] + 1])
            else:
                return np.array([0, indices[0]])

    @property
    def shape_2d(self):
        """Returns the shape of the pixel grid in 2D."""
        shape = self.shape
        dims = self.get_2d_dims()
        return shape[dims]

    @property
    def extent_m_2d(self):
        """Returns the extent of the pixel grid in 2D."""
        extent = self.extent_m
        dims = self.get_2d_dims()
        return np.array(
            [
                extent[2 * dims[0]],
                extent[2 * dims[0] + 1],
                extent[2 * dims[1]],
                extent[2 * dims[1] + 1],
            ]
        )

    @property
    def extent_m_2d_zflipped(self):
        """Returns the extent of the pixel grid in mm with the z-axis flipped."""
        extent = self.extent_m_2d
        return np.array([extent[0], extent[1], extent[3], extent[2]])

    @property
    def dz(self):
        """Returns the spacing in the z-direction."""
        return (self.extent_m[-1] - self.extent_m[-2]) / self.shape[-1]

    @property
    def n_x(self):
        return self.shape[0]

    @property
    def n_z(self):
        return self.shape[-1]

    @property
    def n_y(self):
        return self.shape[1]


def get_flat_grid(shape, spacing, startpoints, center):
    """
    Returns a grid of coordinates of a given shape and spacing.
    The shape can be two or three dimensional.


    Parameters
    ----------
    shape : tuple
        The shape of the grid.
    spacing : tuple
        The spacing between grid points in each dimension.
    startpoints : tuple
        The coordinates of the first grid point.
    center : tuple
        Set to True to center the grid around the origin.

    Returns
    -------
    grid : ndarray
        The grid of coordinates of shape (n_points, n_dimensions).
    """
    assert len(shape) == len(spacing) == len(startpoints) == len(center)

    # Define the grid values along each dimension
    vals = [
        float(start) + np.arange(n_points) * float(delta)
        for start, n_points, delta in zip(startpoints, shape, spacing)
    ]

    # Center the grid values along the specified dimensions
    vals = [val - np.max(val) / 2 if cent else val for cent, val in zip(center, vals)]

    # Create the grid
    positions = np.meshgrid(*vals, indexing="ij")

    # Flatten the grid
    flat_grid = np.stack([pos.flatten() for pos in positions], axis=-1)

    return flat_grid


def get_pixel_grid(shape, spacing, startpoints, center):
    """Produces a PixelGrid object from the given parameters.

    Parameters
    ----------
    shape : tuple
        The shape of the grid.
    spacing : tuple
        The spacing between grid points in each dimension.
    startpoints : tuple
        The coordinates of the first grid point. If center is set to True for a
        dimension, these will be the midpoints of the grid.
    center : tuple
        Set to True to center the grid around the origin. This will change the
        startpoints into midpoints.
    """

    n_dim = len(shape)

    flat_grid = get_flat_grid(shape, spacing, startpoints, center)

    lim_min = flat_grid.min(axis=0)
    lim_max = flat_grid.max(axis=0)

    # 2D case
    if n_dim == 2:
        extent = (lim_min[0], lim_max[0], lim_min[1], lim_max[1])
    # 3D case and higher
    else:
        # Create array to store values
        extent = np.zeros(n_dim * 2)
        # Fill array with values in the order (x0, x1, y0, y1, z0, z1, ...)
        extent[::2] = lim_min
        extent[1::2] = lim_max
        # Turn into tuple
        extent = tuple(extent)

    return PixelGrid(extent, flat_grid, shape)


def get_pixel_grid_from_extent(extent, pixel_size):
    """Produces a PixelGrid object from the given extent and pixel size.

    Parameters
    ----------
    extent : tuple, list, or ndarray
        The extent of the grid as a tuple, list, or ndarray. [x0, x1, z0, z1]
    pixel_size : tuple or float
        The size of the pixels in each dimension.

    Returns
    -------
    grid : PixelGrid
        The pixel grid object.
    """

    extent = np.array(extent)

    n_dims = len(extent) // 2

    if isinstance(pixel_size, (float, int)):
        pixel_size = np.ones(n_dims) * pixel_size

    extent_0 = extent[::2]
    extent_1 = extent[1::2]
    extent_stacked = np.stack([extent_0, extent_1], axis=0)
    extent_0 = np.min(extent_stacked, axis=0)
    extent_1 = np.max(extent_stacked, axis=0)

    sizes = extent_1 - extent_0

    shape = tuple(int(np.round(size / pixel)) for size, pixel in zip(sizes, pixel_size))

    return get_pixel_grid(shape, pixel_size, extent_0, center=(False,) * n_dims)


def get_pixel_grid_from_lims(lims, pixel_size, prioritize_limits=False):
    """Produces a PixelGrid object from the given limits and pixel size.

    Note that it is generally not possible to satisfy both the limits and the pixel
    size exactly. If `prioritize_limits` is set to False the limits will be adjusted
    to fit the pixel size. If `prioritize_limits` is set to True the pixel size will
    be adjusted to fit the limits.

    Parameters
    ----------
    lims : tuple
        The limits of the grid as a tuple of tuples. The first tuple is the
        limits along the first dimension, the second tuple is the limits along
        the second dimension, and so on.
    pixel_size : tuple or float
        The size of the pixels in each dimension.
    prioritize_limits : bool, default=False
        Set to True to prioritize using the exact the limits over the exact pixel
        size.

    Returns
    -------
    grid : PixelGrid
        The pixel grid object.
    """
    assert isinstance(lims, (tuple, list)), "lims must be a tuple or a list."

    # Expand pixel_size if it is a single value
    if isinstance(pixel_size, (float, int)):
        pixel_size = (pixel_size,) * len(lims)

    for lim in lims:
        assert isinstance(lim, (tuple, list)), "Each limit must be a tuple or a list."
        assert len(lim) == 2, "Each limit must have two values."

    # Order the limits
    lims = [sorted(lim) for lim in lims]

    # Get the shape of the grid
    shape = tuple(
        int(np.round((lim[1] - lim[0]) / size)) for lim, size in zip(lims, pixel_size)
    )
    shape = tuple(s if s > 0 else 1 for s in shape)

    # Determine the values of the grid
    if prioritize_limits:
        # Create arrays of values
        vals = [
            np.linspace(lim[0], lim[1], n_points) for lim, n_points in zip(lims, shape)
        ]
    else:
        vals = [
            np.arange(n_points) * size + lim[0]
            for lim, n_points, size in zip(lims, shape, pixel_size)
        ]

    # Create the grid
    positions = np.meshgrid(*vals, indexing="ij")

    positions = np.stack(positions, axis=-1)

    n_dims = len(shape)

    # Flatten the grid
    flat_grid = np.reshape(positions, (-1, n_dims))

    # Create the extent
    extent_m = np.zeros(2 * n_dims)
    extent_m[::2] = np.min(flat_grid, axis=0)
    extent_m[1::2] = np.max(flat_grid, axis=0)

    return PixelGrid(extent_m, flat_grid, shape)


def sort_extent(extent):
    """Sorts the extent such that every first value is less than the second value."""

    extent = np.array(extent)

    extent = extent.reshape((-1, 2))
    np.sort(extent, axis=1)
    extent = extent.flatten()
    return extent
