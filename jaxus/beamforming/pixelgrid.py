import numpy as np


class PixelGrid:
    def __init__(self, pixel_positions_cartesian: np.ndarray):
        """Initializes a PixelGrid object.

        Parameters
        ----------
        pixel_positions_cartesian : np.ndarray
            The positions of the pixels in meters of shape `(n_rows, n_cols, 2)`.
        """

        self.pixel_positions = pixel_positions_cartesian

    @property
    def pixel_positions(self):
        """The positions of all pixels in the beamforming grid in meters of shape
        (n_row, n_col, 2)."""
        return np.copy(self._pixel_positions)

    @pixel_positions.setter
    def pixel_positions(self, value):
        """Sets the pixel positions. The pixel positions must be a 3D array with the
        first dimension being the rows, the second dimension being the columns, and
        the third dimension being the x and z positions."""
        if not isinstance(value, np.ndarray):
            raise TypeError("pixel_positions must be a numpy array.")
        if value.ndim != 3:
            raise ValueError("pixel_positions must be a 3D array.")
        if value.shape[2] != 2:
            raise ValueError("pixel_positions must have size (n_rows, n_cols, 2).")
        self._pixel_positions = value.astype(np.float32)

    @property
    def pixel_positions_flat(self):
        """The positions of all pixels in the beamforming grid in meters of shape
        (n_pixels, 2)."""
        return np.reshape(self.pixel_positions, (-1, 2))

    @property
    def n_pixels(self):
        """The number of pixels in the beamforming grid."""
        return self.pixel_positions.shape[0] * self.pixel_positions.shape[1]

    @property
    def n_cols(self):
        """The number of columns in the pixel grid."""
        return self.pixel_positions.shape[1]

    @property
    def n_rows(self):
        """The number of rows in the pixel grid."""
        return self.pixel_positions.shape[0]

    @property
    def collim(self):
        """The col-axis limits of the pixel grid. in meters. For a cartesian
        grid these values are in meters. For a polar grid these are in radians. For a
        polar grid these are the extreme values at furthest left and right.
        """
        return (self.pixel_positions[0, 0, 0], self.pixel_positions[0, -1, 0])

    @property
    def rowlim(self):
        """The row-axis limits of the pixel grid in meters. For a polar grid
        these are the extreme values at the center"""
        return (self.pixel_positions[0, 0, 1], self.pixel_positions[-1, 0, 1])

    @property
    def xlim(self):
        """The smallest and largest x-position in the grid in meters."""
        xmin = np.min(self.pixel_positions[:, :, 0])
        xmax = np.max(self.pixel_positions[:, :, 0])
        return (xmin, xmax)

    @property
    def zlim(self):
        """The smallest and largest z-position in the grid in meters."""
        zmin = np.min(self.pixel_positions[:, :, 1])
        zmax = np.max(self.pixel_positions[:, :, 1])
        return (zmin, zmax)

    @property
    def extent(self):
        """The extent of the beamformed image. (xmin, xmax, zmax, zmin) in meters."""
        return (
            np.min(self.pixel_positions[:, :, 0]),
            np.max(self.pixel_positions[:, :, 0]),
            np.max(self.pixel_positions[:, :, 1]),
            np.min(self.pixel_positions[:, :, 1]),
        )

    @property
    def extent_mm(self):
        """The extent of the beamformed image in mm."""
        return tuple([val * 1e3 for val in self.extent])

    @property
    def dx(self):
        """The pixel size/spacing in the x-direction in m."""
        return np.abs(self.pixel_positions[0, 1, 0] - self.pixel_positions[0, 0, 0])

    @property
    def dz(self):
        """The pixel size/spacing in the z-direction in m."""
        return np.abs(self.pixel_positions[1, 0, 1] - self.pixel_positions[0, 0, 1])

    @property
    def shape(self):
        """The shape of the pixel grid [n_rows, n_cols]."""
        return (self.n_rows, self.n_cols)


class CartesianPixelGrid(PixelGrid):
    def __init__(self, n_x, n_z, dx_wl, dz_wl, z0, wavelength):
        """Creates a CartesianPixelGrid object. Stores the pixel positions in meters in
        a 2D array of shape `(2, n_rows, n_cols)`.

        Parameters
        ----------
        n_x : int
            The number of pixels in the beamforming grid in the x-direction.
        n_z : int
            The number of pixels in the beamforming grid in the z-direction.
        dx_wl : float
            The pixel size/spacing in the x-direction in wavelengths. (Wavelengths are
            defined as sound_speed/carrier_frequency.)
        dz_wl : float
            The pixel size/spacing in the z-direction in wavelengths. (Wavelengths are
            defined as sound_speed/carrier_frequency.)
        z0 : float
            The start-depth of the beamforming plane in meters.
        wavelength : float
            The wavelength to define the grid spacing in meters.
        """

        # Construct the grid of pixel positions
        x_vals = (np.arange(n_x) - n_x / 2) * dx_wl * wavelength
        z_vals = np.arange(n_z) * dz_wl * wavelength + z0

        x_grid, z_grid = np.meshgrid(x_vals, z_vals)

        self.pixel_positions = np.stack((x_grid, z_grid), axis=2)

        super().__init__(self.pixel_positions)


class PolarPixelGrid(PixelGrid):
    def __init__(self, n_ax, n_theta, dax_wl, dtheta_rad, z0, wavelength):

        total_arc = (n_theta - 1) * dtheta_rad

        # Construct the grid of pixel positions
        ax_vals = (np.arange(n_ax) - n_ax / 2) * dax_wl * wavelength + z0
        theta_vals = np.arange(n_theta) * dtheta_rad - total_arc / 2

        ax_grid, theta_grid = np.meshgrid(ax_vals, theta_vals)

        x_grid = ax_grid * np.sin(theta_grid)
        z_grid = ax_grid * np.cos(theta_grid)

        self.pixel_positions = np.stack((x_grid, z_grid))

        super().__init__(self.pixel_positions)
