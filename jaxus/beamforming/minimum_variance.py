"""Implementation of the Minimum Variance Beamformer.

Source: Adaptive Beamforming Applied to Medical Ultrasound Imaging
        by Johan-Fredrik Synnevag
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Union
from src.utils.beamform import *
from src.utils.beamform import PixelGrid


class MinimumVarianceBeamformer(Beamformer):
    def __init__(
        self,
        pixel_grid: PixelGrid,
        probe_geometry: np.ndarray,
        t0_delays: np.ndarray,
        initial_times: np.ndarray,
        sampling_frequency: Union[float, int],
        center_frequency: Union[float, int],
        sound_speed: Union[float, int],
        t_peak: Union[float, int],
        rx_apodization: np.ndarray,
        f_number: Union[float, int] = 3,
        z0: Union[float, int] = 0,
        iq_beamform: bool = False,
        l: Union[int, None] = None,
        epsilon: Union[float, int] = 1e-2,
    ):
        super().__init__(
            pixel_grid,
            probe_geometry,
            t0_delays,
            initial_times,
            sampling_frequency,
            center_frequency,
            sound_speed,
            t_peak,
            rx_apodization,
            f_number,
            z0,
            iq_beamform,
        )
        self.configure(l, epsilon)

    def configure(
        self,
        l: int | None = None,
        epsilon: float | int = 1e-5,
    ):
        if l is None:
            l = self._probe_geometry.shape[0] // 3
        self._l = l

        if not isinstance(epsilon, (float, int)):
            raise TypeError("epsilon must be a float or int")
        self._epsilon = float(epsilon)

    def _get_beamform_func(
        self,
    ):
        """Returns a jit-compiled function that can be used to beamform a transmit.

        ### Returns:
            `beamform_pixel` (`function`): A function that can be used to beamform a
                transmit
        """

        # ==============================================================================
        # Define the beamform_pixel function to be vmapped over all pixels
        # ==============================================================================
        def beamform_pixel(data, pixel_pos, tx):
            """Beamforms a single pixel of a single frame and single transmit.

            ### Args:
                `data` (`jnp.ndarray`): The IQ data to beamform of shape
                    `(n_samples, n_elements)`.
                `pixel_pos` (`jnp.ndarray`): The position of the pixel to beamform to in
                    meters of shape `(2,)`.
                `tx` (`int`): The transmit to beamform.
                `iq_beamform` (`bool`): Whether to beamform the IQ data directly.
                    Default is False.

            ### Returns:
                `bf_value` (`float`): The beamformed value for the pixel.
            """

            tof_corrected = self._tof_correct_pixel(data, pixel_pos, tx)

            # ==========================================================================
            # Compute R_l
            # ==========================================================================
            N = tof_corrected.shape[0]

            # Create a batch of subvectors
            # Each subvector is vec[n:n+L], for n in range(N-L+1)
            indices = jnp.arange(N - self.l + 1)[:, None] + jnp.arange(self.l)

            # This will be a (N-L+1) x L matrix
            subvectors = tof_corrected[indices]

            # Vectorized outer product for a batch of subvectors
            outer_product = vmap(lambda x: jnp.outer(x, x))(subvectors)

            # Sum the outer products
            R_L = jnp.sum(outer_product, axis=0)

            # Normalize
            R_L = R_L / (N - self.l + 1)

            # Add epsilon to the diagonal
            R_L = R_L + self._epsilon * jnp.trace(R_L) * jnp.eye(self._l)

            # ==========================================================================
            # Compute the weights
            # ==========================================================================
            # The weights are computed as w = (R^-1 @ a) / (a^H @ R^-1 @ a)

            # Compute R^-1 @ a
            R_La = jnp.linalg.solve(R_L, jnp.ones(self._l))

            # Divide by (a^H @ R^-1 @ a)
            # Since the data is already TOF corrected, a is all ones, thus we can just
            # sum the elements of R_La
            weights = R_La / jnp.sum(R_La)

            # ==========================================================================
            # Compute the beamformed value
            # ==========================================================================

            z = vmap(lambda x: jnp.dot(weights.conj().T, x))(subvectors)
            z = jnp.sum(z)
            # Divide by the number of subvectors
            z = z / (N - self.l + 1) / 1000

            return z

        # ==================================================================================
        # vmap the beamform_pixel function over all pixels
        # ==================================================================================
        beamform_pixel = jit(vmap(beamform_pixel, in_axes=(None, 0, None)))

        return beamform_pixel

    @property
    def l(self):
        return self._l
