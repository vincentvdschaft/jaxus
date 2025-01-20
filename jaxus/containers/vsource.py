import numpy as np
import jax.numpy as jnp
from jaxus.utils.ultrasound import deduce_vsource


class VSource:
    """A virtual source class that represents a virtual source in 2D space."""

    def __init__(self, angle, depth):
        self.angle = angle * np.pi / 180
        self.depth = depth

    @property
    def position(self):
        return jnp.array([jnp.sin(self.angle), jnp.cos(self.angle)]) * self.depth

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        self._angle = jnp.mod(angle + jnp.pi, 2 * jnp.pi) - jnp.pi

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, depth):
        self._depth = float(depth)

    @staticmethod
    def from_position(position):
        """Initialize a virtual source from a given position."""
        angle = jnp.arctan2(position[1], position[0])
        depth = jnp.linalg.norm(position)
        return VSource(angle, depth)

    @staticmethod
    def from_t0_delays(probe_geometry, t0_delays, sound_speed):
        vsource_position = deduce_vsource(
            probe_geometry=probe_geometry, t0_delays=t0_delays, sound_speed=sound_speed
        )
        return VSource.from_position(vsource_position)
