"""Functions for checking the validity of inputs and warning the user if values are
suspicious (very low or very high for instance).

When a variabele is of the wrong data type, a TypeError is raised. When an array is
of the wrong shape, a ValueError is raised. When a value is suspicious, a warning is
printed to the console.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import jaxus.utils.log as log


def check_n_ax(n_ax):
    """Checks if the input is a valid number of axial samples."""
    if not isinstance(n_ax, int):
        raise TypeError("n_ax is not an integer")
    if n_ax <= 0:
        raise ValueError("n_ax <= 0")

    if n_ax < 128:
        log.warning(f"n_ax = {n_ax} is very low. Are you sure this is correct?")

    if n_ax > 4096:
        log.warning(f"n_ax = {n_ax} is very high. Are you sure this is correct?")

    return int(n_ax)


def check_n_el(n_el):
    """Checks if the input is a valid number of elements."""
    if not isinstance(n_el, int):
        raise TypeError("n_el is not an integer")
    if n_el <= 0:
        raise ValueError("n_el <= 0")

    if n_el < 16:
        log.warning(f"n_el = {n_el} is very low. Are you sure this is correct?")

    if n_el > 512:
        log.warning(f"n_el = {n_el} is very high. Are you sure this is correct?")

    return int(n_el)


def check_t0_delays(t0_delays, jax_allowed=True, numpy_allowed=True, transmit_dim=True):
    """Checks if the input is a valid array of t0_delays.

    Parameters
    ----------
    t0_delays : np.array or jnp.array
        The input array.
    jax_allowed : bool, default=True
        If True, jax arrays are allowed.
    numpy_allowed : bool, default=True
        If True, numpy arrays are allowed.
    transmit_dim : bool, default=True
        If True, the array should have shape `(n_tx, n_el)`. Defaults to True.

    Raises
    ------
    TypeError
        If the input is not an ndarray.
    TypeError
        If the input is not float32 or float64.
    """
    if not isinstance(t0_delays, (jnp.ndarray, np.ndarray)):
        raise TypeError("t0_delays is not an ndarray")
    if not t0_delays.dtype in [jnp.float32, jnp.float64]:
        raise TypeError("t0_delays.dtype is not float32 or float64")

    if not jax_allowed and isinstance(t0_delays, jnp.ndarray):
        raise TypeError("t0_delays is not a numpy array")

    if not numpy_allowed and isinstance(t0_delays, np.ndarray):
        raise TypeError("t0_delays is not a jax array")

    ndim = 2 if transmit_dim else 1
    if not t0_delays.ndim == ndim:
        raise ValueError("t0_delays.ndim != 2")

    if isinstance(t0_delays, jnp.ndarray):
        module = jnp
    else:
        module = np

    if module.any(t0_delays < 0):
        raise ValueError("Not all t0_delays >= 0")

    if module.min(t0_delays) != 0:
        raise ValueError("Smallest t0_delay is not 0")

    return t0_delays.astype(np.float32)


def check_tx_apodization(tx_apodization, jax_allowed=True, numpy_allowed=True):
    """Checks if the input is a valid array of tx_apodization values.

    Parameters
    ----------
    tx_apodization : np.array or jnp.array
        The input array.
    jax_allowed : bool, default=True
        If True, jax arrays are allowed.
    numpy_allowed : bool, optional
        If True, numpy arrays are allowed.

    Raises
    ------
    TypeError
        If the input is not an ndarray.
    TypeError
        If the input is not float32 or float64.
    """
    if not isinstance(tx_apodization, (jnp.ndarray, np.ndarray)):
        raise TypeError("tx_apodization is not an ndarray")
    if not tx_apodization.dtype in [jnp.float32, jnp.float64]:
        raise TypeError("tx_apodization.dtype is not float32 or float64")

    if not jax_allowed and isinstance(tx_apodization, jnp.ndarray):
        raise TypeError("tx_apodization is not a numpy array")

    if not numpy_allowed and isinstance(tx_apodization, np.ndarray):
        raise TypeError("tx_apodization is not a jax array")

    if not tx_apodization.ndim == 1:
        raise ValueError("tx_apodization.ndim != 1")

    if isinstance(tx_apodization, jnp.ndarray):
        module = jnp
    else:
        module = np

    if module.any(tx_apodization < 0):
        raise ValueError("Not all tx_apodization >= 0")

    if module.any(tx_apodization > 1):
        raise ValueError("Not all tx_apodization <= 1")

    return tx_apodization.astype(np.float32)


def check_sound_speed(sound_speed):
    """Checks if the input is a valid sound speed."""
    if not isinstance(sound_speed, (float, int)):
        raise TypeError("sound_speed is not a float or an integer")
    if sound_speed <= 0:
        raise ValueError("sound_speed <= 0")

    if sound_speed < 1000:
        log.warning(
            f"sound_speed = {sound_speed} m/s is very low. Are you sure this is correct?"
        )

    if sound_speed > 2000:
        log.warning(
            f"sound_speed = {sound_speed} m/s is very high. Are you sure this is correct?"
        )

    return float(sound_speed)


def check_frequency(frequency, verbose=False):
    """Checks if the input is a valid frequency."""

    if isinstance(frequency, (np.ndarray, jnp.ndarray)):
        if frequency.size != 1:
            raise ValueError("frequency.size != 1")
        frequency = frequency.item()

    if not isinstance(frequency, (float, int)):
        raise TypeError("frequency is not a float or an integer")
    if frequency <= 0:
        raise ValueError("frequency <= 0")

    if verbose:
        if frequency < 1e6:
            log.warning(
                f"frequency = {frequency} Hz is very low. Are you sure this is correct?"
            )

        if frequency > 190e6:
            log.warning(
                f"frequency = {frequency} Hz is very high. Are you sure this is correct?"
            )

    return float(frequency)


def check_element_width(element_width, unit="mm"):
    """Checks if the input is a valid element width."""
    assert unit in ["mm", "wl"], "unit must be 'mm' or 'wl'"

    if not isinstance(element_width, (float, int)):
        raise TypeError("element_width is not a float or an integer")
    if element_width <= 0:
        raise ValueError("element_width <= 0")

    if unit == "wl":
        factor = 1.3 / 2e-3
    else:
        factor = 1

    if element_width < 1e-4 * factor:
        log.warning(
            f"element_width = {element_width*factor*1e3:.2f} {unit} is very low. "
            "Are you sure this is correct?"
        )

    if element_width > 0.01 * factor:
        log.warning(
            f"element_width = {element_width*factor*1e3:.2f} {unit} is very high. "
            "Are you sure this is correct?"
        )

    return float(element_width)


def check_pulse_width(pulse_width):
    """Checks if the input is a valid pulse width."""
    if not isinstance(pulse_width, (float, int)):
        raise TypeError("pulse_width is not a float or an integer")
    if pulse_width <= 0:
        raise ValueError("pulse_width <= 0")

    if pulse_width < 100e-9:
        log.warning(
            f"pulse_width = {pulse_width} s is very low. Are you sure this is correct?"
        )

    if pulse_width > 5000e-9:
        log.warning(
            f"pulse_width = {pulse_width} s is very high. Are you sure this is correct?"
        )

    return float(pulse_width)


def check_attenuation_coefficient(attenuation_coefficient):
    """Checks if the input is a valid attenuation coefficient."""
    if not isinstance(attenuation_coefficient, (float, int)):
        raise TypeError("attenuation_coefficient is not a float or an integer")
    if attenuation_coefficient < 0:
        raise ValueError("attenuation_coefficient < 0")

    if attenuation_coefficient > 10:
        log.warning(
            f"attenuation_coefficient = {attenuation_coefficient} dB/cm/MHz is very high. Are you sure this is correct?"
        )

    return float(attenuation_coefficient)


def check_element_angles(element_angles):
    """Checks if the input is a valid array of element angles.

    Parameters
    ----------
    element_angles : np.array or jnp.array
        The input array.

    Raises
    ------
    TypeError
        If the input is not an ndarray.
    TypeError
        If the input is not float32 or float64.
    """
    if not isinstance(element_angles, (jnp.ndarray, np.ndarray)):
        raise TypeError("element_angles is not an ndarray")
    if not element_angles.dtype in [jnp.float32, jnp.float64]:
        raise TypeError("element_angles.dtype is not float32 or float64")

    if not element_angles.ndim == 1:
        raise ValueError("element_angles.ndim != 1")

    if isinstance(element_angles, jnp.ndarray):
        module = jnp
    else:
        module = np

    if module.any(element_angles < -np.pi / 2):
        raise ValueError("Not all element_angles >= -pi/2")

    if module.any(element_angles > np.pi / 2):
        raise ValueError("Not all element_angles <= pi/2")

    return element_angles.astype(np.float32)


def check_pos_array(
    positions, jax_allowed=True, numpy_allowed=True, ax_dim=1, name="positions_array"
):
    """Checks if the input is a valid scatterer positions array.

    Parameters
    ----------
    positions : np.array or jnp.array
        The input array.
    jax_allowed : bool, default=True
        If True, jax arrays are allowed.
    numpy_allowed : bool, default=True
        If True, numpy arrays are allowed.
    ax_dim : int, default=1
        The dimension specifying wheter it is the x- or z-coordinate.
    name : str, default="positions_array"
        The name of the array.

    Raises
    ------
    TypeError
        If the input is not an ndarray.
    TypeError
        If the input does not have shape `(2, n)`.
    TypeError
        If the input is not float32 or float64.
    """
    if not isinstance(positions, (jnp.ndarray, np.ndarray)):
        raise TypeError(f"{name} is not an ndarray")
    if positions.ndim != 2:
        raise ValueError(f"{name}.ndim != 2")
    if positions.shape[ax_dim] != 2:
        raise ValueError(f"{name}.shape[{ax_dim}] != 2")
    if not positions.dtype in [jnp.float32, jnp.float64]:
        raise TypeError(f"{name}.dtype is not float32 or float64")

    if not jax_allowed and isinstance(positions, jnp.ndarray):
        raise TypeError(f"{name} is not a numpy array")

    if not numpy_allowed and isinstance(positions, np.ndarray):
        raise TypeError(f"{name} is not a jax array")

    return positions.astype(np.float32)


def check_chunk_size(chunk_size):
    """Checks if the input is a valid chunk size."""
    if not isinstance(chunk_size, int):
        raise TypeError("chunk_size is not an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size <= 0")

    return int(chunk_size)


def check_initial_times(initial_times):
    """Checks if the input is a valid array of initial times."""

    if not isinstance(initial_times, (jnp.ndarray, np.ndarray)):
        raise TypeError("initial_times is not an ndarray")

    initial_times = initial_times.astype(np.float32)

    if np.any(initial_times < 0):
        raise ValueError("Not all initial_times >= 0")

    if np.min(initial_times) > 1e-3:
        log.warning(
            f"initial_times = {initial_times} s is very high. Are you sure this is correct?"
        )

    return initial_times.astype(np.float32)


def check_waveform_function(waveform_function):
    """Checks if the input is a valid waveform function."""
    # Check that the waveform function is None or a function
    if waveform_function is not None:
        if not callable(waveform_function):
            raise TypeError(
                "waveform_function must be None or a function. "
                f"Got {type(waveform_function)}."
            )


def _check_standard_rf_data(
    rf_data, jax_allowed=True, numpy_allowed=True, n_channels=1
):
    """Checks if the input is a valid RF data array.

    Parameters
    ----------
    rf_data : np.array or jnp.array
        The input array of RF data.
    jax_allowed : bool, default=True
        If True, jax arrays are allowed.
    numpy_allowed : bool, default=True
        If True, numpy arrays are allowed.

    Returns
    -------
    np.array
        The input array as a float32 array.

    Raises
    ------
    TypeError
        If the input is not an ndarray.
    TypeError
        If the input is not float32 or float64.
    """
    if not isinstance(rf_data, (jnp.ndarray, np.ndarray)):
        raise TypeError("rf_data is not an ndarray")
    if not rf_data.dtype in [jnp.float32, jnp.float64]:
        raise TypeError("rf_data.dtype is not float32 or float64")

    if not jax_allowed and isinstance(rf_data, jnp.ndarray):
        raise TypeError("rf_data is not a numpy array")

    if not numpy_allowed and isinstance(rf_data, np.ndarray):
        raise TypeError("rf_data is not a jax array")

    if not rf_data.ndim == 5:
        raise ValueError("rf_data.ndim != 5")

    if not rf_data.shape[4] == n_channels:
        raise ValueError(
            f"rf_data.shape[4] should be {n_channels}. Got {rf_data.shape[4]}."
        )

    return rf_data.astype(np.float32)


def check_standard_rf_data(rf_data, jax_allowed=True, numpy_allowed=True):
    """Checks if the input is a valid RF data array.

    Parameters
    ----------
    rf_data : np.array or jnp.array
        The input array of RF data.
    jax_allowed : bool, default=True
        If True, jax arrays are allowed.
    numpy_allowed : bool, default=True
        If True, numpy arrays are allowed.

    Returns
    -------
    np.array
        The input array as a float32 array.

    Raises
    ------
    TypeError
        If the input is not an ndarray.
    TypeError
        If the input is not float32 or float64.
    """
    return _check_standard_rf_data(rf_data, jax_allowed, numpy_allowed, 1).astype(
        np.float32
    )


def check_standard_iq_data(iq_data, jax_allowed=True, numpy_allowed=True):
    """Checks if the input is a valid IQ data array.

    Parameters
    ----------
    iq_data : np.array or jnp.array
        The input array of IQ data.
    jax_allowed : bool, default=True
        If True, jax arrays are allowed.
    numpy_allowed : bool, default=True
        If True, numpy arrays are allowed.

    Returns
    -------
    np.array
        The input array as a float32 array.

    Raises
    ------
    TypeError
        If the input is not an ndarray.
    TypeError
        If the input is not float32 or float64.
    """
    return _check_standard_rf_data(iq_data, jax_allowed, numpy_allowed, 2).astype(
        np.float32
    )


def check_standard_rf_or_iq_data(rf_or_iq_data, jax_allowed=True, numpy_allowed=True):
    """Checks if the input is a valid RF or IQ data array in the format
    `(n_frames, n_tx, n_ax, n_el, n_channels)`.

    Parameters
    ----------
    rf_or_iq_data : np.array or jnp.array
        The input array of RF or IQ data.
    jax_allowed : bool, default=True
        If True, jax arrays are allowed.
    numpy_allowed : bool, default=True
        If True, numpy arrays are allowed.

    Returns
    -------
    np.array
        The input array as a float32 array.
    """
    try:
        return check_standard_rf_data(rf_or_iq_data, jax_allowed, numpy_allowed).astype(
            np.float32
        )
    except ValueError:
        return check_standard_iq_data(rf_or_iq_data, jax_allowed, numpy_allowed).astype(
            np.float32
        )


def check_path(path):
    """Checks if the input is a valid path. Returns a Path object."""
    if not isinstance(path, str):
        raise TypeError("path is not a string")
    if len(path) == 0:
        raise ValueError("path is empty")

    return Path(path)


def check_existing_path(path):
    """Checks if the input is a valid path to an existing file. Returns a Path object."""
    path = check_path(path)

    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    return Path(path)


def check_posint(n, name="value"):
    """Checks if the input is a positive integer."""
    if not isinstance(n, int):
        raise TypeError(f"{name} is not an integer")
    if n <= 0:
        raise ValueError(f"{name} <= 0")

    return int(n)


def check_posfloat(n, name="value"):
    """Checks if the input is a positive float."""
    if not isinstance(n, (float, int)):
        raise TypeError(f"{name} is not a float or an integer")
    if n <= 0:
        raise ValueError(f"{name} <= 0")

    return float(n)


def check_nonnegfloat(n, name="value"):
    """Checks if the input is a non-negative float."""
    if not isinstance(n, (float, int)):
        raise TypeError(f"{name} is not a float or an integer")
    if n < 0:
        raise ValueError(f"{name} < 0")

    return float(n)


def check_shapes_consistent(probe_geometry, intial_times, t0_delays, apodization):
    """Checks if the shapes of the input arrays are consistent.

    Parameters
    ----------
    probe_geometry : np.array
        The probe geometry array.
    intial_times : np.array
        The initial times array.
    t0_delays : np.array
        The t0_delays array.
    apodization : np.array
        The apodization array.

    Raises
    ------
    ValueError
        If the number of elements in all arrays is not equal.
    ValueError
        If the number of transmitters in all arrays is not equal.
    """
    # Check the number of elements in all arrays
    if not probe_geometry.shape[0] == apodization.shape[0] == t0_delays.shape[1]:
        raise ValueError(
            "probe_geometry.shape[0] != apodization.shape[1] != t0_delays.shape[1]"
        )

    # Check the number of transmitters in all arrays
    if not intial_times.shape[0] == t0_delays.shape[0]:
        raise ValueError("intial_times.shape[0] != t0_delays.shape[0]")
