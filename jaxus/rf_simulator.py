from typing import Union

import jax.numpy as jnp
import numpy as np
from jax import device_put, jit, vmap
from tqdm import tqdm

import jaxus.utils.log as log
from jaxus.containers.waveform import get_pulse
from jaxus.utils.checks import *


def _get_vectorized_simulate_function(
    t0_delays,
    probe_geometry,
    element_angles,
    tx_apodization,
    initial_time,
    element_width_wl,
    sampling_frequency,
    sound_speed,
    attenuation_coefficient,
    waveform_function,
    wavefront_only=False,
    tx_angle_sensitivity=True,
    rx_angle_sensitivity=True,
):
    """Returns a vectorized function that takes in an array of element indices,
    an array of sample indices, an array of scatterer positions and an array of
    scatterer amplitudes and returns the rf data of shape (n_ax, n_el, n_scat)

    Parameters
    ----------
    t0_delays : jnp.array
        The t0_delays of shape `(n_el,)`. These are shifted such that the smallest value
        in t0_delays is 0.
    probe_geometry : jnp.array
        The probe geometry of shape `(2, n_el)`.
    element_angles : jnp.array
        The element angles in radians of shape `(n_el,)`. Can be used to simulate curved
        arrays.
    tx_apodization : jnp.array
        The transmit apodization of shape `(n_el,)`.
    initial_time : float
        The time instant of the first sample in seconds.
    element_width_wl : float
        The width of the elements in wavelengths of the center frequency.
    sampling_frequency : float
        The sampling frequency in Hz.
    sound_speed : float
        The speed of sound in the medium.
    attenuation_coefficient : float
        The attenuation coefficient in Nepers/(MHz*carrier_frequency)
    waveform_function : function
        The function that returns the transmit waveform amplitude.
    wavefront_only : bool
        Set to True to only compute the wavefront of the rf data. Otherwise the rf data
        is computed as the sum of the wavefronts from indivudual transmit elements.
    tx_angle_sensitivity : bool
        Set to True to include the angle dependent strength of the transducer elements
        in the response.
    rx_angle_sensitivity : bool
        Set to True to include the angle dependent strength of the transducer elements
        in the response.

    Returns
    -------
    function: The vectorized function.
    """

    # Define initial function that takes a single scatterer and returns the rf data
    # that only depends on the vectorized parameters
    def _simulate_rf_sample(
        ax_index,
        element_index,
        scatterer_position,
        scatterer_amplitude,
    ):
        """Computes the amplitude of a single rf sample in response to a single
        scatterer.

        Parameters
        ----------
        ax_index : int
            The sample index.
        element_index : int
            The receiving element index.
        scatterer_position : jnp.array
            The scatterer position of shape `(2,)`.
        scatterer_amplitude : jnp.array
            The scatterer amplitude of shape ().
        t0_delays : jnp.array
            The t0_delays of shape `(n_el,)`. These are shifted such that the smallest
            value in t0_delays is 0.
        probe_geometry : jnp.array
            The probe geometry of shape `(n_el, 2)`.
        element_angles : jnp.array
            The element angles in radians of shape `(n_el,)`. Can be used to simulate
            curved arrays.
        tx_apodization : jnp.array
            The transmit apodization of shape `(n_el,)`.
        initial_time : float
            The time instant of the first sample in seconds.
        element_width : float
            The width of the elements in wavelengths of the center frequency.
        sampling_frequency : float
            The sampling frequency in Hz.
        carrier_frequency : float
            The center frequencu in Hz.
        pulse_width : float
            The pulse width in seconds.
        sound_speed : float
            The speed of sound in the medium.
        attenuation_coefficient : float
            The attenuation coefficient in Nepers/m

        Returns
        -------
            jnp.array: Amplitude of the sample.
        """
        sampling_period = 1 / sampling_frequency

        # Get the position of the receiving element
        rx_element_pos = probe_geometry[element_index]

        # Compute the total time of flight required to end up in the sample at ax_index
        t_sample = ax_index * sampling_period

        # Compute the time of flight from the pixel to the receiving element
        t_rx = jnp.linalg.norm(scatterer_position - rx_element_pos) / sound_speed

        # Compute the time of flight from each transmitting element to the pixel
        t_tx = (
            jnp.linalg.norm(probe_geometry - scatterer_position[None, :], axis=1)
            / sound_speed
            + t0_delays
        )

        # Compute the delay relative to the time instant of the sample
        # If all delays sum to be equal to t_sample then the signal will be received at
        # the sample with no delay
        delay = t_rx + t_tx

        # If wavefront only is True we only add a single wavefront with delay equal to
        # the smallest delay
        if wavefront_only:
            delay = jnp.min(delay)

        # A transducer element does not have equal strength/sensitivity in all
        # directions. This influences the response of the transducer to a scatterer
        # depending on the angle between the scatterer and the transducer element.
        # We compute both the angle for the transmitting element to the scatterer and
        # the angle from the scatterer to the receiving element and apply the
        # attenuation to the response.
        if tx_angle_sensitivity:
            thetatx = jnp.arctan2(
                (scatterer_position[None, 0] - probe_geometry[:, 0]),
                (scatterer_position[None, 1] - probe_geometry[:, 1]),
            )

            # Subtract element angles to get the angle relative to the element
            thetatx -= element_angles

            angular_response_tx = jnp.sinc(
                element_width_wl * jnp.sin(thetatx)
            ) * jnp.cos(thetatx)
        else:
            angular_response_tx = 1

        if rx_angle_sensitivity:
            # Apply angle dependent attenuation by computing the angle between the pixel and
            # the receiving element and then computing the sinc function
            theta = jnp.arctan2(
                (scatterer_position[0] - rx_element_pos[0]),
                (scatterer_position[1] - rx_element_pos[1]),
            )

            # Subtract element angles to get the angle relative to the element
            theta -= element_angles[element_index]

            angular_response_rx = jnp.sinc(element_width_wl * jnp.sin(theta)) * jnp.cos(
                theta
            )
        else:
            angular_response_rx = 1

        if not attenuation_coefficient == 0:
            attenuation = jnp.exp(-attenuation_coefficient * delay * sound_speed)
        else:
            attenuation = 1

        # Compute the response
        response = (
            waveform_function(t_sample + initial_time - delay)
            * tx_apodization
            * scatterer_amplitude
            * angular_response_tx
            * angular_response_rx
            * attenuation
        )

        # Sum over all transmitting elements
        summed_response = jnp.sum(response)

        return summed_response

    # Vectorize over scatterers
    vectorized_function = vmap(
        _simulate_rf_sample,
        in_axes=(
            None,
            None,
            0,
            0,
        ),
    )

    # Vectorize over element indices
    vectorized_function = vmap(
        vectorized_function,
        in_axes=(
            None,
            0,
            None,
            None,
        ),
    )

    # Vectorize over axial samples
    vectorized_function = vmap(
        vectorized_function,
        in_axes=(
            0,
            None,
            None,
            None,
        ),
    )

    return jit(vectorized_function)


def simulate_rf_transmit(
    n_ax: int,
    scatterer_positions: jnp.array,
    scatterer_amplitudes: jnp.array,
    t0_delays: jnp.array,
    probe_geometry: jnp.array,
    element_angles: jnp.array,
    tx_apodization: jnp.array,
    initial_time: float,
    element_width_wl: float,
    sampling_frequency: Union[float, int],
    carrier_frequency: Union[float, int],
    sound_speed: Union[float, int] = 1540,
    attenuation_coefficient: Union[float, int] = 0.0,
    wavefront_only: bool = False,
    tx_angle_sensitivity: bool = True,
    rx_angle_sensitivity: bool = True,
    waveform_function: bool = None,
    ax_chunk_size: int = 1024,
    scatterer_chunk_size: int = 1024,
    progress_bar: bool = False,
    device=None,
    verbose: bool = False,
):
    """Simulates rf data based on transmit settings and scatterer positions and
    amplitudes.


    Parameters
    ----------
    n_ax : int
        The number of axial samples.
    scatterer_positions : jnp.array
        The scatterer positions of shape `(n_scat, 2)`.
    scatterer_amplitudes : jnp.array
        The scatterer amplitudes of shape `(n_scat,)`.
    ax_chunk_size : int
        The number of axial samples to compute simultaneously.
    scatterer_chunk_size : int
        The number of scatterers to compute simultaneously.
    t0_delays : jnp.array
        The t0_delays of shape `(n_el,)`. These are shifted such that the smallest value
        in t0_delays is 0.
    probe_geometry : jnp.array
        The probe geometry of shape `(n_el, 2)`.
    element_angles : jnp.array
        The element angles in radians of shape `(n_el,)`. Can be used to simulate curved
        arrays.
    tx_apodization : jnp.array
        The transmit apodization of shape `(n_el,)`.
    initial_time : float
        The time instant of the first sample in seconds.
    element_width_wl : float
        The width of the elements in wavelengths of the center frequency.
    sampling_frequency : float
        The sampling frequency in Hz.
    carrier_frequency : float
        The center frequency of the transmit pulse in Hz.
    sound_speed : float
        The speed of sound in the medium.
    attenuation_coefficient : float
        The attenuation coefficient in dB/(MHz*cm)
    wavefront_only : bool
        Set to True to only compute the wavefront of the rf data. Otherwise the rf data
        is computed as the sum of the wavefronts from indivudual transmit elements.
    tx_angle_sensitivity : bool
        Set to True to include the angle dependent strength of the transducer elements
        in the response.
    rx_angle_sensitivity : bool
        Set to True to include the angle dependent
        strength of the transducer elements in the response.
    progress_bar : bool
        Set to True to display a progress bar.
    device : jax.Device
        The device to run the simulation on. If None then the simulation is run on the
        default device.

    Returns
    -------
        jnp.array: The rf data of shape `(n_ax, n_el)`
    """
    # Check that the waveform function is None or a function
    check_waveform_function(waveform_function)
    check_element_width(element_width_wl, unit="wl")
    check_frequency(carrier_frequency)
    check_frequency(sampling_frequency)
    check_sound_speed(sound_speed)
    check_attenuation_coefficient(attenuation_coefficient)
    check_pos_array(scatterer_positions, name="scatterer_positions")
    check_pos_array(probe_geometry, name="probe_geometry")
    check_t0_delays(t0_delays, transmit_dim=False)
    check_element_angles(element_angles)
    check_tx_apodization(tx_apodization)

    if waveform_function is None:
        tx_waveform = get_pulse(carrier_frequency, 400e-9)
    else:
        tx_waveform = waveform_function

    if not wavefront_only:
        tx_waveform = vmap(tx_waveform)

    # Convert to jnp arrays in case numpy arrays are passeds
    scatterer_positions = scatterer_positions[scatterer_amplitudes > 0]
    scatterer_amplitudes = scatterer_amplitudes[scatterer_amplitudes > 0]
    scatterer_positions = jnp.array(scatterer_positions)
    scatterer_amplitudes = device_put(scatterer_amplitudes, device=device)

    scatterer_amplitudes = jnp.array(scatterer_amplitudes)
    scatterer_amplitudes = device_put(scatterer_amplitudes, device=device)

    t0_delays = jnp.array(t0_delays)
    t0_delays = device_put(t0_delays, device=device)

    probe_geometry = jnp.array(probe_geometry)
    probe_geometry = device_put(probe_geometry, device=device)

    element_angles = jnp.array(element_angles)
    element_angles = device_put(element_angles, device=device)

    if wavefront_only is True:
        if tx_angle_sensitivity is True:
            tx_angle_sensitivity = False
            log.warning(
                "tx_angle_sensitivity is set to True while in wavefront_only mode. "
                "Changed to False."
            )
        if verbose:
            log.warning(
                "wavefront only is True. tx_apodization will be ignored. Elements that are "
                "turned off will be ignored by adding a large t0_delay."
            )

    tx_apodization = jnp.array(tx_apodization)
    tx_apodization = device_put(tx_apodization, device=device)

    # Convert attenuation coefficient from dB/(MHz*cm) to Nepers/m
    attenuation_coefficient = (
        attenuation_coefficient * np.log(10) / 20 * carrier_frequency * 1e-6 * 100
    )

    # Get a JIT compiled function that generates the rf data for a batch of axial sample
    # indices, a batch of element indices, and a batch of scatterer positions and
    # amplitudes
    simulation_function = _get_vectorized_simulate_function(
        t0_delays,
        probe_geometry,
        element_angles,
        tx_apodization,
        initial_time,
        element_width_wl,
        sampling_frequency,
        sound_speed,
        attenuation_coefficient,
        wavefront_only=wavefront_only,
        tx_angle_sensitivity=tx_angle_sensitivity,
        rx_angle_sensitivity=rx_angle_sensitivity,
        waveform_function=tx_waveform,
    )

    n_el = probe_geometry.shape[0]
    n_scat = scatterer_positions.shape[0]

    # Generate indices for the elements. (We will always compute all elements
    # simultaneously)
    el_indices = jnp.arange(n_el)
    el_indices = device_put(el_indices, device=device)

    # Get either a progresbar function or a dummy function
    progbar_func = lambda x, **kwargs: (x if not progress_bar else tqdm(x, **kwargs))

    # Initialize a list that will contain the chunks of rf data each containing a
    # subset of the rows)
    rf_data_row_sections = []

    for ax0 in progbar_func(
        range(0, n_ax, ax_chunk_size), desc="Axial chunks", leave=False
    ):
        ax1 = min(ax0 + ax_chunk_size, n_ax)
        ax_indices_chunk = jnp.arange(ax0, ax1)
        ax_indices_chunk = device_put(ax_indices_chunk, device=device)

        rf_data_chunk = jnp.zeros((ax1 - ax0, n_el))
        rf_data_chunk = device_put(rf_data_chunk, device=device)

        for scat0 in progbar_func(
            range(0, n_scat, scatterer_chunk_size),
            desc="Scatterer chunks",
            leave=False,
        ):
            scat1 = min(scat0 + scatterer_chunk_size, n_scat)
            scatterer_positions_chunk = scatterer_positions[scat0:scat1]
            scatterer_amplitudes_chunk = scatterer_amplitudes[scat0:scat1]

            # Compute the rf data for the current chunk of scatterers and add it to the
            # rf data chunk
            rf_data_chunk += jnp.sum(
                simulation_function(
                    ax_indices_chunk,
                    el_indices,
                    scatterer_positions_chunk,
                    scatterer_amplitudes_chunk,
                ),
                axis=2,
            )

        rf_data_row_sections.append(rf_data_chunk)

    # Concatenate the rf data row sections to get the full rf data
    rf_data = jnp.concatenate(rf_data_row_sections, axis=0)

    return rf_data


def simulate_rf_data(
    n_ax: int,
    scatterer_positions: jnp.array,
    scatterer_amplitudes: jnp.array,
    t0_delays: jnp.array,
    probe_geometry: jnp.array,
    element_angles: jnp.array,
    tx_apodizations: jnp.array,
    initial_times: float,
    element_width_wl: float,
    sampling_frequency: Union[float, int],
    carrier_frequency: Union[float, int],
    sound_speed: Union[float, int] = 1540,
    attenuation_coefficient: Union[float, int] = 0.0,
    wavefront_only: bool = False,
    tx_angle_sensitivity: bool = True,
    rx_angle_sensitivity: bool = True,
    waveform_function: bool = None,
    ax_chunk_size: int = 1024,
    scatterer_chunk_size: int = 1024,
    progress_bar: bool = False,
    device=None,
    verbose: bool = False,
):
    """Simulates rf data based on transmit settings and scatterer positions and
    amplitudes.


    Parameters
    ----------
    n_ax : int
        The number of axial samples.
    scatterer_positions : jnp.array
        The scatterer positions of shape `(n_scat, 2)`.
    scatterer_amplitudes : jnp.array
        The scatterer amplitudes of shape `(n_scat,)`.
    ax_chunk_size : int
        The number of axial samples to compute simultaneously.
    scatterer_chunk_size : int
        The number of scatterers to compute simultaneously.
    t0_delays : jnp.array
        The t0_delays of shape `(n_tx, n_el)`. These are shifted such that the smallest value
        in t0_delays is 0.
    probe_geometry : jnp.array
        The probe geometry of shape `(n_el, 2)`.
    element_angles : jnp.array
        The element angles in radians of shape `(n_el,)`. Can be used to simulate curved
        arrays.
    tx_apodizations : jnp.array
        The transmit apodization of shape `(n_tx, n_el)`.
    initial_times : np.array
        The time instant of the first sample in seconds of shape `(n_tx,)`.
    element_width_wl : float
        The width of the elements in wavelengths of the center frequency.
    sampling_frequency : float
        The sampling frequency in Hz.
    carrier_frequency : float
        The center frequency of the transmit pulse in Hz.
    sound_speed : float
        The speed of sound in the medium.
    attenuation_coefficient : float
        The attenuation coefficient in dB/(MHz*cm)
    wavefront_only : bool
        Set to True to only compute the wavefront of the rf data. Otherwise the rf data
        is computed as the sum of the wavefronts from indivudual transmit elements.
    tx_angle_sensitivity : bool
        Set to True to include the angle dependent strength of the transducer elements
        in the response.
    rx_angle_sensitivity : bool
        Set to True to include the angle dependent
        strength of the transducer elements in the response.
    progress_bar : bool
        Set to True to display a progress bar.
    device : jax.Device
        The device to run the simulation on. If None then the simulation is run on the
        default device.

    Returns
    -------
        jnp.array: The rf data of shape `(n_ax, n_el)`
    """

    n_tx, n_el = tx_apodizations.shape

    rf_transmits = []

    if progress_bar:
        pbar_fn = lambda x: tqdm(x, desc="Transmit", leave=False, colour="red")
    else:
        pbar_fn = lambda x: x

    for tx in pbar_fn(range(n_tx)):
        rf_transmit = simulate_rf_transmit(
            n_ax=n_ax,
            scatterer_positions=scatterer_positions,
            scatterer_amplitudes=scatterer_amplitudes,
            t0_delays=t0_delays[tx],
            probe_geometry=probe_geometry,
            element_angles=element_angles,
            tx_apodization=tx_apodizations[tx],
            initial_time=initial_times[tx],
            element_width_wl=element_width_wl,
            sampling_frequency=sampling_frequency,
            carrier_frequency=carrier_frequency,
            sound_speed=sound_speed,
            attenuation_coefficient=attenuation_coefficient,
            wavefront_only=wavefront_only,
            tx_angle_sensitivity=tx_angle_sensitivity,
            rx_angle_sensitivity=rx_angle_sensitivity,
            waveform_function=waveform_function,
            ax_chunk_size=ax_chunk_size,
            scatterer_chunk_size=scatterer_chunk_size,
            progress_bar=progress_bar,
            device=device,
            verbose=verbose,
        )

        rf_transmits.append(rf_transmit)

    rf_data = jnp.stack(rf_transmits, axis=0)

    return rf_data[None, :, :, :, None]
