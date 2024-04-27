"""Functionality to convert Verasonics matlab raw files to the hdf5 format.

## Example
```MATLAB
>> setup_script;
>> VSX;
>> save_raw('C:/path/to/raw_data.mat');
```
Then in python:
```python
from hdf5.data_format.hdf5_from_matlab_raw import hdf5_from_matlab_raw

hdf5_from_matlab_raw("C:/path/to/raw_data.mat", "C:/path/to/output.hdf5")
```

Or alternatively, use the script below to convert all .mat files in a directory.
```commandline
python hdf5_from_matlab_raw.py
```
"""

from pathlib import Path

import h5py
import numpy as np

import jaxus.utils.log as log

from .hdf5_data_format import generate_hdf5_dataset


def dereference_index(file, dataset, index):
    """Get the element at the given index from the dataset, dereferencing it if
    necessary.

    MATLAB stores items in struct array differently depending on the size. If the size
    is 1, the item is stored as a regular dataset. If the size is larger, the item is
    stored as a dataset of references to the actual data.

    This function dereferences the dataset if it is a reference. Otherwise, it returns
    the dataset.
    """
    if isinstance(dataset.fillvalue, h5py.h5r.Reference):
        reference = dataset[index, 0]
        return file[reference][:]
    else:
        if index > 0:
            print(
                f"Warning: index {index} is not a reference. You are probably "
                "incorrectly indexing a dataset."
            )
        return dataset


def get_reference_size(dataset):
    """Get the size of a reference dataset."""
    if isinstance(dataset.fillvalue, h5py.h5r.Reference):
        return len(dataset)
    else:
        return 1


def decode_string(dataset):
    """Decode a string dataset."""
    return "".join([chr(c) for c in dataset.squeeze()])


def read_probe_geometry(file):
    """
    Read the probe geometry from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the probe geometry from. (The file should be opened in read
        mode.)

    Returns
    -------
    probe_geometry : np.ndarray
        The probe geometry of shape `(n_el, 3)`.
    """
    # Read the probe geometry from the file
    probe_geometry = file["Trans"]["ElementPos"][:3, :]

    # Transpose the probe geometry to have the shape (n_el, 3)
    probe_geometry = probe_geometry.T

    # Read the unit
    unit = decode_string(file["Trans"]["units"][:])

    # Convert the probe geometry to meters
    if unit == "mm":
        probe_geometry = probe_geometry / 1000
    else:
        wavelength = read_wavelength(file)
        probe_geometry = probe_geometry * wavelength

    return probe_geometry


def read_wavelength(file):
    """Reads the wavelength from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the wavelength from. (The file should be opened in read mode.)

    Returns
    -------
    wavelength : float
        The wavelength of the probe.
    """
    center_frequency = read_probe_center_frequency(file)
    sound_speed = read_sound_speed(file)
    wavelength = sound_speed / center_frequency
    return wavelength


def read_transmit_events(file):
    """Read the events from the file and finds the order in which transmits and receives
    appear in the events.

    Parameters
    ----------
    file : h5py.File
        The file to read the events from. (The file should be opened in read mode.)

    Returns
    -------
    tx_order : list
        The order in which the transmits appear in the events.
    rcv_order : list
        The order in which the receives appear in the events.
    time_to_next_acq : np.ndarray
        The time to next acquisition of shape `(n_frames, n_tx)`.
    """
    num_events = file["Event"]["info"].shape[0]

    # In the Verasonics the transmits may not be in order in the TX structure and a
    # transmit might be reused. Therefore, we need to keep track of the order in which
    # the transmits appear in the Events.
    tx_order = []
    rcv_order = []
    time_to_next_acq = []

    for i in range(num_events):

        # Get the tx
        event_tx = dereference_index(file, file["Event"]["tx"], i)
        event_tx = int(event_tx.item())

        # Get the rcv
        event_rcv = dereference_index(file, file["Event"]["rcv"], i)
        event_rcv = int(event_rcv.item())

        if not bool(event_tx) == bool(event_rcv):
            print(
                "Events should have both a transmit and a receive or neither. "
                f"Event {i} has a transmit but no receive or vice versa."
            )

        if not event_tx:
            continue

        # Subtract one to make the indices 0-based
        event_tx -= 1
        event_rcv -= 1

        # Check in the Receive structure if this is still the first frame
        framenum_ref = file["Receive"]["framenum"][event_rcv, 0]
        framenum = file[framenum_ref][:].item()

        # Only add the event to the list if it is the first frame since we assume
        # that all frames have the same transmits and receives
        if framenum == 1:
            # Add the event to the list
            tx_order.append(event_tx)
            rcv_order.append(event_rcv)

        # Read the time_to_next_acq
        seq_control_indices = dereference_index(file, file["Event"]["seqControl"], i)

        for seq_control_index in seq_control_indices:
            seq_control_index = int(seq_control_index.item() - 1)
            seq_control = dereference_index(
                file, file["SeqControl"]["command"], seq_control_index
            )
            # Decode the seq_control int array into a string
            seq_control = decode_string(seq_control)
            if seq_control == "timeToNextAcq":
                value = dereference_index(
                    file, file["SeqControl"]["argument"], seq_control_index
                ).item()
                value = value * 1e-6
                time_to_next_acq.append(value)

    n_tx = len(tx_order)
    time_to_next_acq = np.array(time_to_next_acq)
    time_to_next_acq = np.reshape(time_to_next_acq, (-1, n_tx))

    return tx_order, rcv_order, time_to_next_acq


def read_t0_delays_apod(file, tx_order):
    """
    Read the t0 delays and apodization from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the t0 delays from. (The file should be opened in read mode.)

    Returns
    -------
    t0_delays : np.ndarray
        The t0 delays of shape `(n_tx, n_el)`.
    apod : np.ndarray
        The apodization of shape `(n_el,)`.
    """

    t0_delays_list = []
    tx_apodizations_list = []

    wavelength = read_wavelength(file)
    sound_speed = read_sound_speed(file)

    for n in tx_order:
        # Get column vector of t0_delays
        t0_delays = dereference_index(file, file["TX"]["Delay"], n)
        # Turn into 1d array
        t0_delays = t0_delays[:, 0]

        t0_delays_list.append(t0_delays)

        # Get column vector of apodizations
        tx_apodizations = dereference_index(file, file["TX"]["Apod"], n)
        # Turn into 1d array
        tx_apodizations = tx_apodizations[:, 0]
        tx_apodizations_list.append(tx_apodizations)

    t0_delays = np.stack(t0_delays_list, axis=0)
    apodizations = np.stack(tx_apodizations_list, axis=0)

    # Convert the t0_delays to meters
    t0_delays = t0_delays * wavelength / sound_speed

    return t0_delays, apodizations


def read_sampling_frequency(file):
    """
    Read the sampling frequency from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the sampling frequency from. (The file should be opened in read
        mode.)

    Returns
    -------
    sampling_frequency : float
        The sampling frequency.
    """
    # Read the sampling frequency from the file
    adc_rate = dereference_index(file, file["Receive"]["decimSampleRate"], 0)
    quaddecim = dereference_index(file, file["Receive"]["quadDecim"], 0)

    sampling_frequency = adc_rate / quaddecim * 1e6

    sampling_frequency = float(sampling_frequency[0, 0])

    return sampling_frequency


def read_waveforms(file, tx_order):
    """
    Read the waveforms from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the waveforms from. (The file should be opened in read mode.)

    Returns
    -------
    waveforms : np.ndarray
        The waveforms of shape `(n_tx, n_samples)`.
    """
    waveforms_one_way_list = []
    waveforms_two_way_list = []

    # Read all the waveforms from the file
    n_tw = get_reference_size(file["TW"]["Wvfm1Wy"])
    for n in range(n_tw):
        # Get the row vector of the 1-way waveform
        waveform_one_way = dereference_index(file, file["TW"]["Wvfm1Wy"], n)[:]
        # Turn into 1d array
        waveform_one_way = waveform_one_way[0, :]

        # Get the row vector of the 2-way waveform
        waveform_two_way = dereference_index(file, file["TW"]["Wvfm2Wy"], n)[:]
        # Turn into 1d array
        waveform_two_way = waveform_two_way[0, :]

        waveforms_one_way_list.append(waveform_one_way)
        waveforms_two_way_list.append(waveform_two_way)

    waveform_tx_indices = []

    for n in tx_order:
        # Read the waveform
        waveform_index = dereference_index(file, file["TX"]["waveform"], n)[:]
        # Subtract one to make the indices 0-based
        waveform_index -= 1
        # Turn into integer
        waveform_index = int(waveform_index.item())
        waveform_tx_indices.append(waveform_index)

    return (
        np.array(waveform_tx_indices).astype(np.int32),
        waveforms_one_way_list,
        waveforms_two_way_list,
    )


def read_polar_angles(file, tx_order):
    """
    Read the polar angles from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the polar angles from. (The file should be opened in read
        mode.)

    Returns
    -------
    polar_angles : np.ndarray
        The polar angles of shape `(n_tx,)`.
    """
    polar_angles_list = []

    for n in tx_order:
        # Read the polar angle
        polar_angle = dereference_index(file, file["TX"]["Steer"], n)[:]
        # Turn into 1d array
        polar_angle = polar_angle[0, 0]

        polar_angles_list.append(polar_angle)

    polar_angles = np.stack(polar_angles_list, axis=0)

    return polar_angles


def read_azimuth_angles(file, tx_order):
    """
    Read the azimuth angles from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the azimuth angles from. (The file should be opened in read
        mode.)

    Returns
    -------
    azimuth_angles : np.ndarray
        The azimuth angles of shape `(n_tx,)`.
    """
    azimuth_angles_list = []

    for n in tx_order:
        # Read the azimuth angle
        azimuth_angle = dereference_index(file, file["TX"]["Steer"], n)[:]
        # Turn into 1d array
        azimuth_angle = azimuth_angle[1, 0]

        azimuth_angles_list.append(azimuth_angle)

    azimuth_angles = np.stack(azimuth_angles_list, axis=0)

    return azimuth_angles


def read_raw_data(file):
    """
    Read the raw data from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the raw data from. (The file should be opened in read mode.)

    Returns
    -------
    raw_data : np.ndarray
        The raw data of shape `(n_rcv, n_samples)`.
    """
    # Get the number of axial samples
    start_sample = dereference_index(file, file["Receive"]["startSample"], 0).item()
    end_sample = dereference_index(file, file["Receive"]["endSample"], 0).item()
    n_ax = int(end_sample - start_sample + 1)
    # Obtain the number of transmit events per frame
    tx_order, _, _ = read_transmit_events(file)
    n_tx = len(tx_order)

    # Read the raw data from the file
    raw_data = dereference_index(file, file["RcvData"], 0)

    raw_data = raw_data[:, :, : n_ax * n_tx]

    raw_data = raw_data.reshape((raw_data.shape[0], raw_data.shape[1], n_tx, -1))

    raw_data = np.transpose(raw_data, (0, 2, 3, 1))

    # Add channel dimension
    raw_data = raw_data[..., None]

    return raw_data


def read_probe_center_frequency(file):
    """Reads the center frequency of the probe from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the center frequency from. (The file should be opened in read
        mode.)

    Returns
    -------
    float
        The center frequency of the probe.
    """
    center_frequency = file["Trans"]["frequency"][0, 0] * 1e6
    return center_frequency


def read_sound_speed(file):
    """Reads the speed of sound from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the speed of sound from. (The file should be opened in read
        mode.)

    Returns
    -------
    float
        The speed of sound.
    """

    sound_speed = file["Resource"]["Parameters"]["speedOfSound"][0, 0].item()
    return sound_speed


def read_initial_times(file, rcv_order, sound_speed):
    """Reads the initial times from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the initial times from. (The file should be opened in read
        mode.)
    rcv_order : list
        The order in which the receives appear in the events.
    sound_speed : float
        The speed of sound.

    Returns
    -------
    np.ndarray
        The initial times of shape `(n_rcv,)`.
    """
    wavelength = read_wavelength(file)
    initial_times = []
    for n in rcv_order:
        start_depth = dereference_index(file, file["Receive"]["startDepth"], n).item()

        initial_times.append(2 * start_depth * wavelength / sound_speed)

    return np.array(initial_times).astype(np.float32)


def read_probe_name(file):
    """Reads the name of the probe from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the name of the probe from. (The file should be opened in read
        mode.)

    Returns
    -------
    str
        The name of the probe.
    """
    probe_name = file["Trans"]["name"][:]
    probe_name = decode_string(probe_name)
    return probe_name


def read_probe_element_width(file):
    """Reads the element width from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the element width from. (The file should be opened in read
        mode.)

    Returns
    -------
    float
        The element width.
    """
    element_width = file["Trans"]["elementWidth"][:][0, 0]

    # Read the unit
    unit = decode_string(file["Trans"]["units"][:])

    # Convert the probe element width to meters
    if unit == "mm":
        element_width = element_width / 1000
    else:
        wavelength = read_wavelength(file)
        element_width = element_width * wavelength

    return element_width


def read_probe_bandwidth(file):
    """Reads the transducer bandwidth from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the transducer bandwidth from. (The file should be opened in
        read mode.)

    Returns
    -------
    bandwidth : tuple
        The bandwidth of the probe in Hz.
    """
    bandwidth = file["Trans"]["Bandwidth"][:]
    bandwidth = (bandwidth[0, 0] * 1e6, bandwidth[1, 0] * 1e6)
    return bandwidth


def read_focus_distances(file, tx_order):
    """Reads the focus distances from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the focus distances from. (The file should be opened in read
        mode.)
    tx_order : list
        The order in which the transmits appear in the events.

    Returns
    -------
    list
        The focus distances.
    """
    focus_distances = []
    for n in tx_order:
        focus_distance = dereference_index(file, file["TX"]["focus"], n)[0, 0]
        focus_distances.append(focus_distance)
    return np.array(focus_distances).astype(np.float32)


def read_tgc_gain_curve(file):
    """Reads the TGC gain curve from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the TGC gain curve from. (The file should be opened in read
        mode.)

    Returns
    -------
    np.ndarray
        The TGC gain curve of shape `(n_ax,)`.
    """

    gain_curve = file["TGC"]["Waveform"][:][:, 0]

    # Normalize the gain_curve to [0, 40]dB
    gain_curve = gain_curve / 1023 * 40

    # The gain curve is sampled at 800ns (See Verasonics documentation for details.
    # Specifically the tutorial sequence programming)
    gain_curve_sampling_period = 800e-9

    # Define the time axis for the gain curve
    t_gain_curve = np.arange(gain_curve.size) * gain_curve_sampling_period

    # Read the number of axial samples
    start_sample = dereference_index(file, file["Receive"]["startSample"], 0).item()
    end_sample = dereference_index(file, file["Receive"]["endSample"], 0).item()
    n_ax = int(end_sample - start_sample + 1)

    # Read the sampling frequency
    sampling_frequency = read_sampling_frequency(file)

    # Define the time axis for the axial samples
    t_samples = np.arange(n_ax) / sampling_frequency

    # Interpolate the gain_curve to the number of axial samples
    gain_curve = np.interp(t_samples, t_gain_curve, gain_curve)

    # The gain_curve gains are in dB, so we need to convert them to linear scale
    gain_curve = 10 ** (gain_curve / 20)

    return gain_curve


def read_bandwidth_percent(file):
    """Reads the bandwidth percent from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the bandwidth percent from. (The file should be opened in read
        mode.)

    Returns
    -------
    int
        The bandwidth percent.
    """
    bandwidth_percent = dereference_index(file, file["Receive"]["sampleMode"], 0)
    bandwidth_percent = decode_string(bandwidth_percent)
    bandwidth_percent = int(bandwidth_percent[2:-2])
    return bandwidth_percent


def read_lens_correction(file):
    """Reads the lens correction from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the lens correction from. (The file should be opened in read
        mode.)

    Returns
    -------
    np.ndarray
        The lens correction.
    """
    lens_correction = file["Trans"]["lensCorrection"][0, 0].item()
    return lens_correction


def read_image_data_p(file):
    """Reads the image data from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the image data from. (The file should be opened in read mode.)

    Returns
    -------
    np.ndarray
        The image data.
    """
    # Get the dataset reference
    image_data_ref = file["ImgDataP"][0, 0]
    # Dereference the dataset
    image_data = file[image_data_ref][:]
    # Get the relevant dimensions
    image_data = image_data[:, 0, :, :]
    return image_data


def hdf5_from_matlab_raw(input_path, output_path):
    """Converts a Verasonics matlab raw file to the hdf5 format. The MATLAB file
    should be created using the `save_raw` function and be stored in "v7.3" format.

    Parameters
    ----------
    input_path : str
        The path to the input file (.mat file).
    output_path : str
        The path to the output file (.hdf5 file).
    """
    # Load the data
    with h5py.File(input_path, "r") as file:
        probe_geometry = read_probe_geometry(file)
        tx_order, rcv_order, time_to_next_transmit = read_transmit_events(file)
        t0_delays, tx_apodizations = read_t0_delays_apod(file, tx_order)
        sampling_frequency = read_sampling_frequency(file)
        waveform_tx_indices, waveforms_one_way_list, waveforms_two_way_list = (
            read_waveforms(file, tx_order)
        )
        polar_angles = read_polar_angles(file, tx_order)
        azimuth_angles = read_azimuth_angles(file, tx_order)
        bandwidth_percent = read_bandwidth_percent(file)
        raw_data = read_raw_data(file)
        image_data = read_image_data_p(file)
        center_frequency = read_probe_center_frequency(file)
        sound_speed = read_sound_speed(file)
        initial_times = read_initial_times(file, rcv_order, sound_speed)
        probe_name = read_probe_name(file)
        focus_distances = read_focus_distances(file, tx_order)
        lens_correction = read_lens_correction(file)
        try:
            bandwidth = read_probe_bandwidth(file)
        except KeyError:
            bandwidth = None
        try:
            element_width = read_probe_element_width(file)
        except KeyError:
            element_width = None
        try:
            tgc_gain_curve = read_tgc_gain_curve(file)
        except KeyError:
            tgc_gain_curve = None

        if "setup_script_text" in file:
            setup_script_text = file["setup_script_text"][:]
            setup_script_text = decode_string(setup_script_text)

        # If the data is captured in BS100BW mode or BS50BW mode, the data is stored in
        # as complex IQ data and the sampling frequency is halved.
        if bandwidth_percent in (50, 100):
            raw_data = np.concatenate(
                (
                    raw_data[:, :, 0::2, :, :],
                    -raw_data[:, :, 1::2, :, :],
                ),
                axis=-1,
            )
            sampling_frequency = sampling_frequency / 2

        # Create the output directory if it does not exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate the hdf5 dataset
        generate_hdf5_dataset(
            path=output_path,
            raw_data=raw_data,
            image=image_data,
            probe_geometry=probe_geometry,
            sampling_frequency=sampling_frequency,
            center_frequency=center_frequency,
            initial_times=initial_times,
            sound_speed=sound_speed,
            probe_name=probe_name,
            description="",
            focus_distances=focus_distances,
            polar_angles=polar_angles,
            azimuth_angles=azimuth_angles,
            tx_apodizations=tx_apodizations,
            t0_delays=t0_delays,
            bandwidth_percent=bandwidth_percent,
            time_to_next_transmit=time_to_next_transmit,
            waveform_indices=waveform_tx_indices,
            waveform_samples_one_way=waveforms_one_way_list,
            waveform_samples_two_way=waveforms_two_way_list,
            lens_correction=lens_correction,
            bandwidth=bandwidth,
            element_width=element_width,
            tgc_gain_curve=tgc_gain_curve,
        )
        log.info(f"Converted {log.yellow(input_path)} to {log.yellow(output_path)}")
