from pathlib import Path
from typing import Union

import h5py
import numpy as np


def _first_not_none_item(arr):
    """
    Finds and returns the first non-None item in the given array.

    ### Args:
        `arr` (`list`): The input array.

    ### Returns:
        The first non-None item found in the array, or None if no such item exists.
    """
    non_none_items = [item for item in arr if item is not None]
    return non_none_items[0] if non_none_items else None


def generate_usbmd_dataset(
    path,
    raw_data=None,
    aligned_data=None,
    envelope_data=None,
    beamformed_data=None,
    image=None,
    image_sc=None,
    probe_geometry=None,
    sampling_frequency=None,
    center_frequency=None,
    initial_times=None,
    t0_delays=None,
    sound_speed=None,
    probe_name=None,
    description="No description was supplied",
    focus_distances=None,
    polar_angles=None,
    azimuth_angles=None,
    tx_apodizations=None,
    bandwidth_percent=None,
    time_to_next_transmit=None,
    waveform_indices=None,
    waveform_samples_one_way=None,
    waveform_samples_two_way=None,
    lens_correction=None,
    element_width=None,
    bandwidth=None,
    tgc_gain_curve=None,
):
    """
    Generates a dataset in the USBMD format.

    ### Args:
        `path` (`str`): The path to write the dataset to.
        `raw_data` (`np.ndarray`): The raw data of the ultrasound measurement of
            shape (n_frames, n_tx, n_ax, n_el, n_ch).
        `aligned_data` (`np.ndarray`): The aligned data of the ultrasound measurement of
            shape (n_frames, n_tx, n_ax, n_el, n_ch).
        `envelope_data` (`np.ndarray`): The envelope data of the ultrasound measurement of
            shape (n_frames, n_z, n_x).
        `beamformed_data` (`np.ndarray`): The beamformed data of the ultrasound measurement of
            shape (n_frames, n_z, n_x).
        `image` (`np.ndarray`): The ultrasound images to be saved of shape (n_frames, n_z, n_x).
        `image_sc` (`np.ndarray`): The scan converted ultrasound images to be saved
            of shape (n_frames, output_size_z, output_size_x).
        `probe_geometry` (`np.ndarray`): The probe geometry of shape (n_el, 3).
        `sampling_frequency` (`float`): The sampling frequency in Hz.
        `center_frequency` (`float`): The center frequency in Hz.
        `initial_times` (`list`): The times when the A/D converter starts sampling
            in seconds of shape (n_tx,). This is the time between the first element
            firing and the first recorded sample.
        `t0_delays` (`np.ndarray`): The t0_delays of shape (n_tx, n_el).
        `sound_speed` (`float`): The speed of sound in m/s.
        `probe_name` (`str`): The name of the probe.
        `description` (`str`): The description of the dataset.
        `focus_distances` (`np.ndarray`): The focus distances of shape (n_tx, n_el).
        `polar_angles` (`np.ndarray`): The polar angles of shape (n_el,).
        `azimuth_angles` (`np.ndarray`): The azimuth angles of shape (n_tx,).
        `tx_apodizations` (`np.ndarray`): The transmit delays for each element defining
            the wavefront in seconds of shape (n_tx, n_elem).
            This is the time between the first element firing and the last element firing.
        `bandwidth_percent` (`float`): The bandwidth of the transducer as a
            percentage of the center frequency.
        `time_to_next_transmit` (`np.ndarray`): The time between subsequent transmit
            events in s.
        `waveform_indices` (`np.ndarray`): The indices of the waveforms used for each
            transmit event.
        `waveform_samples_one_way` (`list`): The samples of the waveforms used for
            each transmit wave of shape (n_tw, n_samples).
        `waveform_samples_two_way` (`list`): The samples of the waveforms used for
            each transmit wave of shape (n_tw, n_samples).
        `lens_correction` (`np.ndarray`): Extra time added to the transmit delays to
            account for the lens in wavelengths.
        `element_width` (`float`): The width of the elements in the probe in meters.
        `bandwidth` (`tuple`): The beginning and end of the transducer in Hz.
        `tgc_gain_curve` (`np.ndarray`): The time gain compensation curve of shape
            `(n_ax,)`.



    ### Returns:
        (h5py.File): The example dataset.
    """

    assert isinstance(probe_name, str), "The probe name must be a string."
    assert isinstance(description, str), "The description must be a string."

    # Convert path to Path object
    path = Path(path)

    if path.exists():
        raise FileExistsError(f"The file {path} already exists.")

    # Create the directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as dataset:
        dataset.attrs["probe"] = probe_name
        dataset.attrs["description"] = description

        assert (
            isinstance(raw_data, np.ndarray) and raw_data.ndim == 5
        ), "The raw_data must be a numpy array of shape (n_frames, n_tx, n_ax, n_el, n_ch)."

        def convert_datatype(x, astype=np.float32):
            return x.astype(astype) if x is not None else None

        def first_not_none_shape(arr, axis):
            data = _first_not_none_item(arr)
            return data.shape[axis] if data is not None else None

        def add_dataset(group, name, data, description, unit):
            """Adds a dataset to the given group with a description and unit.
            If data is None, the dataset is not added."""
            if data is None:
                return
            dataset = group.create_dataset(name, data=data)
            dataset.attrs["description"] = description
            dataset.attrs["unit"] = unit

        n_frames = _first_not_none_item(
            [raw_data, aligned_data, envelope_data, beamformed_data, image_sc, image]
        ).shape[0]
        n_tx = first_not_none_shape([raw_data, aligned_data], axis=1)
        n_el = first_not_none_shape([raw_data, aligned_data], axis=3)
        n_ax = first_not_none_shape([raw_data, aligned_data], axis=2)
        n_ch = first_not_none_shape([raw_data, aligned_data], axis=4)

        # Write data group
        data_group = dataset.create_group("data")
        data_group.attrs["description"] = "This group contains the data."

        add_dataset(
            group=data_group,
            name="raw_data",
            data=convert_datatype(raw_data),
            description="The raw_data of shape (n_frames, n_tx, n_el, n_ax, n_ch).",
            unit="unitless",
        )

        add_dataset(
            group=data_group,
            name="aligned_data",
            data=convert_datatype(aligned_data),
            description="The aligned_data of shape (n_frames, n_tx, n_el, n_ax, n_ch).",
            unit="unitless",
        )

        add_dataset(
            group=data_group,
            name="envelope_data",
            data=convert_datatype(envelope_data),
            description="The envelope_data of shape (n_frames, n_z, n_x).",
            unit="unitless",
        )

        add_dataset(
            group=data_group,
            name="beamformed_data",
            data=convert_datatype(beamformed_data),
            description="The beamformed_data of shape (n_frames, n_z, n_x).",
            unit="unitless",
        )

        add_dataset(
            group=data_group,
            name="image",
            data=convert_datatype(image),
            unit="unitless",
            description="The images of shape [n_frames, n_z, n_x]",
        )

        add_dataset(
            group=data_group,
            name="image_sc",
            data=convert_datatype(image_sc),
            unit="unitless",
            description=(
                "The scan converted images of shape [n_frames, output_size_z,"
                " output_size_x]"
            ),
        )

        # Write scan group
        scan_group = dataset.create_group("scan")
        scan_group.attrs["description"] = "This group contains the scan parameters."

        add_dataset(
            group=scan_group,
            name="n_ax",
            data=n_ax,
            description="The number of axial samples.",
            unit="unitless",
        )

        add_dataset(
            group=scan_group,
            name="n_el",
            data=n_el,
            description="The number of elements in the probe.",
            unit="unitless",
        )

        add_dataset(
            group=scan_group,
            name="n_tx",
            data=n_tx,
            description="The number of transmits per frame.",
            unit="unitless",
        )

        add_dataset(
            group=scan_group,
            name="n_ch",
            data=n_ch,
            description=(
                "The number of channels. For RF data this is 1. For IQ data "
                "this is 2."
            ),
            unit="unitless",
        )

        add_dataset(
            group=scan_group,
            name="n_frames",
            data=n_frames,
            description="The number of frames.",
            unit="unitless",
        )

        add_dataset(
            group=scan_group,
            name="sound_speed",
            data=sound_speed,
            description="The speed of sound in m/s",
            unit="m/s",
        )

        add_dataset(
            group=scan_group,
            name="probe_geometry",
            data=probe_geometry,
            description="The probe geometry of shape (n_el, 3).",
            unit="m",
        )

        add_dataset(
            group=scan_group,
            name="sampling_frequency",
            data=sampling_frequency,
            description="The sampling frequency in Hz.",
            unit="Hz",
        )

        add_dataset(
            group=scan_group,
            name="center_frequency",
            data=center_frequency,
            description="The center frequency of the transducer in Hz.",
            unit="Hz",
        )

        add_dataset(
            group=scan_group,
            name="initial_times",
            data=initial_times,
            description=(
                "The times when the A/D converter starts sampling "
                "in seconds of shape (n_tx,). This is the time between the "
                "first element firing and the first recorded sample."
            ),
            unit="s",
        )

        add_dataset(
            group=scan_group,
            name="t0_delays",
            data=t0_delays,
            description="The t0_delays of shape (n_tx, n_el).",
            unit="s",
        )

        add_dataset(
            group=scan_group,
            name="tx_apodizations",
            data=tx_apodizations,
            description=(
                "The transmit delays for each element defining the"
                " wavefront in seconds of shape (n_tx, n_elem). This is"
                " the time at which each element fires shifted such that"
                " the first element fires at t=0."
            ),
            unit="unitless",
        )

        add_dataset(
            group=scan_group,
            name="focus_distances",
            data=focus_distances,
            description=(
                "The transmit focus distances in meters of "
                "shape (n_tx,). For planewaves this is set to Inf."
            ),
            unit="m",
        )

        add_dataset(
            group=scan_group,
            name="polar_angles",
            data=polar_angles,
            description=(
                "The polar angles of the transmit beams in radians of shape (n_tx,)."
            ),
            unit="rad",
        )

        add_dataset(
            group=scan_group,
            name="azimuth_angles",
            data=azimuth_angles,
            description=(
                "The azimuthal angles of the transmit beams in radians of shape (n_tx,)."
            ),
            unit="rad",
        )

        add_dataset(
            group=scan_group,
            name="bandwidth_percent",
            data=bandwidth_percent,
            description=(
                "The receive bandwidth of RF signal in percentage of center frequency."
            ),
            unit="unitless",
        )

        add_dataset(
            group=scan_group,
            name="time_to_next_transmit",
            data=time_to_next_transmit,
            description=("The time between subsequent transmit events."),
            unit="s",
        )

        add_dataset(
            group=scan_group,
            name="tx_waveform_indices",
            data=waveform_indices,
            description=("The indices of the waveforms used for each transmit event."),
            unit="unitless",
        )

        if waveform_samples_one_way is not None:
            # Create a group named waveforms
            waveforms_group_one_way = scan_group.create_group("waveforms_one_way")
            waveforms_group_one_way.attrs["description"] = (
                "This group contains the 1-way waveforms. That is the waveforms after "
                "passing through the transducer bandwidth once."
            )

            for n in range(len(waveform_samples_one_way)):
                add_dataset(
                    group=waveforms_group_one_way,
                    name=f"waveform_{str(n).zfill(3)}",
                    data=waveform_samples_one_way[n],
                    description=(
                        "The samples of the waveforms used for each transmit wave."
                    ),
                    unit="unitless",
                )
        if waveform_samples_two_way is not None:
            # Create a group named waveforms
            waveforms_group_two_way = scan_group.create_group("waveforms_two_way")
            waveforms_group_two_way.attrs["description"] = (
                "This group contains the 2-way waveforms. That is the waveforms after "
                "passing through the transducer bandwidth twice."
            )

            for n in range(len(waveform_samples_one_way)):
                add_dataset(
                    group=waveforms_group_two_way,
                    name=f"waveform_{str(n).zfill(3)}",
                    data=waveform_samples_two_way[n],
                    description=(
                        "The samples of the waveforms used for each transmit wave."
                    ),
                    unit="unitless",
                )

        if lens_correction is not None:
            add_dataset(
                group=scan_group,
                name="lens_correction",
                data=lens_correction,
                description=(
                    "Extra time added to the transmit delays to account for the lens in "
                    "wavelengths."
                ),
                unit="wavelengths",
            )

        if element_width is not None:
            add_dataset(
                group=scan_group,
                name="element_width",
                data=element_width,
                description="The width of the elements in the probe in meters.",
                unit="m",
            )

        if bandwidth is not None:
            add_dataset(
                group=scan_group,
                name="bandwidth",
                data=np.array(bandwidth, dtype=np.float32),
                description="The beginning and end of the transducer bandwidth in Hz.",
                unit="Hz",
            )

        if tgc_gain_curve is not None:
            add_dataset(
                group=scan_group,
                name="tgc_gain_curve",
                data=tgc_gain_curve,
                description="The time gain compensation curve of shape (n_ax,).",
                unit="unitless",
            )

    validate_dataset(path)


def validate_dataset(path):
    """Reads the hdf5 dataset at the given path and validates its structure.

    ### Args:
        path (str, pathlike): The path to the hdf5 dataset.

    """
    with h5py.File(path, "r") as dataset:

        def check_key(dataset, key):
            assert key in dataset.keys(), f"The dataset does not contain the key {key}."

        # Validate the root group
        check_key(dataset, "data")

        # validate the data group
        for key in dataset["data"].keys():

            # Validate data shape
            data_shape = dataset["data"][key].shape
            if key == "raw_data":
                assert (
                    len(data_shape) == 5
                ), "The raw_data group does not have a shape of length 5."
                assert (
                    data_shape[1] == dataset["scan"]["n_tx"][()]
                ), "n_tx does not match the second dimension of raw_data."
                assert (
                    data_shape[2] == dataset["scan"]["n_ax"][()]
                ), "n_ax does not match the third dimension of raw_data."
                assert (
                    data_shape[3] == dataset["scan"]["n_el"][()]
                ), "n_el does not match the fourth dimension of raw_data."
                assert data_shape[4] in (
                    1,
                    2,
                ), (
                    "The fifth dimension of raw_data, which is the complex channel "
                    "dimension is not 1 or 2."
                )

            elif key == "aligned_data":
                print("No validation has been defined for aligned data.")
            elif key == "beamformed_data":
                print("No validation has been defined for beamformed data.")
            elif key == "envelope_data":
                print("No validation has been defined for envelope data.")
            elif key == "image":
                assert (
                    len(data_shape) == 3
                ), "The image group does not have a shape of length 3."
            elif key == "image_sc":
                assert (
                    len(data_shape) == 3
                ), "The image_sc group does not have a shape of length 3."

        assert_unit_and_description_present(dataset)


def assert_scan_keys_present(dataset):
    """Ensure that all required keys are present.

    ### Args:
        `dataset` (`h5py.File`): The dataset instance to check.

    Raises:
        AssertionError: If a required key is missing or does not have the right shape.
    """

    # Ensure that all keys have the correct shape
    for key in dataset["scan"].keys():
        if key == "probe_geometry":
            correct_shape = (dataset["scan"]["n_el"][()], 3)
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The probe_geometry does not have the correct shape."

        elif key == "t0_delays":
            correct_shape = (
                dataset["scan"]["n_tx"][()],
                dataset["scan"]["n_el"][()],
            )
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The t0_delays does not have the correct shape."

        elif key == "tx_apodizations":
            correct_shape = (
                dataset["scan"]["n_tx"][()],
                dataset["scan"]["n_el"][()],
            )
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The tx_apodizations does not have the correct shape."

        elif key == "focus_distances":
            correct_shape = (dataset["scan"]["n_tx"][()],)
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The focus_distances does not have the correct shape."

        elif key == "polar_angles":
            correct_shape = (dataset["scan"]["n_tx"][()],)
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The polar_angles does not have the correct shape."

        elif key == "azimuth_angles":
            correct_shape = (dataset["scan"]["n_tx"][()],)
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The azimuthal_angles does not have the correct shape."

        elif key == "initial_times":
            correct_shape = (dataset["scan"]["n_tx"][()],)
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The initial_times does not have the correct shape."

        elif key in (
            "sampling_frequency",
            "center_frequency",
            "n_frames",
            "n_tx",
            "n_el",
            "n_ax",
            "n_ch",
            "sound_speed",
            "bandwidth_percent",
            "time_to_next_transmit",
        ):
            assert (
                dataset["scan"][key].size == 1
            ), f"{key} does not have the correct shape."

        else:
            print("No validation has been defined for %s.", key)


def assert_unit_and_description_present(hdf5_file, _prefix=""):
    """Checks that all datasets have a unit and description attribute.

    ### Args:
        `hdf5_file` (`h5py.File`): The hdf5 file to check.

    Raises:
        AssertionError: If a dataset does not have a unit or description
            attribute.
    """
    for key in hdf5_file.keys():
        if isinstance(hdf5_file[key], h5py.Group):
            assert_unit_and_description_present(
                hdf5_file[key], _prefix=_prefix + key + "/"
            )
        else:
            assert (
                "unit" in hdf5_file[key].attrs.keys()
            ), f"The dataset {_prefix}/{key} does not have a unit attribute."
            assert (
                "description" in hdf5_file[key].attrs.keys()
            ), f"The dataset {_prefix}/{key} does not have a description attribute."
