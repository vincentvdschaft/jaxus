from jaxus.containers import Image
from jaxus.metrics.gcnr import gcnr_disk_annulus
from jaxus.metrics.fwhm import fwhm_image, correct_fwhm_point, _sample_line
import numpy as np
from copy import deepcopy


def image_measure_gcnr_disk_annulus(
    image: Image, disk_center, disk_r, annulus_offset, annulus_width, return_copy=False
):
    """Computes the gCNR between a disk and a surrounding annulus and adds the result
    to the image metadata.

    Parameters
    ----------
    image : Image
        The image to compute the GCNR on.
    disk_center : tuple
        The position of the disk.
    disk_r : float
        The radius of the disk.
    annulus_offset : float
        The space between disk and annulus.
    annulus_width : float
        The width of the annulus.

    Returns
    -------
    image : Image
        The image with the gCNR value added to the metadata.
    """
    if return_copy:
        image = deepcopy(image)

    gcnr = gcnr_disk_annulus(
        image=image,
        disk_center=disk_center,
        disk_r=disk_r,
        annulus_offset=annulus_offset,
        annulus_width=annulus_width,
    )

    gcnr_metadata = {
        "gcnr_type": "disk_annulus",
        "gcnr_value": gcnr,
        "disk_center": disk_center,
        "disk_r": disk_r,
        "annulus_offset": annulus_offset,
        "annulus_width": annulus_width,
    }
    image.append_metadata(key="gcnr", value=gcnr_metadata)

    return image


def image_measure_fwhm(
    image: Image,
    position,
    axial_direction,
    max_offset,
    correct_position=False,
    max_correction_distance=1e-3,
    n_samples=100,
    return_copy=False,
):
    """Computes the FWHM of a line profile in the image and adds the result to the image metadata.

    Parameters
    ----------
    image : Image
        The image to compute the FWHM on.
    position : tuple
        The position of the line profile.
    axial_direction : tuple
        The axial direction of the line profile.
    max_offset : float
        The maximum offset from the position to sample the line profile.
    correct_position : bool
        Whether to correct the position of the FWHM point.
    max_correction_distance : float
        The maximum distance to search for the FWHM point.

    Returns
    -------
    image : Image
        The image with the FWHM value added to the metadata.
    """
    if return_copy:
        image = deepcopy(image)

    if correct_position:
        corrected_position = correct_fwhm_point(
            image, position, max_diff=max_correction_distance
        )
        position = corrected_position

    # Normalize axial direction
    axial_direction = np.array(axial_direction)
    axial_direction /= np.linalg.norm(axial_direction)

    lateral_direction = np.array([-axial_direction[1], axial_direction[0]])

    fwhm_value_axial = fwhm_image(
        image,
        position,
        axial_direction,
        max_offset=max_offset,
    )
    values_axial, positions_axial = _sample_line(
        image=image.data,
        extent=image.extent,
        position=position,
        vec=axial_direction,
        max_offset=max_offset,
        n_samples=n_samples,
    )

    fwhm_value_lateral = fwhm_image(
        image,
        position,
        lateral_direction,
        max_offset=max_offset,
    )
    values_lateral, positions_lateral = _sample_line(
        image=image.data,
        extent=image.extent,
        position=position,
        vec=lateral_direction,
        max_offset=max_offset,
        n_samples=n_samples,
    )

    fwhm_metadata = {
        "fwhm_value_axial": fwhm_value_axial,
        "fwhm_value_lateral": fwhm_value_lateral,
        "position": position,
        "axial_direction": axial_direction,
        "max_offset": max_offset,
        "n_samples": n_samples,
        "positions_axial": positions_axial,
        "values_axial": values_axial,
        "positions_lateral": positions_lateral,
        "values_lateral": values_lateral,
    }
    image.append_metadata(key="fwhm", value=fwhm_metadata)

    return image
