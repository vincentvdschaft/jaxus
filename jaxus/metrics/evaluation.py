from jaxus.containers import Image
from jaxus.metrics.gcnr import gcnr_disk_annulus
import numpy as np


def image_measure_gcnr_disk_annulus(
    image: Image, disk_center, disk_r, annulus_offset, annulus_width
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
    data = image.data
    extent = image.extent
    gcnr = gcnr_disk_annulus(
        data,
        extent=extent,
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
