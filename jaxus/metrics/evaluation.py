from jaxus.containers import Image
from jaxus.metrics.gcnr import gcnr_disk_annulus
import numpy as np


def image_measure_gcnr_disk_annulus(
    image: Image, disk_center, disk_radius, annulus_radius0, annulus_radius1
):
    data = image.data
    extent = image.extent
    gcnr = gcnr_disk_annulus(
        data,
        extent=extent,
        disk_center=disk_center,
        disk_r=disk_radius,
        annul_r0=annulus_radius0,
        annul_r1=annulus_radius1,
    )

    gcnr_metadata = {
        "gcnr_type": "disk",
        "gcnr_value": gcnr,
        "disk_center": disk_center,
        "disk_radius": disk_radius,
        "annulus_radius0": annulus_radius0,
        "annulus_radius1": annulus_radius1,
    }
    image.append_metadata(key="gcnr", value=gcnr_metadata)

    return image
