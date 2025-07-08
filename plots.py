from pathlib import Path

import matplotlib.pyplot as plt
from imagelib import Image
from plotlib import *

from jaxus import plot_beamformed

use_style(STYLE_DARK)

source_dir = Path("sos_images")
paths = list(source_dir.glob("*.hdf5"))

for path in paths:
    print(path)
    image = Image.load(path)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    plot_beamformed(ax, image, title="Beamformed Image")

    image_path = path.parent / "images" / Path(path.stem + "_image").with_suffix(".png")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)

    plot_path = path.parent / "plots" / Path(path.stem + "_plot").with_suffix(".png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
