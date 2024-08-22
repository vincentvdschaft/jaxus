"""Script to convert all png files in a folder to mp4 video."""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
from PIL import Image

# pylint: disable=C0413
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import jaxus.utils.log as log

# Parse command line arguments
# --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "directory",
    type=str,
    default="config/custom/build_h.yaml",
    help="Path to the config file.",
)

parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Frames per second.",
)

args = parser.parse_args()


def pngs_to_gif(folder_path, output_filename, duration=500):
    """
    Convert all PNG images in the specified folder into a single GIF.

    Parameters:
    `folder_path` (`str`): The path to the folder containing PNG images.
    `output_filename` (`str`): The filename for the output GIF.
    `duration` (`int`): Duration of each frame in the GIF in milliseconds.
    """
    images = []
    # Loop through all files in the folder
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))
    # Sort the images by name
    image_paths.sort()
    for image_path in image_paths:
        images.append(Image.open(image_path))

    log.info(f"Found {log.yellow(len(images))} images in {log.yellow(folder_path)}")
    print(image_paths)

    # Save the images as a GIF
    images[0].save(
        output_filename,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration,
        loop=0,
    )


# Example usage

folder_path = Path(args.directory)
output_gif_path = folder_path / "video.gif"
if not folder_path.exists():
    log.critical(f"Directory {folder_path} does not exist. Aborting.")


if args.fps < 1 or args.fps > 60:
    log.critical(f"FPS must be between 1 and 60. Got {args.fps}. Aborting.")

pngs_to_gif(str(folder_path), str(output_gif_path), duration=1000 * int(1 / args.fps))

log.succes("Gif generated!")
log.info(f"Gif saved to {log.yellow(output_gif_path)}")
