import argparse
from pathlib import Path

from jaxus import interpret_range, log, reduce_hdf5_file

parser = argparse.ArgumentParser()
parser.add_argument("path", type=Path)
parser.add_argument("output_path", type=Path, default=None, nargs="?")
parser.add_argument("--frames", type=str, default="all")
parser.add_argument("--transmits", type=str, default="all")


args = parser.parse_args()


path = Path(args.path)
output_path = (
    Path(args.output_path)
    if args.output_path is not None
    else path.parent / f"{path.stem}_reduced.hdf5"
)

if not path.exists():
    raise FileNotFoundError(f"Path {path} does not exist.")

frames_str = args.frames
transmits_str = args.transmits

reduce_hdf5_file(path, output_path, frames_str, transmits_str)
