#!/bin/bash


uv run python scripts/beamform.

out_dir="pala"
mkdir -p $out_dir
index=0
for frame in {32..95};
do
    for transmit in {0..4};
    do
        echo "Processing frame $frame, transmit $transmit"
        # Define save path with zero padding
        save_path="$out_dir/im_$(printf "%03d" $index).hdf5"


        /home/vincent/uv/jax/bin/python scripts/beamform_file.py /home/vincent/3-data/pala/RF_usbmd/RF_024.hdf5 --frames $frame --transmits $transmit --extent -7 7 4 13 --no-show --save-path $save_path --fnumber 0.8 --dynamic-range 35
        index=$((index+1))
    done
done