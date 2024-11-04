#!/bin/bash

pip install -e ./baselines/pie-g/

if [ ! -d "./baselines/data/places365_standard" ]; then
    wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar -P ./baselines/data/
    tar -xvf ./baselines/data/places365standard_easyformat.tar -C baselines/data/
    rm ./baselines/data/places365standard_easyformat.tar
else
    echo "The folder 'places365_standard' already exists. Skipping download and extraction."
fi