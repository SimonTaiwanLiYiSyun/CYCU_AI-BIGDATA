#!/bin/bash

rm -rf test/ train/ v_data/
bash /home/Desktop/example1/data.sh; python --version
python /home/Desktop/example1/Image_Classification_Tranning.py
python /home/Desktop/example1/Image_Classification_Test.py
rm model_saved.h5;