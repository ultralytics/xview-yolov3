#!/bin/bash

# example use case:
# run.sh /absolute/path/to/a/test/image.tif /absolute/path/to/output/directory
# bash run.sh /Users/glennjocher/Documents/PyCharmProjects/yolov3/data/xview_samples/2031.tif /Users/glennjocher/Documents/PyCharmProjects/yolov3/data/xview_predictions

python detect.py --image_folder $1 --output_folder $2