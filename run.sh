#!/bin/bash

# WARNING: Run 'chmod +x run.sh' BEFORE BUILDING DOCKER CONTAINER

# example:
# bash run.sh ./1047.tif ./output
# bash run.sh /Users/glennjocher/Documents/PyCharmProjects/xview/samples/5.tif /Users/glennjocher/Documents/PyCharmProjects/xview
# docker tag friendlyhello ultralytics/xview:v1
# time docker run -it --memory=4g --cpus=1 ultralytics/xview:submission1_v1 bash -c './run.sh /samples/5.tif /tmp && cat /tmp/5.tif.txt'

python3 detect.py -image_folder $1 -output_folder $2
