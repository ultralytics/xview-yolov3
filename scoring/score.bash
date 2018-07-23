#!/bin/bash

set -exuo pipefail

export input_repo=$(ls /pfs -1 | grep -v out | grep -v labels)

export groundtruth=$(ls /pfs/validate_labels -1)
groundtruth="/pfs/validate_labels/$groundtruth"

export userID=`echo $input_repo | cut -f 1 -d "_"`
echo "Scoring userID=$userID"

timestamp=`date +%F:%T`
mkdir -p /pfs/out/$timestamp

# Note ... score.py needs the trailing slash on the input path
python score.py /pfs/$input_repo/ $groundtruth --output /pfs/out/$timestamp

