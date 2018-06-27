#!/usr/bin/env bash

# reattach screen
# screen -r

# detach from screen
# ctrl + a + d

# clone repo
#cd ..
sudo rm -rf yolo
git clone https://github.com/ultralytics/yolo

# do training
cd yolo
python3 train.py -epochs 5000

# move checkpoint to fuse drive
#mv yolov3/checkpoints/epoch0.pt drive


# get xview training data
wget -O train_images.tgz 'https://d307kc0mrhucc3.cloudfront.net/train_images.tgz?Expires=1530124049&Signature=JrQoxipmsETvb7eQHCfDFUO-QEHJGAayUv0i-ParmS-1hn7hl9D~bzGuHWG82imEbZSLUARTtm0wOJ7EmYMGmG5PtLKz9H5qi6DjoSUuFc13NQ-~6yUhE~NfPaTnehUdUMCa3On2wl1h1ZtRG~0Jq1P-AJbpe~oQxbyBrs1KccaMa7FK4F4oMM6sMnNgoXx8-3O77kYw~uOpTMFmTaQdHln6EztW0Lx17i57kK3ogbSUpXgaUTqjHCRA1dWIl7PY1ngQnLslkLhZqmKcaL-BvWf0ZGjHxCDQBpnUjIlvMu5NasegkwD9Jjc0ClgTxsttSkmbapVqaVC8peR0pO619Q__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'
tar -xvzf train_images.tgz
sudo rm -rf train_images/._*
# lastly convert each .tif to a .bmp for faster loading in cv2
