import json

import numpy as np
from tqdm import tqdm


def get_labels(fname):
    with open(fname) as f:
        data = json.load(f)

    coords = np.zeros((len(data['features']), 4))
    chips = np.zeros((len(data['features'])), dtype="object")
    classes = np.zeros((len(data['features'])))
    for i in tqdm(range(len(data['features']))):
        if data['features'][i]['properties']['bounds_imcoords'] != []:
            b_id = data['features'][i]['properties']['image_id']
            val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
            chips[i] = b_id
            classes[i] = data['features'][i]['properties']['type_id']
            if val.shape[0] != 4:
                print("Issues at %d!" % i)
            else:
                coords[i] = val
        else:
            chips[i] = 'None'

    return coords, chips, classes


def plotResults():
    import numpy as np
    import matplotlib.pyplot as plt
    results = np.loadtxt('printedResults.txt', usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).T
    x, y, w, h, conf, cls, total, AP, mAP, time = results[:,55000:]
    plt.plot(mAP)


path = '/Users/glennjocher/Downloads/DATA/xview/'
# path = ''
fname = path + 'xView_train.geojson'
coords, chips, classes = get_labels(fname)

uchips = np.unique(chips)
n = len(uchips)
shapes = np.zeros((n, 2))
stats = np.zeros((n, 6))
import cv2

for i, chip in enumerate(path + 'train_images/' + uchips):
    print(i)
    img = cv2.imread(chip)
    if img is not None:
        shapes[i] = img.shape[:2]
        for j in range(3):
            stats[i, j] = img[:, :, j].mean()
            stats[i, j + 3] = img[:, :, j].std()

import scipy.io

scipy.io.savemat('xview.mat', {'coords': coords, 'chips': chips, 'classes': classes, 'shapes': shapes, 'stats': stats,
                               'uchips': uchips})

## create train_labels folder in coco format
# nF = []  # number of features
# os.makedirs(path + 'train_labels/', exist_ok=True)
# for name in tqdm(np.unique(chips)):
#    rows = [i for i, x in enumerate(chips) if x == name]
#    nF.append(len(rows))
#    if any(rows):
#        with open(path + 'train_labels/' + name.replace('.tif', '.txt'), 'a') as file:
#            for i in rows:
#                file.write('%g %g %g %g %g\n' % (classes[i], *coords[i]))
