import json
import os

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


#path = '/Users/glennjocher/Downloads/DATA/xview/'
path = ''
fname = path + 'xView_train.geojson'
coords, chips, classes = get_labels(fname)

# create train_labels folder in coco format
nF = [] # number of features
os.makedirs(path + 'train_labels/', exist_ok=True)
for name in tqdm(np.unique(chips)):
    rows = [i for i, x in enumerate(chips) if x == name]
    nF.append(len(rows))
    if any(rows):
        with open(path + 'train_labels/' + name.replace('.tif', '.txt'), 'a') as file:
            for i in rows:
                file.write('%g %g %g %g %g\n' % (classes[i], *coords[i]))

