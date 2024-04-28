import json

import numpy as np
from tqdm import tqdm


def xview_class2name(classes):
    """Converts numerical class IDs to their corresponding names using 'data/xview.names'."""
    with open("data/xview.names", "r") as f:
        x = f.readlines()
    return x[classes].replace("\n", "")


def get_labels(fname="xView_train.geojson"):  # from utils.utils_xview import get_labels; get_labels()
    """Parses 'xView_train.geojson' to extract object coordinates, chip IDs, and class IDs as numpy arrays."""
    with open(fname) as f:
        data = json.load(f)

    n = len(data["features"])
    coords = np.zeros((n, 4))
    chips = np.zeros(n, dtype="object")
    classes = np.zeros(n)
    for i in tqdm(range(n)):
        if data["features"][i]["properties"]["bounds_imcoords"] != []:
            b_id = data["features"][i]["properties"]["image_id"]
            val = np.array([int(num) for num in data["features"][i]["properties"]["bounds_imcoords"].split(",")])
            chips[i] = b_id
            classes[i] = data["features"][i]["properties"]["type_id"]
            if val.shape[0] != 4:
                print("Issues at %d!" % i)
            else:
                coords[i] = val
        else:
            chips[i] = "None"

    return coords, chips, classes


def create_mat_file():
    """Saves geojson data as a MATLAB (.mat) file, enriching it with image statistics and shapes."""
    import scipy.io
    import cv2
    import numpy as np

    path = "/Users/glennjocher/Downloads/DATA/xview/"
    coords, chips, classes = get_labels(path + "xView_train.geojson")

    uchips = np.unique(chips)
    n = len(uchips)
    shapes = np.zeros((n, 2))
    stats = np.zeros((n, 12))  # BGR mean and std, HSV mean and std
    for i, chip in enumerate(path + "train_images/" + uchips):
        print(i)
        img = cv2.imread(chip.replace(".tif", ".bmp"))
        if img is not None:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            shapes[i] = img.shape[:2]
            for j in range(3):
                stats[i, j + 0] = img[:, :, j].astype(np.float32).mean()
                stats[i, j + 3] = img[:, :, j].astype(np.float32).std()
                stats[i, j + 6] = hsv[:, :, j].mean()
                stats[i, j + 9] = hsv[:, :, j].std()

    scipy.io.savemat(
        "xview.mat",
        {"coords": coords, "chips": chips, "classes": classes, "shapes": shapes, "stats": stats, "uchips": uchips},
    )
