# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *

targets_path = "utils/targets_c60.mat"

parser = argparse.ArgumentParser()
# Get data configuration
if platform == "darwin":  # macos
    parser.add_argument("-image_folder", type=str, default="/Users/glennjocher/Downloads/DATA/xview/train_images/5.tif")
    parser.add_argument("-output_folder", type=str, default="./output_xview", help="path to outputs")
else:  # gcp
    # cd yolo && python3 detect.py -secondary_classifier 1
    parser.add_argument("-image_folder", type=str, default="../train_images/5.tif", help="path to images")
    parser.add_argument("-output_folder", type=str, default="../output", help="path to outputs")
cuda = False  # torch.cuda.is_available()
parser.add_argument("-plot_flag", type=bool, default=True)
parser.add_argument("-secondary_classifier", type=bool, default=False)
parser.add_argument("-cfg", type=str, default="cfg/c60_a30symmetric.cfg", help="cfg file path")
parser.add_argument("-class_path", type=str, default="data/xview.names", help="path to class label file")
parser.add_argument("-conf_thres", type=float, default=0.99, help="object confidence threshold")
parser.add_argument("-nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
parser.add_argument("-batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("-img_size", type=int, default=32 * 51, help="size of each image dimension")
opt = parser.parse_args()
print(opt)


def detect(opt):
    """Detects objects in images using Darknet model, optionally uses a secondary classifier, and performs NMS."""
    if opt.plot_flag:
        os.system(f"rm -rf {opt.output_folder}_img")
        os.makedirs(f"{opt.output_folder}_img", exist_ok=True)
    os.system(f"rm -rf {opt.output_folder}")
    os.makedirs(opt.output_folder, exist_ok=True)
    device = torch.device("cuda:0" if cuda else "cpu")

    # Load model 1
    model = Darknet(opt.cfg, opt.img_size)
    checkpoint = torch.load("weights/xview_best_lite.pt", map_location="cpu")

    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    del checkpoint

    # current = model.state_dict()
    # saved = checkpoint['model']
    # # 1. filter out unnecessary keys
    # saved = {k: v for k, v in saved.items() if ((k in current) and (current[k].shape == v.shape))}
    # # 2. overwrite entries in the existing state dict
    # current.update(saved)
    # # 3. load the new state dict
    # model.load_state_dict(current)
    # model.to(device).eval()
    # del checkpoint, current, saved

    # Load model 2
    if opt.secondary_classifier:
        model2 = ConvNetb()
        checkpoint = torch.load("weights/classifier.pt", map_location="cpu")

        model2.load_state_dict(checkpoint["model"])
        model2.to(device).eval()
        del checkpoint

        # current = model2.state_dict()
        # saved = checkpoint['model']
        # # 1. filter out unnecessary keys
        # saved = {k: v for k, v in saved.items() if ((k in current) and (current[k].shape == v.shape))}
        # # 2. overwrite entries in the existing state dict
        # current.update(saved)
        # # 3. load the new state dict
        # model2.load_state_dict(current)
        # model2.to(device).eval()
        # del checkpoint, current, saved
    else:
        model2 = None

    # Set Dataloader
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    dataloader = ImageFolder(opt.image_folder, batch_size=opt.batch_size, img_size=opt.img_size)

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    prev_time = time.time()
    detections = None
    mat_priors = scipy.io.loadmat(targets_path)
    for batch_i, (img_paths, img) in enumerate(dataloader):
        print("\n", batch_i, img.shape, end=" ")

        np.ascontiguousarray(np.flip(img, axis=1))
        np.ascontiguousarray(np.flip(img, axis=2))

        preds = []
        length = opt.img_size
        ni = int(math.ceil(img.shape[1] / length))  # up-down
        nj = int(math.ceil(img.shape[2] / length))  # left-right
        for i in range(ni):  # for i in range(ni - 1):
            print(f"row {i:g}/{ni:g}: ", end="")

            for j in range(nj):  # for j in range(nj if i==0 else nj - 1):
                print(f"{j:g} ", end="", flush=True)

                # forward scan
                y2 = min((i + 1) * length, img.shape[1])
                y1 = y2 - length
                x2 = min((j + 1) * length, img.shape[2])
                x1 = x2 - length

                # Get detections
                with torch.no_grad():
                    # Normal orientation
                    chip = torch.from_numpy(img[:, y1:y2, x1:x2]).unsqueeze(0).to(device)
                    pred = model(chip)
                    pred = pred[pred[:, :, 4] > opt.conf_thres]
                    # if (j > 0) & (len(pred) > 0):
                    #     pred = pred[(pred[:, 0] - pred[:, 2] / 2 > 2)]  # near left border
                    # if (j < nj) & (len(pred) > 0):
                    #     pred = pred[(pred[:, 0] + pred[:, 2] / 2 < 606)]  # near right border
                    # if (i > 0) & (len(pred) > 0):
                    #     pred = pred[(pred[:, 1] - pred[:, 3] / 2 > 2)]  # near top border
                    # if (i < ni) & (len(pred) > 0):
                    #     pred = pred[(pred[:, 1] + pred[:, 3] / 2 < 606)]  # near bottom border
                    if len(pred) > 0:
                        pred[:, 0] += x1
                        pred[:, 1] += y1
                        preds.append(pred.unsqueeze(0))

                    # # Flipped Up-Down
                    # chip = torch.from_numpy(img_ud[:, y1:y2, x1:x2]).unsqueeze(0).to(device)
                    # pred = model(chip)
                    # pred = pred[pred[:, :, 4] > opt.conf_thres]
                    # if len(pred) > 0:
                    #     pred[:, 0] += x1
                    #     pred[:, 1] = img.shape[1] - (pred[:, 1] + y1)
                    #     preds.append(pred.unsqueeze(0))

                    # # Flipped Left-Right
                    # chip = torch.from_numpy(img_lr[:, y1:y2, x1:x2]).unsqueeze(0).to(device)
                    # pred = model(chip)
                    # pred = pred[pred[:, :, 4] > opt.conf_thres]
                    # if len(pred) > 0:
                    #     pred[:, 0] = img.shape[2] - (pred[:, 0] + x1)
                    #     pred[:, 1] += y1
                    #     preds.append(pred.unsqueeze(0))

        if preds:
            detections = non_max_suppression(
                torch.cat(preds, 1), opt.conf_thres, opt.nms_thres, mat_priors, img, model2, device
            )
            img_detections.extend(detections)
            imgs.extend(img_paths)

        print("Batch %d... (Done %.3fs)" % (batch_i, time.time() - prev_time))
        prev_time = time.time()

    # Bounding-box colors
    color_list = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    if not img_detections:
        return

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print(f"image {img_i:g}: '{path}'")

        if opt.plot_flag:
            img = cv2.imread(path)

        # # The amount of padding that was added
        # pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
        # pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
        # # Image height and width after padding is removed
        # unpad_h = opt.img_size - pad_y
        # unpad_w = opt.img_size - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_classes = detections[:, -1].cpu().unique()
            bbox_colors = random.sample(color_list, len(unique_classes))

            # write results to .txt file
            results_path = os.path.join(opt.output_folder, path.split("/")[-1])
            if os.path.isfile(f"{results_path}.txt"):
                os.remove(f"{results_path}.txt")

            results_img_path = os.path.join(f"{opt.output_folder}_img", path.split("/")[-1])
            with open(results_path.replace(".bmp", ".tif") + ".txt", "a") as file:
                for i in unique_classes:
                    n = (detections[:, -1].cpu() == i).sum()
                    print(f"{n:g} {classes[int(i)]}s")

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    # Rescale coordinates to original dimensions
                    # box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                    # box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                    # y1 = (((y1 - pad_y // 2) / unpad_h) * img.shape[0]).round().item()
                    # x1 = (((x1 - pad_x // 2) / unpad_w) * img.shape[1]).round().item()
                    # x2 = (x1 + box_w).round().item()
                    # y2 = (y1 + box_h).round().item()
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)

                    # write to file
                    xvc = xview_indices2classes(int(cls_pred))  # xview class
                    # if (xvc != 21) & (xvc != 72):
                    file.write(f"{x1:g} {y1:g} {x2:g} {y2:g} {xvc:g} {cls_conf * conf:g} \n")

                    if opt.plot_flag:
                        # Add the bbox to the plot
                        label = f"{classes[int(cls_pred)]} {cls_conf:.2f}" if cls_conf > 0.05 else None
                        color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]
                        plot_one_box([x1, y1, x2, y2], img, label=label, color=color, line_thickness=1)

            if opt.plot_flag:
                # Save generated image with detections
                cv2.imwrite(results_img_path.replace(".bmp", ".jpg").replace(".tif", ".jpg"), img)

    if opt.plot_flag:
        from scoring import score

        score.score(
            f"{opt.output_folder}/",
            "/Users/glennjocher/Downloads/DATA/xview/xView_train.geojson",
            ".",
        )


class ConvNetb(nn.Module):
    """Defines a convolutional neural network model with configurable classes and multiple convolutional layers."""

    def __init__(self, num_classes=60):
        """Initializes a ConvNetb model with configurable number of classes, defaulting to 60, and a series of
        convolutional layers.
        """
        super().__init__()
        n = 64  # initial convolution size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, n, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(n), nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(n, n * 2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(n * 2), nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(n * 2, n * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 4),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(n * 4, n * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 8),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(n * 8, n * 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n * 16),
            nn.LeakyReLU(),
        )
        # self.layer6 = nn.Sequential(
        #     nn.Conv2d(n * 16, n * 32, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(n * 32),
        #     nn.LeakyReLU())

        # self.fc = nn.Linear(int(8192), num_classes)  # 64 pixels, 4 layer, 64 filters
        self.fully_conv = nn.Conv2d(n * 16, 60, kernel_size=4, stride=1, padding=0, bias=True)

    def forward(self, x):  # 500 x 1 x 64 x 64
        """Processes input through 5 layers and a fully connected layer, returning squeezed output; expects input shape
        500x1x64x64.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = self.layer6(x)
        # x = self.fc(x.reshape(x.size(0), -1))
        x = self.fully_conv(x)
        return x.squeeze()  # 500 x 60


if __name__ == "__main__":
    torch.cuda.empty_cache()
    detect(opt)
    torch.cuda.empty_cache()
