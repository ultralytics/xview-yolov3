import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *

targets_path = 'utils/targets_c60.mat'

parser = argparse.ArgumentParser()
# Get data configuration
if platform == 'darwin':  # macos
    parser.add_argument('-image_folder', type=str, default='/Users/glennjocher/Downloads/DATA/xview/train_images/5.bmp',
                        help='path to images')
    parser.add_argument('-output_folder', type=str, default='./output_xview', help='path to outputs')
    cuda = torch.cuda.is_available()
else:  # gcp
    # cd yolo && python3 detect.py -img_size 1632
    parser.add_argument('-image_folder', type=str, default='../train_images/5.bmp', help='path to images')
    parser.add_argument('-output_folder', type=str, default='../output', help='path to outputs')
    cuda = False

parser.add_argument('-cfg', type=str, default='cfg/c60_a30.cfg', help='cfg file path')
parser.add_argument('-class_path', type=str, default='data/xview.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.99, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
parser.add_argument('-batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('-img_size', type=int, default=32 * 51, help='size of each image dimension')
parser.add_argument('-plot_flag', type=bool, default=True, help='plots predicted images if True')
opt = parser.parse_args()
print(opt)

# @profile
def detect(opt):
    os.system('rm -rf ' + opt.output_folder)
    os.system('rm -rf ' + opt.output_folder + '_img')
    os.makedirs(opt.output_folder, exist_ok=True)
    os.makedirs(opt.output_folder + '_img', exist_ok=True)
    device = torch.device('cuda:0' if cuda else 'cpu')

    # Load model 1
    model = Darknet(opt.cfg, opt.img_size)
    checkpoint = torch.load('checkpoints/latest.pt', map_location='cpu')
    # model.load_state_dict(checkpoint) #['model'])
    # del checkpoint

    current = model.state_dict()
    saved = checkpoint['model']
    # 1. filter out unnecessary keys
    saved = {k: v for k, v in saved.items() if ((k in current) and (current[k].shape == v.shape))}
    # 2. overwrite entries in the existing state dict
    current.update(saved)
    # 3. load the new state dict
    model.load_state_dict(current)
    del checkpoint, current, saved
    model.to(device).eval()

    # Load model 2
    try:
        model2 = ConvNetb()
        checkpoint = torch.load('/Users/glennjocher/Documents/PyCharmProjects/mnist/best64_4layer3.pt', map_location='cpu')
        model2.load_state_dict(checkpoint['model'])
        model2.to(device).eval()
        del checkpoint
    except:
        model2 = []

    # Set Dataloader
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    dataloader = ImageFolder(opt.image_folder, batch_size=opt.batch_size, img_size=opt.img_size)

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    prev_time = time.time()
    detections = None
    mat_priors = scipy.io.loadmat(targets_path)
    for batch_i, (img_paths, img) in enumerate(dataloader):
        print('\n', batch_i, img.shape, end=' ')

        preds = []
        length = opt.img_size
        ni = int(math.ceil(img.shape[1] / length))  # up-down
        nj = int(math.ceil(img.shape[2] / length))  # left-right
        for i in range(ni):  # for i in range(ni - 1):
            print('row %g/%g: ' % (i, ni), end='')

            for j in range(nj):  # for j in range(nj if i==0 else nj - 1):
                print('%g ' % j, end='', flush=True)

                # forward scan
                y2 = min((i + 1) * length, img.shape[1])
                y1 = y2 - length
                x2 = min((j + 1) * length, img.shape[2])
                x1 = x2 - length
                chip = img[:, y1:y2, x1:x2]

                # Get detections
                chip = torch.from_numpy(chip).unsqueeze(0).to(device)
                with torch.no_grad():
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

                # # backward scan
                # y2 = max(img.shape[1] - i * length, length)
                # y1 = y2 - length
                # x2 = max(img.shape[2] - j * length, length)
                # x1 = x2 - length
                # chip = img[:, y1:y2, x1:x2]
                #
                # # plot
                # #import matplotlib.pyplot as plt
                # #plt.subplot(ni, nj, i * nj + j + 1).imshow(chip[1])
                # # plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
                #
                # # Get detections
                # chip = torch.from_numpy(chip).unsqueeze(0).to(device)
                # with torch.no_grad():
                #     pred = model2(chip)
                #     pred = pred[pred[:, :, 4] > opt.conf_thres]
                #
                #     # if (j < nj) & (len(pred) > 0):
                #     #     pred = pred[(pred[:, 0] - pred[:, 2] / 2 > 2)]  # near left border
                #     # if (j > 0) & (len(pred) > 0):
                #     #     pred = pred[(pred[:, 0] + pred[:, 2] / 2 < 606)]  # near right border
                #     # if (i < ni) & (len(pred) > 0):
                #     #     pred = pred[(pred[:, 1] - pred[:, 3] / 2 > 2)]  # near top border
                #     # if (i > 0) & (len(pred) > 0):
                #     #     pred = pred[(pred[:, 1] + pred[:, 3] / 2 < 606)]  # near bottom border
                #
                #     if len(pred) > 0:
                #         pred[:, 0] += x1
                #         pred[:, 1] += y1
                #         preds.append(pred.unsqueeze(0))

        if len(preds) > 0:
            detections = non_max_suppression(torch.cat(preds, 1), opt.conf_thres, opt.nms_thres, mat_priors, img,
                                             model2, device)
            img_detections.extend(detections)
            imgs.extend(img_paths)

        print('Batch %d... (Done %.3fs)' % (batch_i, time.time() - prev_time))
        prev_time = time.time()

    # Bounding-box colors
    color_list = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    if (len(img_detections) == 0):  # | (img_detections[0] is None):
        return

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("image %g: '%s'" % (img_i, path))

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
            results_path = os.path.join(opt.output_folder, path.split('/')[-1])
            if os.path.isfile(results_path + '.txt'):
                os.remove(results_path + '.txt')

            results_img_path = os.path.join(opt.output_folder + '_img', path.split('/')[-1])
            with open(results_path.replace('.bmp', '.tif') + '.txt', 'a') as file:
                for i in unique_classes:
                    n = (detections[:, -1].cpu() == i).sum()
                    print('%g %ss' % (n, classes[int(i)]))

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
                    file.write(('%g %g %g %g %g %g \n') % (x1, y1, x2, y2, xvc, cls_conf * conf))

                    if opt.plot_flag:
                        # Add the bbox to the plot
                        label = '%s %.2f' % (classes[int(cls_pred)], cls_conf) if cls_conf > 0.05 else None
                        color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]
                        plot_one_box([x1, y1, x2, y2], img, label=label, color=color, line_thickness=1)

            if opt.plot_flag:
                # Save generated image with detections
                cv2.imwrite(results_img_path.replace('.bmp', '.jpg'), img)

    if opt.plot_flag:
        from scoring import score
        score.score(opt.output_folder + '/', '/Users/glennjocher/Downloads/DATA/xview/xView_train.geojson', '.')


class ConvNetb(nn.Module):
    def __init__(self, num_classes=60):
        super(ConvNetb, self).__init__()
        n = 64  # initial convolution size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, n, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(n, n * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n * 2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(n * 2, n * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n * 4),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(n * 4, n * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n * 8),
            nn.ReLU())
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(n * 8, n * 16, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(n * 16),
        #     nn.ReLU())
        # self.fc = nn.Linear(65536, num_classes)  # 64 pixels, 3 layer, 64 filters
        # self.fc = nn.Linear(32768, num_classes)  # 64 pixels, 3 layer, 32 filters
        self.fc = nn.Linear(32768, num_classes)  # 64 pixels, 4 layer, 64 filters

    def forward(self, x):  # x.size() = [512, 1, 28, 28]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        # x, _, _ = normalize(x,1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    torch.cuda.empty_cache()
    detect(opt)
    torch.cuda.empty_cache()

