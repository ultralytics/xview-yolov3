import argparse
import time
import os

try:
    from torch.utils.data import DataLoader
except:  # required packaged not installed
    os.system('conda install -y numpy opencv pytorch')
    from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/train_images8', help='path to images')
parser.add_argument('--output_folder', type=str, default='data/xview_predictions', help='path to outputs')
parser.add_argument('--config_path', type=str, default='cfg/yolovx.cfg', help='path to model cfg file')
parser.add_argument('--weights_path', type=str, default='checkpoints/Adam_IOUfix__final_epoch_249_608.pt', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/xview.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.995, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.25, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=32 * 19, help='size of each image dimension')
parser.add_argument('--plot_flag', type=bool, default=True, help='plots predicted images if True')
opt = parser.parse_args()
print(opt)


def main(opt):
    os.system('rm -rf ' + opt.output_folder)
    os.makedirs(opt.output_folder, exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Set up model
    if not os.path.isfile(opt.weights_path):
        print('Network weights downloading. Please wait...\n')
        os.system('wget -c https://storage.googleapis.com/ultralytics/xvw1.pt')
        opt.weights_path = 'xvw1.pt'
    model = Darknet(opt.config_path, img_size=opt.img_size).to(device).eval()
    model.load_state_dict(torch.load(opt.weights_path, map_location=device.type))

    # Set dataloader
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    prev_time = time.time()
    print('\nRunning inference:')
    for batch_i, (img_paths, img) in enumerate(dataloader):
        # Configure input
        img = img.type(Tensor)

        # Get detections
        with torch.no_grad():
            detections = non_max_suppression(model(img), opt.conf_thres, opt.nms_thres)

        # Log progress
        print('Batch %d... (Done %.3fs)' % (batch_i, time.time() - prev_time))
        prev_time = time.time()

        imgs.extend(img_paths)
        img_detections.extend(detections)


    # Bounding-box colors
    color_list = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print("image %g: '%s'" % (img_i, path))

        # read image
        img = cv2.imread(path)

        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_classes = detections[:, -1].cpu().unique()
            bbox_colors = random.sample(color_list, len(unique_classes))

            # write results to .txt file
            results_path = os.path.join(opt.output_folder, path.split('/')[-1])
            if os.path.isfile(results_path + '.txt'):
                os.remove(results_path + '.txt')

            with open(results_path + '.txt', 'a') as file:
                for i in unique_classes:
                    n = (detections[:, -1].cpu() == i).sum()
                    print('%g %ss' % (n, classes[int(i)]))

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    # Rescale coordinates to original dimensions
                    box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                    box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                    y1 = (((y1 - pad_y // 2) / unpad_h) * img.shape[0] - 3).round().item()
                    x1 = (((x1 - pad_x // 2) / unpad_w) * img.shape[1]).round().item()
                    x2 = (x1 + box_w - 3).round().item()
                    y2 = (y1 + box_h).round().item()

                    # write to file
                    file.write(('%g %g %g %g %g %g \n') % (x1, y1, x2, y2, xview_indices2classes(int(cls_pred)), conf))

                    if opt.plot_flag:
                        # Add the bbox to the plot
                        # label = classes[int(cls_pred)]
                        color = bbox_colors[int(np.where(unique_classes == int(cls_pred))[0])]
                        plot_one_box([x1, y1, x2, y2], img, color=color, line_thickness=2)

            if opt.plot_flag:
                # Save generated image with detections
                cv2.imwrite(results_path, img)


if __name__ == '__main__':
    main(opt)
