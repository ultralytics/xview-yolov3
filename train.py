import argparse
import time
from sys import platform

from torch.utils.data import DataLoader

from detect import detect
from models import *
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *

run_name = 'anchors18'
parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('-image_folder', type=str, default='data/train_images', help='path to images')
parser.add_argument('-output_folder', type=str, default='data/xview_predictions', help='path to outputs')
parser.add_argument('-batch_size', type=int, default=8, help='size of each image batch')
parser.add_argument('-config_path', type=str, default='cfg/yolovx_18.cfg', help='path to model cfg file')
parser.add_argument('-weights_path', type=str, default='checkpoints/test_final_epoch_249_416.pt',
                    help='path to weights file')
parser.add_argument('-class_path', type=str, default='data/xview.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.9, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('-n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('-img_size', type=int, default=32 * 13, help='size of each image dimension')
parser.add_argument('-checkpoint_interval', type=int, default=5000, help='interval between saving model weights')
parser.add_argument('-checkpoint_dir', type=str, default='checkpoints', help='directory for saving model checkpoints')
parser.add_argument('-plot_flag', type=bool, default=True, help='plots predicted images if True')
opt = parser.parse_args()
print(opt)


# @profile
def main(opt):
    os.makedirs('checkpoints', exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    torch.manual_seed(1)
    random.seed(1)

    # Get data configuration
    if platform == 'darwin':  # macos
        train_path = '/Users/glennjocher/Downloads/DATA/xview/'
    else:
        train_path = '../'

    # Get hyper parameters
    hyperparams = parse_model_config(opt.config_path)[0]
    lr = float(hyperparams['learning_rate'])
    momentum = float(hyperparams['momentum'])
    decay = float(hyperparams['decay'])
    burn_in = int(hyperparams['burn_in'])

    # Initiate model
    model = Darknet(opt.config_path, opt.img_size).to(device).train()

    # Get dataloader
    dataloader = DataLoader(ListDataset_xview(train_path, opt.img_size),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    # optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=decay, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.5)

    # reload saved optimizer state
    resume_training = False
    if resume_training:
        model.load_state_dict(torch.load(opt.weights_path, map_location=device.type))
        # optimizer.load_state_dict(torch.load('optim.pth'))
        # optimizer.state = defaultdict(dict, optimizer.state)
    else:
        model.apply(weights_init_normal)  # initialize with random weights

    t0 = time.time()
    print('%10s' * 14 % (
        'Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'precision', 'recall', 'p_mu', 'r_mu', 'time'))
    for epoch in range(opt.epochs):
        # scheduler.step()
        t1 = time.time()
        epochp, epochr = 0, 0

        for batch_i, (impath, imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            for _ in range(1):
                loss = model(imgs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epochp = (epochp * batch_i + model.precision) / (batch_i + 1)
                epochr = (epochr * batch_i + model.recall) / (batch_i + 1)

                s = ('%10s%10s' + '%10.3g' * 12) % (
                    '%g/%g' % (epoch, opt.epochs - 1), '%g/%g' % (batch_i, len(dataloader) - 1), model.losses['x'],
                    model.losses['y'], model.losses['w'], model.losses['h'], model.losses['conf'], model.losses['cls'],
                    model.losses['loss'], model.precision, model.recall, epochp, epochr, time.time() - t1)
                print(s)
                with open('printedResults.txt', 'a') as file:
                    file.write(s + '\n')
                t1 = time.time()
                model.seen += imgs.shape[0]

        if (epoch > 0) & (epoch % opt.checkpoint_interval == 0):
            torch.save(model.state_dict(), '%s/%s_epoch_%d_%g.pt' % (opt.checkpoint_dir, run_name, epoch, opt.img_size))

    # save final model
    dt = time.time() - t0
    print('Finished %g epochs in %.2fs (%.2fs/epoch, %.2fs/image)' % (epoch, dt, dt / (epoch + 1), dt / model.seen))
    s = '%s/%s_final_epoch_%d_%g.pt' % (opt.checkpoint_dir, run_name, epoch, opt.img_size)
    torch.save(model.state_dict(), s)

    opt.weights_path = s
    detect(opt)


if __name__ == '__main__':
    main(opt)
    torch.cuda.empty_cache()
