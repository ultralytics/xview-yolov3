import argparse
import time
from sys import platform

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *

# from tqdm import tqdm

run_name = 'SGD'
parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=250, help='number of epochs')
parser.add_argument('-batch_size', type=int, default=3, help='size of each image batch')
parser.add_argument('-model_config_path', type=str, default='cfg/yolovx.cfg', help='path to model cfg file')
parser.add_argument('-weights_path', type=str, default='checkpoints/epoch21_sgd_608.pt', help='path to weights file')
parser.add_argument('-class_path', type=str, default='data/xview.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('-n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
parser.add_argument('-img_size', type=int, default=32 * 13, help='size of each image dimension')
parser.add_argument('-checkpoint_interval', type=int, default=50, help='interval between saving model weights')
parser.add_argument('-checkpoint_dir', type=str, default='checkpoints', help='directory for saving model checkpoints')
opt = parser.parse_args()
print(opt)


# @profile
def main(opt):
    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    torch.manual_seed(1)

    # Get data configuration
    if platform == 'darwin':  # macos
        train_path = '/Users/glennjocher/Downloads/DATA/xview/'
    else:
        train_path = '../'

    # Get hyper parameters
    hyperparams = parse_model_config(opt.model_config_path)[0]
    lr = float(hyperparams['learning_rate'])
    momentum = float(hyperparams['momentum'])
    decay = float(hyperparams['decay'])
    burn_in = int(hyperparams['burn_in'])

    # Initiate model
    model = Darknet(opt.model_config_path, opt.img_size, opt.batch_size).to(device).train()

    # Get dataloader
    dataloader = DataLoader(ListDataset_xview(train_path, opt.img_size),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # reload saved optimizer state
    resume_training = False
    if resume_training:
        model.load_state_dict(torch.load(opt.weights_path, map_location=device.type))
        # optimizer.load_state_dict(torch.load('optim.pth'))
        # optimizer.state = defaultdict(dict, optimizer.state)
    else:
        model.apply(weights_init_normal)  # initialize with random weights

    print('%10s' * 12 % ('Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'AP', 'mAP', 'time'))
    for epoch in range(opt.epochs):
        t0 = time.time()
        epochAP = 0

        for batch_i, (impath, imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            for j in range(4):
                loss = model(imgs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epochAP = (epochAP * batch_i + model.losses['AP']) / (batch_i + 1)
            s = ('%10s%10s' + '%10.3g' * 10) % (
                '%g/%g' % (epoch, opt.epochs - 1), '%g/%g' % (batch_i, len(dataloader) - 1), model.losses['x'],
                model.losses['y'], model.losses['w'], model.losses['h'], model.losses['conf'], model.losses['cls'],
                model.losses['loss'], model.losses['AP'], epochAP, time.time() - t0)
            print(s)
            with open('printedResults.txt', 'a') as file:
                file.write(s + '\n')
            t0 = time.time()

            model.seen += imgs.shape[0]

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), '%s/%s_epoch_%d_%g.pt' % (opt.checkpoint_dir, run_name, epoch, opt.img_size))

    # save final model
    torch.save(model.state_dict(), '%s/%s_epoch_%d_%g.pt' % (opt.checkpoint_dir, run_name, epoch, opt.img_size))


if __name__ == '__main__':
    main(opt)
    torch.cuda.empty_cache()
