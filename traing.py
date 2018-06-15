import argparse
import time
from sys import platform

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=36, help='number of epochs')
parser.add_argument('-image_folder', type=str, default='train_images', help='path to dataset')
parser.add_argument('-batch_size', type=int, default=4, help='size of each image batch')
parser.add_argument('-model_config_path', type=str, default='config/yolovx.cfg', help='path to model config file')
parser.add_argument('-data_config_path', type=str, default='config/xview.data', help='path to data config file')
parser.add_argument('-weights_path', type=str, default='checkpoints/epoch270.pt', help='path to weights file')
parser.add_argument('-class_path', type=str, default='data/xview.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('-n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('-img_size', type=int, default=32 * 27, help='size of each image dimension')
parser.add_argument('-checkpoint_interval', type=int, default=4, help='interval between saving model weights')
parser.add_argument('-checkpoint_dir', type=str, default='checkpoints', help='directory for saving model checkpoints')
opt = parser.parse_args()
print(opt)


# @profile
def main(opt):
    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    torch.manual_seed(1)
    # classes = load_classes(opt.class_path)

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
    model = Darknet(opt.model_config_path)
    # model.load_state_dict(torch.load(opt.weights_path, map_location=device.type))
    model.apply(weights_init_normal)  # random weights
    model.to(device).train()

    # Get dataloader
    dataloader = DataLoader(ListDataset_xview(train_path, opt.img_size),
                            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=0, weight_decay=decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, amsgrad=True)

    for epoch in range(opt.epochs):
        print('%10s' * 12 % ('Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'AP', 'mAP', 'time'))
        t0 = time.time()
        epochAP, nGT = 0, 0
        for batch_i, (impath, imgs, targets, nT) in enumerate(dataloader):
            imgs = imgs.type(Tensor)
            targets = targets.type(Tensor)

            # model.current_img_path = imgs
            for repeat_batch_i in range(8):
                loss = model(imgs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epochAP = (epochAP * batch_i + model.losses['AP']) / (batch_i + 1)
            nGT += model.losses['nGT']
            s = ('%10s%10s' + '%10.3g' * 10) % (
                '%g/%g' % (epoch, opt.epochs - 1), '%g/%g' % (batch_i, len(dataloader) - 1),
                model.losses['x'], model.losses['y'], model.losses['w'],
                model.losses['h'], model.losses['conf'], model.losses['cls'],
                loss.item(), model.losses['AP'], epochAP, time.time() - t0)
            print(s)

            with open('printedResults.txt', 'a') as file:
               file.write(s + '\n')
            model.seen += imgs.shape[0]
            
            t0 = time.time()

        if cuda and (epoch % opt.checkpoint_interval == 0):
            torch.save(model.state_dict(), '%s/epoch%d_adam_864.pt' % (opt.checkpoint_dir, epoch))
    torch.save(model.state_dict(), '%s/epoch%d_adam_864%d.pt' % (opt.checkpoint_dir, epoch))


if __name__ == '__main__':
    main(opt)
