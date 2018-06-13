import argparse
import time
import os

import torch.optim as optim
from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.parse_config import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=30, help='number of epochs')
parser.add_argument('-image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('-batch_size', type=int, default=4, help='size of each image batch')
parser.add_argument('-model_config_path', type=str, default='config/yolovx.cfg', help='path to model config file')
parser.add_argument('-data_config_path', type=str, default='config/xview.data', help='path to data config file')
parser.add_argument('-weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('-class_path', type=str, default='data/xview.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('-n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
parser.add_argument('-img_size', type=int, default=32 * 13, help='size of each image dimension')
parser.add_argument('-checkpoint_interval', type=int, default=1, help='interval between saving model weights')
parser.add_argument('-checkpoint_dir', type=str, default='checkpoints', help='directory for saving model checkpoints')
opt = parser.parse_args()
print(opt)


#@profile
def main(opt):
    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    torch.manual_seed(1)
    classes = load_classes(opt.class_path)

    # Get data configuration
    # data_config = parse_data_config(opt.data_config_path)
    train_path = '/Users/glennjocher/Downloads/DATA/xview/'  # data_config['train']
    if cuda:
        train_path = ''

    # Get hyper parameters
    hyperparams = parse_model_config(opt.model_config_path)[0]
    lr = float(hyperparams['learning_rate'])
    momentum = float(hyperparams['momentum'])
    decay = float(hyperparams['decay'])
    burn_in = int(hyperparams['burn_in'])

    # Initiate model
    model = Darknet(opt.model_config_path)
    # model.load_weights(opt.weights_path)
    model.apply(weights_init_normal)
    model.to(device)
    model.train()

    # Get dataloader
    dataloader = DataLoader(ListDataset_xview(train_path, opt.img_size),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=0, weight_decay=decay)

    for epoch in range(opt.epochs):
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            t0 = time.time()
            imgs = imgs.type(Tensor)
            targets = targets.type(Tensor)

            # import matplotlib.pyplot as plt
            # plt.imshow(imgs[0,0])

            loss = model(imgs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            s = '[Epoch %d/%d, Batch %d/%d] [x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, AP: %.5f] %.3fs' % (
                epoch, opt.epochs, batch_i, len(dataloader),
                model.losses['x'], model.losses['y'], model.losses['w'],
                model.losses['h'], model.losses['conf'], model.losses['cls'],
                loss.item(), model.losses['AP'], time.time() - t0)
            print(s)
            with open('printedResults.txt', 'a') as file:
                file.write(s + '\n')

            model.seen += imgs.shape[0]

            #if batch_i == 0:
            #   return

        if epoch % opt.checkpoint_interval == 0:
            model.save_weights('%s/%d.weights' % (opt.checkpoint_dir, epoch))

if __name__ == '__main__':
    main(opt)
    #os.system('sudo shutdown')