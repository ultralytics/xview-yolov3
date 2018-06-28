import argparse
import time
from sys import platform

# from detect import detect
from models import *
from utils.datasets import *
from utils.utils import *

# batch_size 8: 32*17 = 544
# batch_size 4: 32*25 = 800 (1.47 vs 544) or 32*23 = 736
# batch_size 2: 32*35 = 1120 (1.40 vs 800, 2.06 cumulative)
# batch_size 1: 32*49 = 1568 (1.40 vs 1120, 2.88 cumulative)


parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=9999, help='number of epochs')
parser.add_argument('-image_folder', type=str, default='data/train_images8', help='path to images')
parser.add_argument('-output_folder', type=str, default='data/xview_predictions', help='path to outputs')
parser.add_argument('-batch_size', type=int, default=8, help='size of each image batch')
parser.add_argument('-config_path', type=str, default='cfg/yolovx_30_pixelAnchors.cfg', help='cfg file path')
parser.add_argument('-weights_path', type=str, default='checkpoints/june22_e400_608.pt', help='weights')
parser.add_argument('-class_path', type=str, default='data/xview.names', help='path to class label file')
parser.add_argument('-conf_thres', type=float, default=0.99, help='object confidence threshold')
parser.add_argument('-nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('-n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('-img_size', type=int, default=32 * 19, help='size of each image dimension')
parser.add_argument('-checkpoint_interval', type=int, default=20, help='interval between saving model weights')
parser.add_argument('-checkpoint_dir', type=str, default='checkpoints', help='directory for saving model checkpoints')
parser.add_argument('-plot_flag', type=bool, default=True, help='plots predicted images if True')
opt = parser.parse_args()
print(opt)


# @profile
def main(opt):
    os.makedirs('checkpoints', exist_ok=True)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    # Get data configuration
    if platform == 'darwin':  # macos
        run_name = 'june27_crop16_4mini_noemptymini_'
        train_path = '/Users/glennjocher/Downloads/DATA/xview/'
    else:
        torch.backends.cudnn.benchmark = True
        run_name = 'june29_pixelAnchors_'
        train_path = '../'

    # Initiate model
    model = Darknet(opt.config_path, opt.img_size).to(device).train()

    # Get dataloader
    dataloader = ListDataset_xview_crop(train_path, batch_size=opt.batch_size, img_size=opt.img_size)

    # optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.98, weight_decay=0.0005, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    # reload saved optimizer state
    resume_training = True
    if (platform != 'darwin') and resume_training:
        model.load_state_dict(torch.load('../june27_crop16_2minibatch__best_608.pt', map_location=device.type))
    # optimizer.load_state_dict(torch.load('optim.pth'))
    # optimizer.state = defaultdict(dict, optimizer.state)
    # else:
    # model.apply(weights_init_normal)  # initialize with random weights
    # torch.save(model.state_dict(), 'weights/init.pt')

    # modelinfo(model)
    t0 = time.time()
    t1 = time.time()
    best_loss = float('inf')
    print('%10s' * 16 % (
        'Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'precision', 'recall', 'nGT', 'TP', 'FP', 'FN',
        'time'))
    for epoch in range(opt.epochs):
        rloss = defaultdict(float)  # running loss
        ui = -1
        for i, (imgs, targets) in enumerate(dataloader):

            n = 4  # number of pictures at a time
            for j in range(int(len(imgs) / n)):
                targets_j = targets[j * n:j * n + n]
                nGT = sum([len(x) for x in targets_j])
                if nGT == 0:
                    continue

                loss = model(imgs[j * n:j * n + n].to(device), targets_j, requestPrecision=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (model.losses['nGT'] > 0) | (i == 0):
                    ui += 1
                    for key, val in model.losses.items():
                        rloss[key] = (rloss[key] * ui + val) / (ui + 1)

                s = ('%10s%10s' + '%10.3g' * 14) % (
                    '%g/%g' % (epoch, opt.epochs - 1), '%g/%g' % (i, len(dataloader) - 1), rloss['x'],
                    rloss['y'], rloss['w'], rloss['h'], rloss['conf'], rloss['cls'],
                    rloss['loss'], rloss['precision'], rloss['recall'], model.losses['nGT'], model.losses['TP'],
                    model.losses['FP'], model.losses['FN'],
                    time.time() - t1)
                t1 = time.time()
                print(s)
                model.seen += imgs.shape[0]

            # if i == 5:
            #    return

        with open('printedResults.txt', 'a') as file:
            file.write(s + '\n')

        if (epoch > opt.checkpoint_interval) & (rloss['loss'] < best_loss):
            torch.save(model.state_dict(), '%s/%s_best_%g.pt' % (opt.checkpoint_dir, run_name, opt.img_size))
            best_loss = rloss['loss']

    # save final model
    dt = time.time() - t0
    print('Finished %g epochs in %.2fs (%.2fs/epoch, %.2fs/image)' % (epoch, dt, dt / (epoch + 1), dt / model.seen))
    s = '%s/%s_final_e%d_%g.pt' % (opt.checkpoint_dir, run_name, epoch, opt.img_size)
    torch.save(model.state_dict(), s)

    opt.weights_path = s
    # detect(opt)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(opt)
    torch.cuda.empty_cache()
