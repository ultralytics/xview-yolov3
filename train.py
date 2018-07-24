import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *

# batch_size 8: 32*17 = 544
# batch_size 4: 32*25 = 800 (1.47 vs 544) or 32*23 = 736
# batch_size 2: 32*35 = 1120 (1.40 vs 800, 2.06 cumulative)
# batch_size 1: 32*49 = 1568 (1.40 vs 1120, 2.88 cumulative)

targets_path = 'utils/targets_c60.mat'

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=2, help='number of epochs')
parser.add_argument('-batch_size', type=int, default=8, help='size of each image batch')
parser.add_argument('-config_path', type=str, default='cfg/c60.cfg', help='cfg file path')
parser.add_argument('-img_size', type=int, default=32 * 19, help='size of each image dimension')
parser.add_argument('-checkpoint_interval', type=int, default=0, help='interval between saving model weights')
parser.add_argument('-checkpoint_dir', type=str, default='checkpoints', help='directory for saving model checkpoints')
opt = parser.parse_args()
print(opt)


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

    # Configure run
    if platform == 'darwin':  # macos
        # torch.backends.cudnn.benchmark = True
        run_name = 'test2'
        train_path = '/Users/glennjocher/Downloads/DATA/xview/train_images_reduced'
    else:
        torch.backends.cudnn.benchmark = True
        run_name = 'c60'
        train_path = '../train_images'

    # Initiate model
    model = Darknet(opt.config_path, opt.img_size, targets=targets_path).to(device).train()

    # Get dataloader
    dataloader = ListDataset(train_path, batch_size=opt.batch_size, img_size=opt.img_size, targets_path=targets_path)

    # Set optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.98, weight_decay=0.0005, nesterov=True)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # reload saved optimizer state
    resume_training = True
    if resume_training:
        checkpoint = torch.load('checkpoints/test2.pt', map_location='cuda:0' if cuda else 'cpu')

        current = model.state_dict()
        # saved = torch.load('checkpoints/fresh9_5_e201.pt', map_location='cuda:0' if cuda else 'cpu')
        saved = checkpoint['model']
        # 1. filter out unnecessary keys
        saved = {k: v for k, v in saved.items() if ((k in current) and (current[k].shape == v.shape))}
        # 2. overwrite entries in the existing state dict
        current.update(saved)
        # 3. load the new state dict
        model.load_state_dict(current)

        # # Transfer learning
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     #name = name.replace('module_list.', '')
        #     #print('%4g %70s %9s %12g %20s %12g %12g' % (
        #     #    i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
        #     if p.shape[0] != 650:  # not YOLO layer
        #         p.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

        del current, saved, checkpoint
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
        start_epoch = 0

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 24, eta_min=0.00001, last_epoch=-1)
    # y = 0.001 * exp(-0.00921 * x)  # 1e-4 @ 250, 1e-5 @ 500
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99082, last_epoch=start_epoch - 1)

    modelinfo(model)
    t0, t1 = time.time(), time.time()
    best_loss = float('inf')
    print('%10s' * 16 % (
        'Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'P', 'R', 'nGT', 'TP', 'FP', 'FN', 'time'))
    for epoch in range(opt.epochs):
        epoch += start_epoch

        # Update scheduler
        # if epoch % 25 == 0:
        #     scheduler.last_epoch = -1  # for cosine annealing, restart every 25 epochs
        scheduler.step()

        ui = -1
        rloss = defaultdict(float)  # running loss
        metrics = torch.zeros((3, 60))
        for i, (imgs, targets) in enumerate(dataloader):

            n = 4  # number of pictures at a time
            for j in range(int(len(imgs) / n)):
                targets_j = targets[j * n:j * n + n]
                nGT = sum([len(x) for x in targets_j])
                if nGT < 2:
                    continue

                loss = model(imgs[j * n:j * n + n].to(device), targets_j, requestPrecision=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ui += 1
                metrics += model.losses['metrics']
                for key, val in model.losses.items():
                    rloss[key] = (rloss[key] * ui + val) / (ui + 1)

                # Precision
                precision = metrics[0] / (metrics[0] + metrics[1] + 1e-16)
                k = (metrics[0] + metrics[1]) > 0
                if k.sum() > 0:
                    mean_precision = precision[k].mean()
                else:
                    mean_precision = 0

                # Recall
                recall = metrics[0] / (metrics[0] + metrics[2] + 1e-16)
                k = (metrics[0] + metrics[2]) > 0
                if k.sum() > 0:
                    mean_recall = recall[k].mean()
                else:
                    mean_recall = 0

                s = ('%10s%10s' + '%10.3g' * 14) % (
                    '%g/%g' % (epoch, opt.epochs - 1), '%g/%g' % (i, len(dataloader) - 1), rloss['x'],
                    rloss['y'], rloss['w'], rloss['h'], rloss['conf'], rloss['cls'],
                    rloss['loss'], mean_precision, mean_recall, model.losses['nGT'], model.losses['TP'],
                    model.losses['FP'], model.losses['FN'], time.time() - t1)
                t1 = time.time()
                print(s)

            # if i == 3:
            #     return

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '\n')

        # Save if best epoch
        if (epoch >= opt.checkpoint_interval) & (rloss['loss'] < best_loss):
            best_loss = rloss['loss'] / rloss['nGT']
            opt.weights_path = '%s/%s.pt' % (opt.checkpoint_dir, run_name)  # best weight path
            torch.save({'epoch': epoch, 'opt': opt, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       opt.weights_path)

    # Save final model
    dt = time.time() - t0
    print('Finished %g epochs in %.2fs (%.2fs/epoch)' % (epoch, dt, dt / (epoch + 1)))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(opt)
    torch.cuda.empty_cache()
