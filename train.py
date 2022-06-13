import argparse
import os
import random
import time
import warnings

import torch
from tqdm import tqdm

from dataloader import MRIDataset
from logger import Logger
from model import CycleSegGan
from utils import debug_result

warnings.filterwarnings(action='ignore')

loss_names = ["G", "G_A", "G_B", "seg", "D_A", "D_B", "G_A_idt", "G_A_rec", "G_A_gan", "G_B_idt", "G_B_rec", "G_B_gan"]


def val(args, epoch, model, val_loader, logger=None):
    model.set_model_as_eval()

    total_losses = {loss_name: [0, 0] for loss_name in loss_names}

    start_time = time.time()
    with torch.no_grad():
        for i, (cet1s, hrt2s, masks) in tqdm(enumerate(val_loader), leave=False, desc='Validation {}'.format(epoch), total=len(val_loader)):
            if torch.cuda.is_available():
                cet1s = cet1s.cuda()
                hrt2s = hrt2s.cuda()
                masks = masks.cuda()

            model.set_data(A=cet1s, B=hrt2s, mask_A=masks, mask_B=None)
            model.run_val()
            loss_list = model.get_current_loss()

            for loss_name, temp_l in zip(loss_names, loss_list):
                total_losses[loss_name][0] += temp_l.data.item()
                total_losses[loss_name][1] += 1

            if args.debug:
                debug_result(os.path.join(args.result, 'debug', str(epoch)), cet1s, hrt2s, masks, model.B_fake_A, model.A_fake_B,
                             model.A_seg_ori[0] if isinstance(model.A_seg_ori, tuple) else model.A_seg_ori, model.A_seg_rec[0] if isinstance(model.A_seg_rec, tuple) else model.A_seg_rec, tag=str(i))

    if logger is not None:
        logger('*Validation', components_data=total_losses, time=time.time() - start_time)

    return total_losses


def train(args, epoch, model, train_loader, logger=None):
    model.set_model_as_train()

    temp_losses = {loss_name: [0, 0] for loss_name in loss_names}
    total_losses = {loss_name: [0, 0] for loss_name in loss_names}

    num_progress = 0
    next_print = args.print_freq

    start_time = time.time()
    for i, (cet1s, hrt2s, masks) in enumerate(train_loader):
        if torch.cuda.is_available():
            cet1s = cet1s.cuda()
            hrt2s = hrt2s.cuda()
            masks = masks.cuda()

        model.set_data(A=cet1s, B=hrt2s, mask_A=masks, mask_B=None)
        model.run_train()
        loss_list = model.get_current_loss()

        for loss_name, temp_l in zip(loss_names, loss_list):
            total_losses[loss_name][0] += temp_l.data.item()
            total_losses[loss_name][1] += 1
            if logger is not None:
                temp_losses[loss_name][0] += temp_l.data.item()
                temp_losses[loss_name][1] += 1

        num_progress += len(cet1s)
        if num_progress >= next_print:
            if logger is not None:
                logger(epoch=epoch, batch=num_progress, components_data=temp_losses, time=time.time() - start_time)
                temp_losses = {loss_name: [0, 0] for loss_name in loss_names}
            start_time = time.time()
            next_print += args.print_freq

    logger(epoch=epoch, components_data=total_losses)

    return total_losses


def run(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Model
    model = CycleSegGan(args)

    if args.resume is not None:
        # TODO: Resume function should be made
        # This is just a temp resume func
        model = torch.load(args.resume)

    if torch.cuda.is_available():
        model = model.cuda()

    # Dataset
    train_dataset = MRIDataset(args.data, 'train', input_size=args.input_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    val_dataset = MRIDataset(args.data, 'val', input_size=args.input_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size * 2, num_workers=args.workers, pin_memory=True, shuffle=False)

    # Logger
    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(train_loader.dataset), float_round=5)
    logger.set_sort(loss_names)
    logger(str(args))

    # Model save dir
    save_dir = os.path.join(args.result, 'models')
    os.makedirs(save_dir, exist_ok=True)

    if args.preval:
        val(args, 'Pre-Val', model, val_loader, logger=logger)

    print('Training...')
    for epoch in range(args.start_epoch, args.epochs):
        train_total_loss = train(args, epoch, model, train_loader, logger=logger)
        if epoch % args.val_freq == 0:
            val_total_loss = val(args, epoch, model, val_loader, logger=logger)

        save_filename = '{0}.pth'.format(epoch)
        torch.save(model, os.path.join(save_dir, save_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--input_size', default=256, type=int, help='image input size')
    parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--batch_size', default=6, type=int, help='mini-batch size')
    parser.add_argument('--lr_D', default=0.0001, type=float, help='initial learning rate of D')
    parser.add_argument('--lr_G', default=0.0001, type=float, help='initial learning rate of G')
    parser.add_argument('--loss_rate_A', default=1.0, type=float)
    parser.add_argument('--loss_rate_B', default=1.0, type=float)
    parser.add_argument('--loss_rate_idt', default=0.5, type=float)
    parser.add_argument('--loss_rate_rec', default=1.0, type=float)
    parser.add_argument('--loss_rate_gan', default=1.0, type=float)
    parser.add_argument('--loss_rate_seg', default=1.0, type=float)
    parser.add_argument('--print_freq', default=500, type=int, help='print and save frequency (default: 100)')
    parser.add_argument('--val_freq', default=1, type=int, help='validation frequency (default: 5)')
    parser.add_argument('--seed', default=103, type=int, help='seed for initializing training.')
    parser.add_argument('--data', default='~/data/bkkang/i2i', help='path to dataset')
    parser.add_argument('--result', default='results', help='path to results')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
    parser.add_argument('--preval', default=False, action='store_true', help='pre validation before training')
    parser.add_argument('--debug', default=True, action='store_true', help='debug validation')
    args = parser.parse_args()

    args.data = os.path.expanduser(args.data)
    args.result = os.path.expanduser(args.result)
    args.result = os.path.join(args.result, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    os.makedirs(args.result, exist_ok=True)

    run(args)
