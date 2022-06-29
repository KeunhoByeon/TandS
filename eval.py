import argparse
import os
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings(action='ignore')


def blending(img, mask):
    img = img.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = img.astype(np.float32)

    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]

    img_b = np.where(mask == 1, img_b * 2 + 127, img_b)
    img_g = np.where(mask == 2, img_g * 2 + 127, img_g)

    output = np.zeros((img.shape[0], img.shape[1], 3))
    output[:, :, 0] = img_b
    output[:, :, 1] = img_g
    output[:, :, 2] = img_r
    output = output.clip(0, 255)

    return output.astype(np.uint8)


def load_data(data_dir):
    for path, dirs, files in os.walk(data_dir):
        for filename in files:
            if os.path.splitext(filename)[-1].lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            yield os.path.join(path, filename)


def eval(args):
    model = torch.load(args.resume, map_location='cpu')
    model.cpu()
    model.set_model_as_eval()

    with torch.no_grad():
        for i, file_path in tqdm(enumerate(load_data(args.data)), leave=False, desc='Eval'):
            input_np = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            input = input_np.copy().astype(np.float32) / 127. - 1.
            input = input[:, :, np.newaxis]
            input = input.transpose(2, 0, 1)
            input = torch.FloatTensor([input])

            if torch.cuda.is_available() and args.cuda:
                input = input.cuda()

            if args.use_model == 'A':
                output = model.Seg(model.G_E_A(input))
            elif args.use_model == 'B':
                output = model.Seg(model.G_E_B(input))
            else:
                raise AssertionError

            mask = output[0].detach().numpy().argmax(axis=1)[0]
            if np.max(mask) == 0:
                continue
            cv2.imwrite(os.path.join(args.save_dir, os.path.basename(file_path)), blending(input_np, mask))
            # cv2.imwrite(os.path.join(args.save_dir, os.path.basename(file_path)), mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--use_model', default='B')
    parser.add_argument('--data', default='~/data/bkkang/i2i/val/hrt2/')
    parser.add_argument('--resume', default='./results/20220621172027/models/80.pth', type=str, help='path to latest checkpoint')
    parser.add_argument('--save_dir', default='./results/temp_eval', type=str)
    parser.add_argument('--cuda', default=False)
    args = parser.parse_args()

    args.data = os.path.expanduser(args.data)
    args.save_dir = os.path.expanduser(args.save_dir)

    os.makedirs(args.save_dir, exist_ok=True)

    eval(args)
