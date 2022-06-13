import os

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.utils.data as data

from utils import resize_and_pad_image


class MRIDataset(data.Dataset):
    def __init__(self, data_dir, split, input_size=256):
        self.data_dir = data_dir
        self.split = split
        self.input_size = input_size

        self.seq = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.Multiply((0.8, 1.2))),
            iaa.Sometimes(0.2, iaa.LogContrast((0.8, 1.2))),
        ], random_order=True)

        cet1_dir = os.path.join(self.data_dir, self.split, 'cet1')
        hrt2_dir = os.path.join(self.data_dir, self.split, 'hrt2')
        cet1_mask_dir = os.path.join(self.data_dir, self.split, 'mask')

        self.samples_cet1 = []
        for path, dirs, files in os.walk(cet1_dir):
            for filename in files:
                if os.path.splitext(filename)[-1].lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                img_path = os.path.join(path, filename)
                mask_path = img_path.replace(cet1_dir, cet1_mask_dir).replace('_ceT1_', '_Label_')
                mask_path = mask_path if os.path.isfile(mask_path) else None
                self.samples_cet1.append((img_path, mask_path))

        self.samples_hrt2 = []
        for path, dirs, files in os.walk(hrt2_dir):
            for filename in files:
                if os.path.splitext(filename)[-1].lower() not in ('.png', '.jpg', '.jpeg'):
                    continue
                img_path = os.path.join(path, filename)
                self.samples_hrt2.append((img_path, None))

        self.cet1_len = len(self.samples_cet1)
        self.hrt2_len = len(self.samples_hrt2)

        print('Loaded {} data (ceT1: {} hrT2: {})'.format(self.split, self.cet1_len, self.hrt2_len))

    def __getitem__(self, index):
        cet1_index = index % self.cet1_len if index >= self.cet1_len else index
        cet1 = cv2.imread(self.samples_cet1[cet1_index][0], cv2.IMREAD_GRAYSCALE)
        cet1 = resize_and_pad_image(cet1, input_size=(self.input_size, self.input_size))

        hrt2_index = index % self.hrt2_len if index >= self.hrt2_len else index
        hrt2 = cv2.imread(self.samples_hrt2[hrt2_index][0], cv2.IMREAD_GRAYSCALE)
        hrt2 = resize_and_pad_image(hrt2, input_size=(self.input_size, self.input_size))

        if self.samples_cet1[cet1_index][1] is not None:
            mask = cv2.imread(self.samples_cet1[cet1_index][1], cv2.IMREAD_GRAYSCALE)
            mask = resize_and_pad_image(mask, input_size=(self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros(cet1.shape[:2])

        # Data Augmentation
        if self.split == 'train':
            cet1 = self.seq(image=cet1)
            hrt2 = self.seq(image=hrt2)

        cet1 = cet1.astype(np.float32) / 127. - 1.
        cet1 = cet1[:, :, np.newaxis]
        cet1 = cet1.transpose(2, 0, 1)
        cet1 = torch.FloatTensor(cet1)

        hrt2 = hrt2.astype(np.float32) / 127. - 1.
        hrt2 = hrt2[:, :, np.newaxis]
        hrt2 = hrt2.transpose(2, 0, 1)
        hrt2 = torch.FloatTensor(hrt2)

        mask = torch.LongTensor(mask)

        return cet1, hrt2, mask

    def __len__(self):
        return max(len(self.samples_cet1), len(self.samples_hrt2))
