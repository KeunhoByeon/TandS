import os
import random
from math import pi

import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
from torch import nn


# Model Utils-------------------------------------------------------------------------------------------------------------------------------------------------------

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def to_binary(tensor, threshold=0.5):
    s = torch.sign(tensor - threshold)
    b = torch.relu(s)
    return b


def muti_bce_loss_fusion(bce_loss, preds, targets):
    d0, d1, d2, d3, d4, d5, d6 = preds
    loss0 = bce_loss(d0, targets)
    loss1 = bce_loss(d1, targets)
    loss2 = bce_loss(d2, targets)
    loss3 = bce_loss(d3, targets)
    loss4 = bce_loss(d4, targets)
    loss5 = bce_loss(d5, targets)
    loss6 = bce_loss(d6, targets)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss


def muti_loss_fusion(loss_func, preds, targets):
    d0, d1, d2, d3, d4, d5, d6 = preds
    loss0 = loss_func(d0, targets)
    loss1 = loss_func(d1, targets)
    loss2 = loss_func(d2, targets)
    loss3 = loss_func(d3, targets)
    loss4 = loss_func(d4, targets)
    loss5 = loss_func(d5, targets)
    loss6 = loss_func(d6, targets)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss


def blending(img, mask):
    img = img.copy().astype(np.float32)

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


def debug_result(save_dir, cet1s, hrt2s, masks, fakes_1, fakes_2, seg_preds, seg_idt_preds, tag=None):
    os.makedirs(save_dir, exist_ok=True)

    cet1s = ((torch.squeeze(cet1s.cpu().detach(), 1).numpy() + 1.) / 2 * 255.).astype(np.uint8)
    hrt2s = ((torch.squeeze(hrt2s.cpu().detach(), 1).numpy() + 1.) / 2 * 255.).astype(np.uint8)
    fakes_1 = ((torch.squeeze(fakes_1.cpu().detach(), 1).numpy() + 1.) / 2 * 255.).astype(np.uint8)
    fakes_2 = ((torch.squeeze(fakes_2.cpu().detach(), 1).numpy() + 1.) / 2 * 255.).astype(np.uint8)

    masks = (torch.squeeze(masks.cpu().detach(), 1).numpy() * 127.).astype(np.uint8)
    seg_preds = (torch.squeeze(seg_preds.cpu().detach(), 1).numpy().argmax(axis=1) * 127.).astype(np.uint8)
    seg_idt_preds = (torch.squeeze(seg_idt_preds.cpu().detach(), 1).numpy().argmax(axis=1) * 127.).astype(np.uint8)

    for i, (cet1, hrt2, mask, fake_1, fake_2, seg_pred, seg_idt_pred) in enumerate(zip(cet1s, hrt2s, masks, fakes_1, fakes_2, seg_preds, seg_idt_preds)):
        debug_image = np.hstack((cet1, hrt2, mask, fake_1, fake_2, seg_pred, seg_idt_pred))
        save_filename = '' if tag is None else '{}_'.format(tag) + '{}.png'.format(i)
        cv2.imwrite(os.path.join(save_dir, save_filename), debug_image)


# Image Utils-------------------------------------------------------------------------------------------------------------------------------------------------------

def get_M(h, w, f, theta, phi, gamma, dx, dy, dz):
    # Projection 2D -> 3D matrix
    A1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1], [0, 0, 1]])
    # Projection 3D -> 2D matrix
    A2 = np.array([[f, 0, w / 2, 0], [0, f, h / 2, 0], [0, 0, 1, 0]])
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0], [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]])
    RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0], [0, 1, 0, 0], [np.sin(phi), 0, np.cos(phi), 0], [0, 0, 0, 1]])
    RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0], [np.sin(gamma), np.cos(gamma), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    R = np.dot(np.dot(RX, RY), RZ)  # Composed rotation matrix with (RX, RY, RZ)
    T = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])  # Translation matrix

    # Final transformation matrix
    return np.dot(A2, np.dot(T, np.dot(R, A1)))


def rotate_along_axis(image, theta=0, phi=0, gamma=0, dy=0, dx=0, dz=0):
    (height, width) = image.shape[:2]

    # Get radius of rotation along 3 axes
    rtheta = theta * pi / 180.0
    rphi = phi * pi / 180.0
    rgamma = gamma * pi / 180.0

    # Get ideal focal length on z axis
    # NOTE: Change this section to other axis if needed
    d = np.sqrt(height ** 2 + width ** 2)
    focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
    dz = focal

    # Get projection matrix
    mat = get_M(height, width, focal, rtheta, rphi, rgamma, dx, dy, dz)

    return cv2.warpPerspective(image.copy(), mat, (width, height))


def resize(img, input_size=(512, 512)):
    # 1) get ratio
    old_size = img.shape[:2]
    ratio = min(float(input_size[0]) / old_size[0], float(input_size[1]) / old_size[1])

    # 2) resize image
    new_size = tuple([int(x * ratio) for x in old_size])
    interpolation = cv2.INTER_AREA if new_size[0] < old_size[0] else cv2.INTER_CUBIC
    img = cv2.resize(img.copy(), (new_size[1], new_size[0]), interpolation=interpolation)

    return img


def resize_and_pad_image(img, input_size=(512, 512), interpolation=None):
    # 1) get ratio
    old_size = img.shape[:2]
    ratio = min(float(input_size[0]) / old_size[0], float(input_size[1]) / old_size[1])

    # 2) resize image
    new_size = tuple([int(x * ratio) for x in old_size])
    if interpolation is None:
        interpolation = cv2.INTER_AREA if new_size[0] < old_size[0] else cv2.INTER_CUBIC
    img = cv2.resize(img.copy(), (new_size[1], new_size[0]), interpolation=interpolation)

    # 3) pad image
    delta_w = input_size[1] - new_size[1]
    delta_h = input_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img


def recover_original_size(img, original_size):
    now_size = img.shape[:2]

    # 1) get ratio
    old_size = original_size
    ratio = min(float(now_size[0]) / original_size[0], float(now_size[1]) / original_size[1])

    # 2) remove padding
    new_size = tuple([int(x * ratio) for x in old_size])
    delta_w = now_size[1] - new_size[1]
    delta_h = now_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    img = img[top:now_size[0] - bottom, left:now_size[1] - right]

    # 3) resize image
    interpolation = cv2.INTER_AREA if new_size[0] < old_size[0] else cv2.INTER_CUBIC
    img = cv2.resize(img, (original_size[1], original_size[0]), interpolation=interpolation)

    return img


# etc. -------------------------------------------------------------------------------------------------------------------------------------------------------------

def gauss_square(v):
    return random.gauss(0, 0.5) * random.gauss(0, 0.5) * v
