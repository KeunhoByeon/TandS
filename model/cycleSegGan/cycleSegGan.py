from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss

from utils.image_pool import ImagePool
from .discriminator_patchGAN import NLayerDiscriminator
from .generator_resnet import ResnetEncoder, ResnetDecoder
from .segmentStacked import SegmentStacked


class LossFunction(_WeightedLoss):
    def __init__(self, loss_func, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(LossFunction, self).__init__(weight, size_average, reduce, reduction)
        if weight is None:
            self.loss_func = loss_func(size_average=size_average)
        else:
            self.loss_func = loss_func(size_average=size_average, weight=weight)

    def _tuple_input_target_forward(self, input, target):
        d0, d1, d2, d3, d4, d5, d6 = input
        t0, t1, t2, t3, t4, t5, t6 = target
        loss0 = self.loss_func(d0, t0)  # / (2 ** 1)
        loss1 = self.loss_func(d1, t1)  # / (2 ** 2)
        loss2 = self.loss_func(d2, t2)  # / (2 ** 3)
        loss3 = self.loss_func(d3, t3)  # / (2 ** 4)
        loss4 = self.loss_func(d4, t4)  # / (2 ** 5)
        loss5 = self.loss_func(d5, t5)  # / (2 ** 6)
        loss6 = self.loss_func(d6, t6)  # / (2 ** 7)
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        return loss / 7.

    def _tuple_input_forward(self, input, target):
        if isinstance(target, tuple) or isinstance(target, list):
            return self._tuple_input_target_forward(input, target)

        d0, d1, d2, d3, d4, d5, d6 = input
        loss0 = self.loss_func(d0, target)  # / (2 ** 1)
        loss1 = self.loss_func(d1, target)  # / (2 ** 2)
        loss2 = self.loss_func(d2, target)  # / (2 ** 3)
        loss3 = self.loss_func(d3, target)  # / (2 ** 4)
        loss4 = self.loss_func(d4, target)  # / (2 ** 5)
        loss5 = self.loss_func(d5, target)  # / (2 ** 6)
        loss6 = self.loss_func(d6, target)  # / (2 ** 7)
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        return loss / 7.

    def forward(self, input, target):
        if isinstance(input, tuple) or isinstance(input, list):
            return self._tuple_input_forward(input, target)

        return self.loss_func(input, target)


class CycleSegGan:
    def __init__(self, args, input_nc=1, output_nc=1, num_classes=3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # A Generator model
        self.G_E_A = ResnetEncoder(input_nc=input_nc)
        self.G_D_A = ResnetDecoder(output_nc=output_nc)

        # B Generator model
        self.G_E_B = ResnetEncoder(input_nc=input_nc)
        self.G_D_B = ResnetDecoder(output_nc=output_nc)

        # A Discriminator model
        self.D_A = NLayerDiscriminator(input_nc=output_nc)

        # B Discriminator model
        self.D_B = NLayerDiscriminator(input_nc=output_nc)

        # Segmentation model
        if args.seg_model == 'stacked':
            self.Seg = SegmentStacked(num_classes=num_classes)
        elif args.seg_model == 'resnet':
            self.Seg = ResnetDecoder(output_nc=num_classes, activation='softmax')
        else:
            raise AssertionError

        self.do_seg = False

        self.seg_weights = torch.FloatTensor([args.weight0, args.weight1, args.weight2]).to(self.device)

        # Loss funcs
        self.loss_funcs = {
            'idt': LossFunction(torch.nn.L1Loss),  # identity
            'rec': LossFunction(torch.nn.L1Loss),  # recon, cycle loss (Idt)
            'gan': LossFunction(torch.nn.MSELoss),  # from D loss
            'seg': LossFunction(torch.nn.CrossEntropyLoss, size_average=True, weight=self.seg_weights),  # seg, without ignore_index=0
            'scp': LossFunction(torch.nn.L1Loss),  # seg compare
        }

        self.loss_rate = {
            'A': args.loss_rate_A,
            'B': args.loss_rate_B,
            'idt': args.loss_rate_idt,
            'rec': args.loss_rate_rec,
            'gan': args.loss_rate_gan,
            'seg': args.loss_rate_seg,
        }

        # Optimizers
        self.optimizer_G = torch.optim.Adam(list(self.G_E_A.parameters()) + list(self.G_D_A.parameters()) + list(self.G_E_B.parameters()) + list(self.G_D_B.parameters()) + list(self.Seg.parameters()), lr=args.lr_G, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(list(self.D_A.parameters()) + list(self.D_B.parameters()), lr=args.lr_D, betas=(0.5, 0.999))

        # create image buffer to store previously generated images
        self.fake_A_pool = ImagePool()
        self.fake_B_pool = ImagePool()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def cpu(self):
        self.G_E_A = self.G_E_A.cpu()
        self.G_D_A = self.G_D_A.cpu()
        self.G_E_B = self.G_E_B.cpu()
        self.G_D_B = self.G_D_B.cpu()
        self.D_A = self.D_A.cpu()
        self.D_B = self.D_B.cpu()
        self.Seg = self.Seg.cpu()
        self.loss_funcs['seg'] = LossFunction(torch.nn.CrossEntropyLoss, size_average=True, weight=self.seg_weights.cpu())

        return self

    def cuda(self):
        self.G_E_A = self.G_E_A.cuda()
        self.G_D_A = self.G_D_A.cuda()
        self.G_E_B = self.G_E_B.cuda()
        self.G_D_B = self.G_D_B.cuda()
        self.D_A = self.D_A.cuda()
        self.D_B = self.D_B.cuda()
        self.Seg = self.Seg.cuda()
        self.loss_funcs['seg'] = LossFunction(torch.nn.CrossEntropyLoss, size_average=True, weight=self.seg_weights.cuda())

        return self

    def set_model_as_train(self):
        self.G_E_A = self.G_E_A.train()
        self.G_D_A = self.G_D_A.train()
        self.G_E_B = self.G_E_B.train()
        self.G_D_B = self.G_D_B.train()
        self.D_A = self.D_A.train()
        self.D_B = self.D_B.train()
        self.Seg = self.Seg.train()

    def set_model_as_eval(self):
        self.G_E_A = self.G_E_A.eval()
        self.G_D_A = self.G_D_A.eval()
        self.G_E_B = self.G_E_B.eval()
        self.G_D_B = self.G_D_B.eval()
        self.D_A = self.D_A.eval()
        self.D_B = self.D_B.eval()
        self.Seg = self.Seg.eval()

    def forward_A(self):  # A to B
        A_features_A = self.G_E_A(self.A)
        A_features_B = self.G_E_B(self.A)
        self.A_fake_B = self.G_D_B(A_features_A)
        self.A_idt_A = self.G_D_A(A_features_B)

        A_features_rec = self.G_E_B(self.A_fake_B)
        self.A_short_rec_A = self.G_D_A(A_features_A)
        self.A_long_rec_A = self.G_D_A(A_features_rec)

        if self.mask_A is not None and self.do_seg:
            self.A_seg_ori = self.Seg(A_features_A)
            # self.A_seg_idt = self.Seg(A_features_B)
            self.A_seg_rec = self.Seg(A_features_rec)

    def forward_B(self):  # B to A
        B_features_B = self.G_E_B(self.B)
        B_features_A = self.G_E_A(self.B)
        self.B_fake_A = self.G_D_A(B_features_B)
        self.B_idt_B = self.G_D_B(B_features_A)

        B_features_rec = self.G_E_A(self.B_fake_A)
        self.B_short_rec_B = self.G_D_B(B_features_B)
        self.B_long_rec_B = self.G_D_B(B_features_rec)

        if self.mask_B is not None and self.do_seg:
            self.B_seg_ori = self.Seg(B_features_B)
            # self.B_seg_idt = self.Seg(B_features_A)
            self.B_seg_rec = self.Seg(B_features_rec)

    def forward(self):
        self.forward_A()
        self.forward_B()

    def get_loss_G(self):
        # A Loss
        self.G_A_idt_loss = self.loss_funcs['idt'](self.A_idt_A, self.A) * self.loss_rate['A'] * self.loss_rate['idt']
        G_A_short_rec_loss = self.loss_funcs['rec'](self.A_short_rec_A, self.A)
        G_A_long_rec_loss = self.loss_funcs['rec'](self.A_long_rec_A, self.A)
        self.G_A_rec_loss = (G_A_short_rec_loss + G_A_long_rec_loss) * self.loss_rate['A'] * self.loss_rate['rec']
        fake_A_pred = self.D_A(self.B_fake_A)
        self.G_A_gan_loss = self.loss_funcs['gan'](fake_A_pred, torch.tensor(1.0, device=fake_A_pred.device).expand_as(fake_A_pred)) * self.loss_rate['A'] * self.loss_rate['gan']
        self.G_A_loss = self.G_A_idt_loss + self.G_A_rec_loss + self.G_A_gan_loss

        # B Loss
        self.G_B_idt_loss = self.loss_funcs['idt'](self.B_idt_B, self.B) * self.loss_rate['B'] * self.loss_rate['idt']
        G_B_short_rec_loss = self.loss_funcs['rec'](self.B_short_rec_B, self.B)
        G_B_long_rec_loss = self.loss_funcs['rec'](self.B_long_rec_B, self.B)
        self.G_B_rec_loss = (G_B_short_rec_loss + G_B_long_rec_loss) * self.loss_rate['B'] * self.loss_rate['rec']
        fake_B_pred = self.D_B(self.A_fake_B)
        self.G_B_gan_loss = self.loss_funcs['gan'](fake_B_pred, torch.tensor(1.0, device=fake_B_pred.device).expand_as(fake_B_pred)) * self.loss_rate['B'] * self.loss_rate['gan']
        self.G_B_loss = self.G_B_idt_loss + self.G_B_rec_loss + self.G_B_gan_loss

        # Seg Loss
        self.seg_loss = None
        if self.mask_A is not None and self.do_seg:
            A_seg_ori_loss = self.loss_funcs['seg'](self.A_seg_ori, self.mask_A)
            # A_seg_idt_loss = self.loss_funcs['seg'](self.A_seg_idt, self.mask_A)
            A_seg_rec_loss = self.loss_funcs['seg'](self.A_seg_rec, self.mask_A)
            A_seg_scp_loss = self.loss_funcs['scp'](self.A_seg_ori, self.A_seg_rec)
            A_seg_loss = (A_seg_ori_loss + A_seg_rec_loss + A_seg_scp_loss) * self.loss_rate['A'] * self.loss_rate['seg']
            self.seg_loss = A_seg_loss
        if self.mask_B is not None and self.do_seg:
            B_seg_ori_loss = self.loss_funcs['seg'](self.B_seg_ori, self.mask_B)
            # B_seg_idt_loss = self.loss_funcs['seg'](self.B_seg_idt, self.mask_B)
            B_seg_rec_loss = self.loss_funcs['seg'](self.B_seg_rec, self.mask_B)
            B_seg_scp_loss = self.loss_funcs['scp'](self.B_seg_ori, self.B_seg_rec)
            B_seg_loss = (B_seg_ori_loss + B_seg_rec_loss + B_seg_scp_loss) * self.loss_rate['B'] * self.loss_rate['seg']
            self.seg_loss = B_seg_loss if self.seg_loss is None else self.seg_loss + B_seg_loss

        # Total Loss
        self.G_loss = self.G_A_loss + self.G_B_loss
        if self.seg_loss is not None:
            self.G_loss = self.G_loss + self.seg_loss

    def get_loss_D_A(self):
        true_A_pred = self.D_A(self.A)
        fake_A_pred = self.D_A(self.fake_A_pool.query(self.B_fake_A.detach()))
        D_A_true_loss = self.loss_funcs['gan'](true_A_pred, torch.tensor(1.0, device=true_A_pred.device).expand_as(true_A_pred))
        D_A_false_loss = self.loss_funcs['gan'](fake_A_pred, torch.tensor(0.0, device=fake_A_pred.device).expand_as(fake_A_pred))
        self.D_A_loss = (D_A_true_loss + D_A_false_loss) * self.loss_rate['A']

    def get_loss_D_B(self):
        true_B_pred = self.D_B(self.B)
        fake_B_pred = self.D_B(self.fake_B_pool.query(self.A_fake_B.detach()))
        D_B_true_loss = self.loss_funcs['gan'](true_B_pred, torch.tensor(1.0, device=true_B_pred.device).expand_as(true_B_pred))
        D_B_false_loss = self.loss_funcs['gan'](fake_B_pred, torch.tensor(0.0, device=fake_B_pred.device).expand_as(fake_B_pred))
        self.D_B_loss = (D_B_true_loss + D_B_false_loss) * self.loss_rate['B']

    def get_loss_D(self):
        self.get_loss_D_A()
        self.get_loss_D_B()

    def backward_G(self):
        self.G_loss.backward()

    def backward_D_A(self):
        self.D_A_loss.backward()

    def backward_D_B(self):
        self.D_B_loss.backward()

    def backward_D(self):
        self.backward_D_A()
        self.backward_D_B()

    def set_data(self, A, B, mask_A=None, mask_B=None):
        self.A = A
        self.B = B
        self.mask_A = mask_A
        self.mask_B = mask_B

    def run_train(self):
        # forward
        self.forward()

        # G and Seg backward
        self.set_requires_grad([self.D_A, self.D_B], False)
        self.optimizer_G.zero_grad()
        self.get_loss_G()
        self.backward_G()
        self.optimizer_G.step()

        # D model backward
        self.set_requires_grad([self.D_A, self.D_B], True)
        self.optimizer_D.zero_grad()
        self.get_loss_D()
        self.backward_D()
        self.optimizer_D.step()

    def run_val(self):
        with torch.no_grad():
            # forward
            self.forward()

            # G and Seg get loss
            self.get_loss_G()
            self.get_loss_D()

    def get_current_loss(self):
        return self.G_loss, self.G_A_loss, self.G_B_loss, self.seg_loss, self.D_A_loss, self.D_B_loss, self.G_A_idt_loss, self.G_A_rec_loss, self.G_A_gan_loss, self.G_B_idt_loss, self.G_B_rec_loss, self.G_B_gan_loss
