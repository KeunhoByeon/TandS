import torch.nn as nn

from .generator_resnet import ResnetDecoder
from .u2net import U2NETP


class SegmentStacked(nn.Module):
    def __init__(self, middle_ch=1, num_classes=3):
        super(SegmentStacked, self).__init__()

        self.res_decoder = ResnetDecoder(output_nc=middle_ch)
        self.u2net_segment = U2NETP(in_ch=middle_ch, out_ch=num_classes)

    def forward(self, input):
        return self.u2net_segment(self.res_decoder(input))
