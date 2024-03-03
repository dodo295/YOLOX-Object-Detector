import sys
sys.path.append('./')

import torch.nn as nn
import torch
from model.backbone.darknet import Darknet53

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs):
        super().__init__()
        self.use_bn = use_batch_norm
        self.conv2d = nn.conv2d(in_channels, out_channels,  bias=not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.conv2d(x)
        if self.use_bn:
            x = self.bn(x)
            return self.activation(x) 
        else:
            return x


class YOLOFPN(nn.Module):
    def __ini__(self):
        super().__init__()
        self.backbone = Darknet53()
        # self.l_output, self.h_output, self.ssp_output = backbone(inputs)
        self.b1_conv = CNNBlock(512, 256, kernel_size=1)
        self.b1_convs_group = self.create_conv_group([256, 512], 512 + 256)

        self.b2_conv = CNNBlock(256, 128, kernel_size=1)
        self.b2_convs_group = self.create_conv_group([128, 256], 256 + 128)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def create_conv_group(self, in_channels_list, in_filters):
        m = nn.Sequential(
            CNNBlock(in_filters, in_channels_list[0], 1),
            CNNBlock(in_channels_list[0], in_channels_list[1], 3),
            CNNBlock(in_channels_list[1], in_channels_list[0], 1),
            CNNBlock(in_channels_list[0], in_channels_list[1], 3),
            CNNBlock(in_channels_list[1], in_channels_list[0], 1)
        )
        return m

    def forward(self, x):
        l_output, h_output, ssp_output = self.backbone(x)
        out1 = self.upsample(self.b1_conv(ssp_output))
        out1 = torch.cat([out1, h_output], dim=1)
        out1 = self.b1_convs_group(out1)

        out2 = self.upsample(self.b2_conv(out1))
        out2 = torch.cat([out1, l_output], dim=1)
        out2 = self.b1_convs_group(out2)

        return ssp_output, out1, out2
        





    