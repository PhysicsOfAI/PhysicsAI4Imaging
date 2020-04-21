import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, padding=0),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 1, padding=0),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )

def double_conv_k1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, padding=0),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 1, padding=0),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, u_steps):
        super().__init__()

        self.prep = double_conv_k1(u_steps, 64)
        self.dconv_down1 = double_conv(64, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 2, 1)

    def forward(self, x):

        batchsize = x.shape[0]
        shape = x.shape

        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape([batchsize, 2 * shape[1], shape[2], shape[3]])
        x = self.prep(x)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)

        out = x.permute(0, 2, 3, 1)

        return out


