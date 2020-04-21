import torch.nn as nn


def single_conv_k1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, padding=0),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
    )


class Net(nn.Module):

    def __init__(self, u_steps, im_side):
        super().__init__()

        self.im_side = im_side

        self.prep = single_conv_k1(u_steps, 2)
        self.fcn1 = nn.Linear(im_side * im_side * 2, 1)
        self.fcn2 = nn.Linear(1, im_side * im_side * 2)

    def forward(self, x):

        batchsize = x.shape[0]
        shape = x.shape

        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape([batchsize, 2 * shape[1], shape[2], shape[3]])

        x = self.prep(x)
        x = x.view(-1, self.im_side**2 * 2)
        x = self.fcn1(x)
        x = self.fcn2(x)

        x = x.view(-1, 2, self.im_side, self.im_side)
        x = x.permute(0, 2, 3, 1)

        return x
