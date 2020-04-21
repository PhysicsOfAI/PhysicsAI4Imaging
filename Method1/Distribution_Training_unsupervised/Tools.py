import torch

def forward_operator_from_real(x, mask):
    """ Forward operator for real images
    :param x: real input image
    :param mask: mask of radial lines
    :return: x_new
    """
    x_new = torch.rfft(x, signal_ndim=3, onesided=False) / x.shape[1]
    x_new[:, :, :, 0] = torch.mul(torch.from_numpy(mask).float().cuda(), x_new[:, :, :, 0])
    x_new[:, :, :, 1] = torch.mul(torch.from_numpy(mask).float().cuda(), x_new[:, :, :, 1])
    x_new = torch.ifft(x_new, signal_ndim=3) * x.shape[1]

    return x_new


def forward_operator(x, mask):
    """ Forward operator for complex images
    :param x: complex input image
    :param mask: mask of radial lines
    :return: x_new
    """
    x_new = torch.fft(x, signal_ndim=3) / x.shape[1]
    x_new[:, :, :, 0] = torch.mul(torch.from_numpy(mask).float().cuda(), x_new[:, :, :, 0])
    x_new[:, :, :, 1] = torch.mul(torch.from_numpy(mask).float().cuda(), x_new[:, :, :, 1])
    x_new = torch.ifft(x_new, signal_ndim=3) * x.shape[1]

    return x_new


def TV2d(x):
    """ Total variation for real image
    :param x: real input image
    :return: The total variation of x
    """

    batchsize = x.shape[0]
    imSide = x.shape[1]

    x2d = x.view(batchsize, imSide, imSide, 2)
    x2d = torch.sqrt(x2d[:, :, :, 0]**2 + x2d[:, :, :, 1]**2)

    x1_short = x2d[:, 0:imSide - 1, 1:imSide]
    x_shift = x2d[:, 1:imSide, 1:imSide]
    x2_short = x2d[:, 1:imSide, 0:imSide - 1]
    sqDiff = torch.sqrt(((x_shift - x1_short) ** 2 + (x_shift - x2_short) ** 2) + 1e-9)
    sqDiff = sqDiff.view(batchsize, -1)

    return sqDiff.mean(dim=1).mean(dim=0)


def TV2d_c(x):
    """ Total variation for complex image
    :param x: complex input image
    :return: The total variation of x
    """
    batchsize = x.shape[0]
    imSide = x.shape[1]

    x2d = x.view(batchsize, imSide, imSide,2)
    x2dre = x2d[:, :, :, 0]
    x2dim = x2d[:, :, :, 1]

    x1_shortre = x2dre[:, 0:imSide - 1, 1:imSide]
    x_shiftre = x2dre[:, 1:imSide, 1:imSide]
    x2_shortre = x2dre[:, 1:imSide, 0:imSide - 1]
    sqDiffre = torch.sqrt(((x_shiftre - x1_shortre) ** 2 + (x_shiftre - x2_shortre) ** 2) + 1e-9)
    sqDiffre = sqDiffre.view(batchsize, -1)

    x1_shortim = x2dim[:, 0:imSide - 1, 1:imSide]
    x_shiftim = x2dim[:, 1:imSide, 1:imSide]
    x2_shortim = x2dim[:, 1:imSide, 0:imSide - 1]
    sqDiffim = torch.sqrt(((x_shiftim - x1_shortim) ** 2 + (x_shiftim - x2_shortim) ** 2) + 1e-9)
    sqDiffim = sqDiffim.view(batchsize, -1)
    sqDiffim = sqDiffre.mean(dim=1).mean(dim=0) + sqDiffim.mean(dim=1).mean(dim=0)

    return sqDiffim

