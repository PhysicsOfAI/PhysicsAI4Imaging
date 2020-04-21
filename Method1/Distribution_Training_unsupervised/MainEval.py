from DataTransform import *
import numpy as np
import GetKMask
import torch
import Eval
import Architectures
import os

train = True

if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DEFINE whether to use CUDA
    USE_CUDA = False
    if torch.cuda.is_available():
        USE_CUDA = True
    print('\n\nUSE_CUDA = {}\n\n'.format(USE_CUDA))

    img_size = 128
    radial_lines = 44

    Params_dict = {
        'img_size': img_size,  # length of image
        'batchsize': 1,  # number of samples in batch
        'grad_steps': 1,  # number of concatenated gradients
        'train_steps': 1,  # number of optimization steps
        'theta': 0.005,  # weighting of regularizer
        'mask': GetKMask.createkSpaceMask(np.array([img_size, img_size]), radial_lines),  # mask of radial lines
        'optimizer_net': Architectures.UNet,  # optimizer network architecture
        'load_model': True  # whether to start new or resume from last saved point
    }

    solver = Eval.Learnable_Solver(Params_dict)
    solver.run()


