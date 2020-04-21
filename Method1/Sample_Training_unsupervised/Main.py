from DataTransform import *
import matplotlib.pyplot as plt
import numpy as np
import GetKMask
import torch
import Palindrome
import Architectures
import os

if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plt.ioff()

    # define whether to use CUDA
    USE_CUDA = False
    if torch.cuda.is_available():
        USE_CUDA = True
    print('\n\nUSE_CUDA = {}\n\n'.format(USE_CUDA))

    img_size = 128
    radial_lines = 44

    # dictionary of hyperparameters
    params_dict = {
        'img_size': img_size,  # length of image
        'batchsize': 1,  # number of samples in batch
        'grad_steps': 1,  # number of concatenated gradients
        'train_steps': 1,  # number of optimization steps
        'lr': 5e-3,  # learning rate
        'theta': 0.0005,  # weighting of regularizer
        'mask': GetKMask.createkSpaceMask(np.array([img_size, img_size]), radial_lines),  # mask of radial lines
        'optimizer_net': Architectures.Net,  # optimizer network architecture
        'load_model': False  # whether to start new or resume from last saved point
    }

    solver = Palindrome.LearnableSolver(params_dict)
    solver.run()


