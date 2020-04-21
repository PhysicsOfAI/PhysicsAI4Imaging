import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import Tools
import Metrics
from torchvision import *
from DataTransform import *
from timeit import default_timer as timer
import os
from torch.utils.data import DataLoader


def scale(x):
    """ sample-wise scaling between 0 and 1 of images in batch
    :param sample: batch of input images
    :return: scaled batch of images
    """
    for i in range(x.size(0)):

        min = torch.min(x[i, :])
        max = torch.max(x[i, :]-min)
        x[i, :] = ((x[i, :] - min) / max)

    return x


def scale255(sample):
    """ scaling images between 0 and 255
    :param sample: batch of input images
    :return: scaled batch of images
    """
    min = np.min(sample)
    max = np.max(sample-min)

    return ((sample-min)/max)*255


class LearnableSolver:

    def __init__(self, params_dict):

        self.params_dict = params_dict

        self.img_size = params_dict['img_size']
        self.mask = params_dict['mask']
        self.lr = params_dict['lr']
        self.theta = params_dict['theta']
        self.batchsize = params_dict['batchsize']
        self.grad_steps = params_dict['grad_steps']
        self.train_steps = params_dict['train_steps']
        self.load_model = params_dict['load_model']
        self.optimizer_net = params_dict['optimizer_net'](2 * self.grad_steps, self.img_size).cuda()
        self.optimizer_net.apply(self.weights_init)

    def reconstruction_loss(self, x, y):

        loss = torch.mean((Tools.forward_operator(x, self.mask)-y)**2) + self.theta * Tools.TV2d_c(x)

        return loss

    def weights_init(self,m):
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1 or classname.find('Linear') != -1) and classname.find('Block') == -1:
            nn.init.constant_(m.weight.data, 0.0)
            if hasattr(m, 'bias') and not getattr(m, 'bias') is None:
                nn.init.zeros_(m.bias)

        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    def run(self):

        if self.load_model:
            self.optimizer_net.load_state_dict(torch.load("final_model.pth"))
            print("Loading model weights ... ")

        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(self.img_size), transforms.ToTensor()])
        newset = ImageTransform(os.path.join("Data/filenames.csv"),
                                os.path.join("Data/SVAE/train1k/RandSL1k"),
                                transform)
        train, val = split(newset, 1.0)  # two sub datasets

        # define optimizer
        optimizer = torch.optim.Adam(self.optimizer_net.parameters(), lr=self.lr)

        # load data
        data_train = DataLoader(train, batch_size=self.batchsize, shuffle=True)
        iterator_train = iter(data_train)
        num_iter = len(iterator_train)

        ssim_list = []
        psnr_list = []
        loss_list = []
        time_list = []

        for iteration in range(num_iter):

            print(f'\n==================> Batch: {iteration}')

            # initialize gradient list in order to allow a concatenation of gradients if desired
            self.grad_list = torch.zeros(self.batchsize, self.grad_steps, self.img_size, self.img_size, 2).cuda()

            orig = iterator_train.next()
            orig = scale(orig)

            # reference image
            ref = orig.detach().cpu().numpy().reshape([self.batchsize, self.img_size, self.img_size])

            # creating measurement, used as guidance
            y = Tools.forward_operator_from_real(orig.view(self.batchsize, self.img_size, self.img_size).cuda(), self.mask)
            y.requires_grad = False  # don't need grad in y

            # initialize x with measurement
            x = y.clone().data
            x = x.cuda()
            x.requires_grad = True

            # ----- start optimization process -----

            for steps in range(self.train_steps):

                print(f'\n=======> Iteration: {steps}')

                # compute gradient with respect to x
                loss = self.reconstruction_loss(x, y)
                loss.backward(retain_graph=False)
                grad = x.grad

                # concatenation of gradients in case of more than one training step
                self.grad_list = torch.cat((self.grad_list[:, 1:, :, :, :], torch.unsqueeze(grad, 1)), dim=1)

                # define initial x
                x_init = x.clone().data

                # start timer for monitoring optimization time
                start = timer()

                optimize = True
                loss_last = np.inf

                while optimize:

                    # adapt gradient by using an optimization network
                    adapt_grad = self.optimizer_net(self.grad_list)

                    # apply update to x
                    x = x - torch.squeeze(adapt_grad)

                    # compute gradient for theta
                    loss = self.reconstruction_loss(x, y)
                    loss.backward(retain_graph=False)

                    # check for convergence
                    if abs(loss_last - loss.item()) <= 1e-10:
                        optimize = False
                    else:
                        loss_last = loss.item()

                    # update network parameters theta
                    optimizer.step()
                    optimizer.zero_grad()

                    # reset x to initial
                    x = x_init.detach()
                    x.requires_grad = True

                # compute and apply final update to x
                adapt_grad = self.optimizer_net(self.grad_list)
                x = x - torch.squeeze(adapt_grad)
                time = timer() - start

                x = x.detach()
                x.requires_grad = True
                x.grad = torch.zeros_like(grad)

                # reshape x and y
                x_view = torch.sqrt(x[0,:, :, 0] ** 2 + x[0,:, :, 1] ** 2).detach().cpu().numpy().reshape(
                    [self.img_size, self.img_size])
                y_view = torch.sqrt(y[0,:, :, 0] ** 2 + y[0,:, :, 1] ** 2).detach().cpu().numpy().reshape(
                    [self.img_size, self.img_size])

                print('-> time consuming {:.3f}s'.format(time))
                print('loss:', loss.detach().cpu().numpy())
                print('ssim:', Metrics.ssim(x_view, ref[0]))
                print('psnr:', Metrics.psnr(x_view, ref[0]))

                ssim_list.append(Metrics.ssim(x_view, ref[0]))
                psnr_list.append(Metrics.psnr(x_view, ref[0]))
                loss_list.append(loss.detach().cpu().numpy())
                time_list.append(time)

                with open('ssim_list.npy', 'wb') as f:
                    np.save(f, ssim_list)
                with open('psnr_list.npy', 'wb') as f:
                    np.save(f, psnr_list)
                with open('loss_list.npy', 'wb') as f:
                    np.save(f, loss_list)
                with open('time_list.npy', 'wb') as f:
                    np.save(f, time_list)

                plt.subplot(1, 3, 1)
                plt.title("Reconstruction")
                plt.imshow(x_view, cmap='gray')
                plt.subplot(1, 3, 2)
                plt.title("Measurement")
                plt.imshow(y_view, cmap='gray')
                plt.subplot(1, 3, 3)
                plt.title("Target")
                plt.imshow(ref[0], cmap='gray')
                plt.show()

            # torch.save(self.optimizer_net.state_dict(), 'final_model.pth')
