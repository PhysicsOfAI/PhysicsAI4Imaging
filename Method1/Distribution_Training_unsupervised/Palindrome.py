import torch
from torch import nn
from matplotlib import pyplot as plt
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

class Learnable_Solver():

    def __init__(self, Params_dict):

        self.params_dict = Params_dict

        self.img_size = Params_dict['img_size']
        self.mask = Params_dict['mask']
        self.lr = Params_dict['lr']
        self.theta = Params_dict['theta']
        self.batchsize = Params_dict['batchsize']
        self.grad_steps = Params_dict['grad_steps']
        self.train_steps = Params_dict['train_steps']
        self.load_model = Params_dict['load_model']
        self.optimizer_net = Params_dict['optimizer_net'](2 * self.grad_steps).cuda()
        self.optimizer_net.apply(self.weights_init)

    def reconstruction_loss(self,x, y):

        loss = torch.mean((Tools.forward_operator(x, self.mask)-y)**2) + self.theta * Tools.TV2d_c(x)

        return loss

    def weights_init(self,m):

        classname = m.__class__.__name__
        if (classname.find('Conv') != -1 or classname.find('Linear') != -1 ) and classname.find('Block') == -1 :
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m,'bias') and not getattr(m,'bias') is None: nn.init.zeros_(m.bias)

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
        train, val = split(newset, 0.8)  # two sub datasets

        optimizer = torch.optim.Adam(self.optimizer_net.parameters(), lr=self.lr)

        for epoch in range(1000):

            print(f'\n==================> Epoch: {epoch}')

            data_train = DataLoader(train, batch_size=self.batchsize, shuffle=True)
            iterator_train = iter(data_train)
            num_iter = len(iterator_train)


            for iteration in range(num_iter):

                print(f'\n==================> Batch: {iteration}')

                orig = iterator_train.next()
                orig = scale(orig)

                # initialize gradient list in order to allow a concatenation of gradients if desired
                self.grad_list = torch.zeros(orig.size(0), self.grad_steps, self.img_size, self.img_size, 2).cuda()

                # reference image
                ref = orig.detach().cpu().numpy().reshape([orig.size(0), self.img_size, self.img_size])

                # creating measurement, used as guidance
                y = Tools.forward_operator_from_real(orig.view(orig.size(0), self.img_size, self.img_size).cuda(), self.mask)

                y.requires_grad = False  # don't need grad in y

                # initialize x with measurement
                x = y.clone().data
                x = x.cuda()
                x.requires_grad = True

                for steps in range(self.train_steps):

                    # print(f'\n=======> Iteration: {steps}')

                    start = timer()

                    # concatenation of gradients in case of more than one training step
                    self.grad_list = torch.cat((self.grad_list[:, 1:, :, :, :], torch.unsqueeze(x.detach(), 1)), dim=1)

                    # adapt gradient by using an optimization network
                    adapt_grad = self.optimizer_net(self.grad_list)

                    # apply update to x
                    x = x - torch.squeeze(adapt_grad)

                    # compute gradient for theta
                    loss = self.reconstruction_loss(x, y)
                    loss.backward()

                    # update theta
                    optimizer.step()
                    optimizer.zero_grad()

                    time = timer() - start

                    x_view = torch.sqrt(x[0,:, :, 0] ** 2 + x[0,:, :, 1] ** 2).detach().cpu().numpy().reshape(
                        [self.img_size, self.img_size])
                    y_view = torch.sqrt(y[0,:, :, 0] ** 2 + y[0,:, :, 1] ** 2).detach().cpu().numpy().reshape(
                        [self.img_size, self.img_size])

                    print('-> time consuming {:.3f}s'.format(time))
                    print('loss:', loss.detach().cpu().numpy())
                    print('ssim:', Metrics.ssim(x_view, ref[0]))
                    print('psnr:', Metrics.psnr(x_view, ref[0]))

                    if steps == self.train_steps-1 and epoch == 20:

                        plt.subplot(1, 3, 1)
                        plt.title("Reconstruction")
                        plt.imshow(x_view, cmap='gray')
                        plt.subplot(1, 3, 2)
                        plt.title("Measurement")
                        plt.imshow(y_view, cmap='gray')
                        plt.subplot(1, 3, 3)
                        plt.title("Target")
                        plt.imshow(ref[0], cmap='gray')

                torch.save(self.optimizer_net.state_dict(), 'final_model.pth')