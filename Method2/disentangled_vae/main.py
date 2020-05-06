#%%
import os
cwd = os.getcwd() 
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils
import numpy as np
import os.path
from scipy.io import loadmat
from vae_models import *
from utils import *
from args_python import *
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import hdf5storage

EulerN=3
QuaternionN=4
ScaleSpaceAndGainN=2

class CustomDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

#%%

def train(args, model, device, train_loader, optimizer, epoch, writer, Rbeta, zipped_vals, scheduler, kl_weight=None, anneal_rate=None):

    
    model.train()
    run_angle_loss = 0.0
    run_recon_loss = 0.0
    run_kl_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        mu, logvar, angle_gain_scale, in_data, output = model(data)

        if args.UseQuaternionNotEuler:
            R_est = quaternion2R(angle_gain_scale[:,0:QuaternionN])
            R_target = quaternion2R(target[:,0:QuaternionN])

            gt, pred, rot_loss = getlossrotation(True, R_est, R_target)
            gain_scale_loss = getlossspacescale(angle_gain_scale[:,QuaternionN],target[:,QuaternionN]) + getlossgain(angle_gain_scale[:,QuaternionN+1],target[:,QuaternionN+1])
            angle_loss = rot_loss + gain_scale_loss
        else:
            R_est = euler2R(angle_gain_scale[:,0:EulerN])
            R_target = euler2R(target[:,0:EulerN])

            gt, pred, rot_loss = getlossrotation(True, R_est, R_target)
            gain_scale_loss = getlossspacescale(angle_gain_scale[:,EulerN],target[:,EulerN]) + getlossgain(angle_gain_scale[:,EulerN+1],target[:,EulerN+1])
            angle_loss = rot_loss + gain_scale_loss

        recon_loss = nn.MSELoss()(torch.flatten(output,1), torch.flatten(in_data,1))
        kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)))/data.shape[0]

        if args.test:
            print("Ground truth : {}  \n  Predicted values : {}".format(torch.transpose(gt,1,2), pred))
            # also need to show reconstructed images
            break

        run_angle_loss += angle_loss.item()
        run_recon_loss += recon_loss.item()
        run_kl_loss += kl_loss.item()

        kl_weight = min(1.0, kl_weight + anneal_rate) 

        tot_loss = args.coeff_angle_loss*angle_loss + args.coeff_recon_loss*recon_loss + kl_weight*kl_loss
        tot_loss.backward()

        optimizer.step()
        scheduler.step()

        if (batch_idx+1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAngle Loss: {:.8f}, Recon loss: {:.8f}, KL Loss: {:.8f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx * len(data) / len(train_loader.dataset), run_angle_loss/args.log_interval, run_recon_loss/args.log_interval, run_kl_loss/args.log_interval))

            writer.add_scalar('Training/Angle_loss', run_angle_loss/args.log_interval, epoch*len(train_loader)+batch_idx)
            writer.add_scalar('Training/Reconstruction_loss', run_recon_loss/args.log_interval, epoch*len(train_loader)+batch_idx)
            writer.add_scalar('Training/KL_loss', run_kl_loss/args.log_interval, epoch*len(train_loader)+batch_idx)
            
            writer.add_graph(model, data)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(tag, value.detach().cpu().numpy(), batch_idx+1)
                writer.add_histogram(tag+'/grad', value.grad.detach().cpu().numpy(), batch_idx+1)

            run_angle_loss = 0.0
            run_recon_loss = 0.0
            run_kl_loss = 0.0

    
    return kl_weight

def validate(args, model, device, val_loader, Rbeta, zipped_vals):

    model.eval()
    val_angle_loss = 0.0
    val_recon_loss = 0.0
    val_kl_loss = 0.0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            mu, logvar, angle_gain_scale, in_data, output = model(data)

            if args.UseQuaternionNotEuler:
                R_est = quaternion2R(angle_gain_scale[:,0:QuaternionN])
                R_target = quaternion2R(target[:,0:QuaternionN])

                gt, pred, rot_loss = getlossrotation(True, R_est, R_target)
                gain_scale_loss = getlossspacescale(angle_gain_scale[:,QuaternionN],target[:,QuaternionN]) + getlossgain(angle_gain_scale[:,QuaternionN+1],target[:,QuaternionN+1])
                loss_value = rot_loss + gain_scale_loss

            else:
                R_est = euler2R(angle_gain_scale[:,0:EulerN])
                R_target = euler2R(target[:,0:EulerN])

                gt, pred, rot_loss = getlossrotation(True, R_est, R_target)
                gain_scale_loss = getlossspacescale(angle_gain_scale[:,EulerN],target[:,EulerN]) + getlossgain(angle_gain_scale[:,EulerN+1],target[:,EulerN+1])
                loss_value = rot_loss + gain_scale_loss

            val_angle_loss += loss_value
            val_recon_loss += nn.MSELoss()(torch.flatten(output,1), torch.flatten(in_data,1))
            val_kl_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))

    val_angle_loss /= len(val_loader)
    val_recon_loss /= len(val_loader)
    val_kl_loss /= len(val_loader)

    print('\nValidation set: Angle loss: {:.8f}, Recon loss: {:.8f}, KL loss: {:.8f}\n'.format(val_angle_loss.item(), val_recon_loss.item(), val_kl_loss.item()))

    if args.test:
        print("Ground truth : {}    \n\n    Predicted values : {} \n".format(torch.transpose(gt,1,2), pred))

    return val_angle_loss, val_recon_loss, val_kl_loss

def test(args, model, device, test_loader, Rbeta, zipped_vals, data_stat):        

    if args.get_pred_only:

        model.eval()
        test_out_list = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                _, _, output, _, _ = model(data)
                test_out_list.append(output.detach().numpy())

        save_mat = np.concatenate(test_out_list)

        hdf5storage.savemat(args.pred_folder+'/pred_labels.mat', {'labeldata':save_mat})     

    else:

        model.eval()
        test_angle_loss = 0.0
        test_recon_loss = 0.0
        test_kl_loss = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                mu, logvar, angle_gain_scale, in_data, output = model(data)

                if args.UseQuaternionNotEuler:
                    R_est = quaternion2R(angle_gain_scale[:,0:QuaternionN])
                    R_target = quaternion2R(target[:,0:QuaternionN])

                    gt, pred, rot_loss = getlossrotation(True, R_est, R_target)
                    gain_scale_loss = getlossspacescale(angle_gain_scale[:,QuaternionN],target[:,QuaternionN]) + getlossgain(angle_gain_scale[:,QuaternionN+1],target[:,QuaternionN+1])
                    loss_value = rot_loss + gain_scale_loss

                else:
                    R_est = euler2R(angle_gain_scale[:,0:EulerN])
                    R_target = euler2R(target[:,0:EulerN])

                    gt, pred, rot_loss = getlossrotation(True, R_est, R_target)
                    gain_scale_loss = getlossspacescale(angle_gain_scale[:,EulerN],target[:,EulerN]) + getlossgain(angle_gain_scale[:,EulerN+1],target[:,EulerN+1])
                    loss_value = rot_loss + gain_scale_loss
                
                test_angle_loss += loss_value
                test_recon_loss += nn.MSELoss()(torch.flatten(output,1), torch.flatten(in_data,1))
                test_kl_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))

        test_angle_loss /= len(test_loader)
        test_recon_loss /= len(test_loader)
        test_kl_loss /= len(test_loader)

        print('\nTest set: Angle loss: {:.8f}, Recon loss: {:.8f}, KL loss: {:.8f}\n'.format(test_angle_loss.item(), test_recon_loss.item(), test_kl_loss.item()))

        if args.test:
            print("Ground truth : {}    \n\n    Predicted values : {} \n".format(torch.transpose(gt,1,2), pred))

        for data, target in test_loader:
        
            data, target = data.to(device), target.to(device)
            _, _, _, in_data, output_net = model(data)
            
            output = torch.reshape(output_net[0:64,:], (64, data.shape[1], data.shape[2], data.shape[3]))
            data_unnorm = torch.zeros_like(data)
            data_unnorm[0:64,:,:,:] = data[0:64,:,:,:]*data_stat[1] + data_stat[0]
            output[0:64,:,:,:] = output[0:64,:,:,:]*data_stat[1] + data_stat[0]
            
            grid = torchvision.utils.make_grid(data_unnorm[0:64,:,:,:].detach())
            matplotlib_imshow(grid, name="org_image.png", one_channel=True)
            grid = torchvision.utils.make_grid(output[0:64,:,:,:].detach())
            matplotlib_imshow(grid, name="vae_recon.png", one_channel=True)

            # latent space interpolation between 2 images from test-dataset

            start = [0,2,4,6,8,10]
            dest = [30,32,34,36,38,40]
            alpha = np.linspace(0,1,11)
            dec_out = torch.zeros((len(alpha)*len(start), data.shape[1], data.shape[2], data.shape[3]))

            for ii in range(len(start)):
                
                data_interp1 = torch.unsqueeze(data[start[ii],:,:,:],dim=0)
                data_interp2 = torch.unsqueeze(data[dest[ii],:,:,:],dim=0)

                z_mu1, z_logvar1, z_euler1, _, dec_out1 = model(data_interp1)
                z_mu2, z_logvar2, z_euler2, _, dec_out2 = model(data_interp2)

                std1 = torch.exp(0.5*z_logvar1)
                eps1 = torch.randn_like(std1)
                rep1 = z_mu1 + eps1*std1 

                std2 = torch.exp(0.5*z_logvar2)
                eps2 = torch.randn_like(std2)
                rep2 = z_mu2 + eps2*std2

                for a in range(len(alpha)):
                    z_euler_interp = (1-alpha[a])*z_euler1 + alpha[a]*z_euler2
                    
                    feat_vec_interp = torch.cat([z_euler_interp, rep2],dim=1)

                    lin_out_interp = model.dec_in(feat_vec_interp)

                    lin_out_interp = torch.reshape(lin_out_interp, (-1,model.ch_factor_6out6,7,7))
                    d_out = model.decoder(lin_out_interp)
                    dec_out[(len(alpha)*ii)+a,:,:,:] = torch.squeeze(d_out)
            
            # random sampling of latent space by fixing euler angle fed to it. 

            dec_sample = torch.zeros_like(data[0:len(alpha),:,:,:])
            for idx in range(len(alpha)):
                rep_sample =  torch.randn_like(std1)
                z_euler1 = torch.tensor([[0.39644146, 0.75766391, 0.77631556, 1.00424026, 1.0780347]], device=rep_sample.device)
                feat_vec_interp = torch.cat([z_euler1, rep_sample],dim=1)

                lin_out_interp = model.dec_in(feat_vec_interp)
                lin_out_interp = torch.reshape(lin_out_interp, (-1,model.ch_factor_6out6,7,7))
                d_out = model.decoder(lin_out_interp)

                dec_sample[idx,:,:,:] = torch.squeeze(d_out)


            grid = torchvision.utils.make_grid(dec_out.detach(),nrow=len(alpha))
            matplotlib_imshow(grid, name="interpolations.png", one_channel=True)

            grid = torchvision.utils.make_grid(dec_sample.detach(),len(alpha))
            matplotlib_imshow(grid, name="sample_from_gaussian.png", one_channel=True)
            
            break

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: 600)')
    parser.add_argument('--no-cuda', action='store_false', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--arch', default='EulerGainConvVAE',help='the architecture to use. options are VGG, MLP for now. Can add more')

    parser.add_argument('--UseQuaternionNotEuler', action='store_true', default=False, help='give this flag in order to use the Quaternion representation, otherwise the Euler angles representation will be used')
    parser.add_argument('--ScaleSpaceMin', type=float, default=0.8, help='minimum value of the space scaling')
    parser.add_argument('--ScaleSpaceMax', type=float, default=1.2, help='maximum value of the space scaling')
    parser.add_argument('--GainMin', type=float, default=0.8, help='minimum value of the gain')
    parser.add_argument('--GainMax', type=float, default=1.2, help='maximum value of the gain')
    parser.add_argument('--RootDirectory4Data', default='./', help='the name of the root director for the data')
    parser.add_argument('--carve_val', action='store_false', default=True, help='Whether validation set has to be carved out from the training set. Default is true')
    parser.add_argument('--test', action='store_true', default=False, help='Whether train or test mode. Default is train mode.')
    parser.add_argument('--coeff_angle_loss', type=float, default=1, help='Lagrangian multiplier for the angle loss term')
    parser.add_argument('--coeff_recon_loss', type=float, default=2, help='Lagrangian multiplier for the reconstruction loss term')
    parser.add_argument('--coeff_kl_loss', type=float, default=1, help='Lagrangian multiplier for the KL divergence loss term')
    parser.add_argument('--get_pred_only', action='store_true', default=False, help='Get only predictions from images')
    parser.add_argument('--pred_folder',  default='./', help='Directory of file with test images.')

    args = parser.parse_args()

    # args=Args()    
    #     
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    trainingdirectory = args.RootDirectory4Data+"/"+"training"
    trainingimagefile="imagefile.mat"
    traininglabelfile="labelfile.mat"
    train_images = hdf5storage.loadmat(os.path.join(trainingdirectory, trainingimagefile))['imagedata']
    train_labels = hdf5storage.loadmat(os.path.join(trainingdirectory, traininglabelfile))['labeldata']

    if args.carve_val: 
            print("Carving out validation set from training set")
            train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)
    else:
        print("Loading validation set")
        validationdirectory = args.RootDirectory4Data+"/"+"validation"

        validationimagefile="imagefile.mat"
        validationlabelfile="labelfile.mat"

        val_images = hdf5storage.loadmat(os.path.join(validationdirectory, validationimagefile))['imagedata']
        val_labels = hdf5storage.loadmat(os.path.join(validationdirectory, validationlabelfile))['labeldata']

    train_images = np.expand_dims(train_images,1)
    val_images = np.expand_dims(val_images,1)
    
    mean = np.mean(train_images)
    std = np.std(train_images)
    data_stat = [mean, std]

    print("Dataset mean is {}".format(mean))
    print("Dataset std is {}".format(std))

    norm_train_images = (train_images - mean)/std
    norm_val_images = (val_images - mean)/std

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(norm_train_images), torch.Tensor(train_labels))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_dataset = torch.utils.data.TensorDataset(torch.Tensor(norm_val_images), torch.Tensor(val_labels))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # torch.autograd.set_detect_anomaly(True)
    
    if args.arch == "EulerGainConvVAE":
        model = EulerGainConvVAE(args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5, cycle_momentum=False, steps_per_epoch=len(train_loader), epochs=100)

    '''
    STILL IN DEVELOPMENT

    if args.arch == "EulerGainVAE":
        model = EulerGainVAE(args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5, cycle_momentum=False, steps_per_epoch=len(train_loader), epochs=args.epochs)
    if args.arch == "EulerGainConvVAE2":
        model = EulerGainConvVAE2(args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5, cycle_momentum=False, steps_per_epoch=len(train_loader), epochs=100)    
    '''

    if args.UseQuaternionNotEuler:
        ckpts_dir_name = f"checkpoints{args.RootDirectory4Data[7:]}/Quaternion_{args.epochs}/{args.arch}"
        log_dir = f"runs{args.RootDirectory4Data[7:]}/Quaternion_{args.epochs}/{args.arch}"
    else:
        ckpts_dir_name = f"checkpoints{args.RootDirectory4Data[7:]}/Euler_{args.epochs}/{args.arch}"
        log_dir = f"runs{args.RootDirectory4Data[7:]}/Euler_{args.epochs}/{args.arch}"


    # load data

    if args.get_pred_only:

        testingdirectory = args.pred_folder
        testingimagefile="imagefile.mat"

        test_images = hdf5storage.loadmat(os.path.join(testingdirectory, testingimagefile))['imagedata']
        test_images = np.expand_dims(test_images,1)

        norm_test_images = (test_images - mean)/std
        test_dataset = torch.utils.data.TensorDataset(torch.Tensor(norm_test_images))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        model.load_state_dict(torch.load(f"{ckpts_dir_name}/angle_regress_{args.epochs}.pt")) # make sure to load from latest checkpoint

        print("Test set predictions\n")
        zipped_vals = None
        Rbeta = None
        test(args, model, device, test_loader, Rbeta, zipped_vals, data_stat)
    
    else:

        testingdirectory = args.RootDirectory4Data+"/"+"testing"
        testingimagefile="imagefile.mat"
        testinglabelfile="labelfile.mat"
        test_images = hdf5storage.loadmat(os.path.join(testingdirectory, testingimagefile))['imagedata']
        test_labels = hdf5storage.loadmat(os.path.join(testingdirectory, testinglabelfile))['labeldata']
    
        test_images = np.expand_dims(test_images,1)
        norm_test_images = (test_images - mean)/std

        test_dataset = torch.utils.data.TensorDataset(torch.Tensor(norm_test_images), torch.Tensor(test_labels))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)        
        
        # model.load_state_dict(torch.load(f"{ckpts_dir_name}/angle_regress_300.pt")) # make sure to load from latest checkpoint

        os.makedirs(ckpts_dir_name, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir,flush_secs=10)

        Rbeta=None
        zipped_vals=None
        
        if not args.test:

            kl_weight = 0.1
            anneal_rate = (1.0 - 0.1) / (10 * len(train_loader))
            for epoch in range(1, args.epochs + 1):
                if (epoch-1)%100==0:
                    kl_weight=0.1
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5, cycle_momentum=False, steps_per_epoch=len(train_loader), epochs=100)

                kl_weight = train(args, model, device, train_loader, optimizer, epoch, writer, Rbeta, zipped_vals, scheduler, kl_weight, anneal_rate)
                val_angle_loss, val_recon_loss, val_kl_loss = validate(args, model, device, val_loader, Rbeta, zipped_vals)
                writer.add_scalar('Validation/Angle_loss', val_angle_loss, epoch)
                writer.add_scalar('Validation/Reconstruction_loss', val_recon_loss, epoch)
                writer.add_scalar('Validation/KL_loss', val_kl_loss, epoch)
                if epoch%(args.epochs/10)==0:
                    torch.save(model.state_dict(),f"{ckpts_dir_name}/angle_regress_{epoch}.pt")

            writer.close()

        else:

            model.load_state_dict(torch.load(f"{ckpts_dir_name}/angle_regress_{args.epochs}.pt")) # make sure to load from latest checkpoint
            print("Test set predictions\n")
            test(args, model, device, test_loader, Rbeta, zipped_vals, data_stat)
        
        
        
if __name__ == '__main__':
    main()


#%% 
#####################################
# visualize few samples of the data
#####################################

# trainingdirectory="./data_big_Haar0.2/training"
# testingdirectory="./data_big_Haar0.2/testing"

# trainingimagefile="imagefile.mat"
# testingimagefile="imagefile.mat"

# traininglabelfile="labelfile.mat"
# testinglabelfile="labelfile.mat"

# #read the Matlab .mat files

# train_images = hdf5storage.loadmat(os.path.join(trainingdirectory, trainingimagefile))['imagedata']
# train_labels = hdf5storage.loadmat(os.path.join(trainingdirectory, traininglabelfile))['labeldata']
# test_images = hdf5storage.loadmat(os.path.join(testingdirectory, testingimagefile))['imagedata']
# test_labels = hdf5storage.loadmat(os.path.join(testingdirectory, testinglabelfile))['labeldata']

# train_images = np.expand_dims(train_images,1)
# test_images = np.expand_dims(test_images,1)

# use_cuda = False
# kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
# train_dataset = CustomDataset(tensors=(torch.Tensor(train_images), torch.Tensor(train_labels)))
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=24, shuffle=True, **kwargs, drop_last=False)

# for batch_idx, (data, target) in enumerate(train_loader):
#     # print(data.shape)
#     grid = torchvision.utils.make_grid(data)
#     matplotlib_imshow(grid, one_channel=True)
#     # print(target.numpy())
#     break

# %%
###########################
# train test split of data
###########################

# import numpy as np
# import hdf5storage
# from sklearn.model_selection import train_test_split
# images = hdf5storage.loadmat("imagefile1.mat")["imagedata"]
# labels = hdf5storage.loadmat("labelfile.mat")["labeldata"]
# train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1, random_state=42)
# print(train_images.shape)
# print(test_images.shape)
# print(train_labels.shape)
# print(test_labels.shape)
# hdf5storage.savemat('./training/imagefile.mat',{"imagedata":train_images})
# hdf5storage.savemat('./training/labelfile.mat',{"labeldata":train_labels})
# hdf5storage.savemat('./testing/imagefile.mat',{"imagedata":test_images})
# hdf5storage.savemat('./testing/labelfile.mat',{"labeldata":test_labels})

# test_images, val_images, test_labels, val_labels = train_test_split(val_images, val_labels, test_size=0.5, random_state=42)
# hdf5storage.savemat('./validation/imagefile.mat',{"imagedata":val_images})
# hdf5storage.savemat('./validation/labelfile.mat',{"labeldata":val_labels})


# %%
###########################
# read from tf events file
###########################

# import numpy as np
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# import matplotlib as mpl
# import matplotlib.pyplot as plt

# import hdf5storage

# tf_size_guidance = {'scalars':10000}

# event_acc = EventAccumulator('./events.out.tfevents.1582062336.superman.11982.0', tf_size_guidance) 
# event_acc.Reload()

# training_accuracies = event_acc.Scalars('training_loss')
# validation_accuracies = event_acc.Scalars('validation_loss')

# steps_train = len(training_accuracies)
# y_train = np.zeros([steps_train, 1])

# steps_val = len(validation_accuracies)
# y_val = np.zeros([steps_val, 1])

# for i in range(steps_train):
#     y_train[i, 0] = training_accuracies[i][2] # value

# for i in range(steps_val):
#     y_val[i, 0] = validation_accuracies[i][2] # value

# hdf5storage.savemat('./training_curve.mat',{'values':y_train})
# hdf5storage.savemat('./validation_curve.mat',{'values':y_val})

#%%
######################################################
# plot train val curves with x, y labels and title
######################################################

# import numpy as np
# from matplotlib import pyplot as plt

# train_file_name = 'Training_recon_loss.csv'
# val_file_name = 'Val_recon_loss.csv'

# train_data = np.genfromtxt(train_file_name, delimiter=',')
# train_data = train_data[1:,:]

# val_data = np.genfromtxt(val_file_name, delimiter=',')
# val_data = val_data[1:,:]

# plt.figure()
# plt.plot(train_data[:,1], train_data[:,2])
# plt.ylim(0.01,0.1)
# plt.title('Training Reconstruction loss curve')
# plt.xlabel("Iterations")
# plt.ylabel("Loss")
# plt.savefig('train_recon_loss.png')

# plt.figure()
# plt.plot(val_data[:,1], val_data[:,2])
# plt.ylim(0.01,0.1)
# plt.title('Validation Reconstruction loss curve')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.savefig('val_recon_loss.png')


# %%

# visualize real samples

#import torch, torchvision, hdf5storage
#from matplotlib import pyplot as plt

#images = hdf5storage.loadmat('imagefile.mat')['imagedata']
#img_tensor = torch.Tensor(images[0:64,:])
#img_tensor = torch.unsqueeze(img_tensor, 1)

#grid = torchvision.utils.make_grid(img_tensor, nrow=8)
#grid = grid.mean(dim=0)

#npimg = grid.cpu().numpy()
#str_name = "real_images.png"
#plt.imsave(str_name, npimg, cmap="Greys")