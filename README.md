# DARPA PAI - Method 1
Using DNN for inverse problems in imaging

# DARPA PAI - Method 2

## Installation

All the required dependencies are provided in the PAI.yml file in the folders "angle_regression" and "disentangled_vae". Please run the following command to install an anaconda environment: 
```python
git clone https://github.com/PhysicsOfAI/PhysicsAI4Imaging.git
cd Method2
conda env create -f PAI.yml
conda activate PAI
```
## Data 

There are three datasets we consider: a) without noise b) with noise of SNR 1.0 and c) with noise of SNR 0.5

Please add the folder "data" at the same level as "angle_regression" and "disentangled_vae". The structure of "data" has to be as follows: 
```
+-- data
|   +-- Haar
    |   +-- training
        |   +-- imagefile.mat
        |   +-- labelfile.mat
    |   +-- testing
        |   +--imagefile.mat
        |   +--labelfile.mat
    +--Haar_1
    |   +-- training
        |   +-- imagefile.mat
        |   +-- labelfile.mat
    |   +-- testing
        |   +--imagefile.mat
        |   +--labelfile.mat
    +--Haar_0.5
    |   +-- training
        |   +-- imagefile.mat
        |   +-- labelfile.mat
    |   +-- testing
        |   +--imagefile.mat
        |   +--labelfile.mat
```      
      
## Overview 

There are 2 modes of operation in method 2: 
1.  learning a DNN in a supervised way to perform angle estimation
2.  learning a VAE whose latent variables are interpretable euler angles

The commands to execute in order to run these codes are provided in angle_regression/main_scripts.scr and disentangled_vae/main_scripts.scr

There are 3 modes to run the code. 
1.  Training - This performs the training of the neural network and saves the model in ./checkpoints/ and tensorboard logs in ./runs/
2.  Testing - Here, inference is performed on the test set and example predictions are printed out on the command window (corresponding images of interpolations get saved in case of VAE)
              You need to provide the flag --test. 
3.  Get predictions - Here, data from a test folder (for ex: realImages) is used to perform inference and the predicted Euler angles/ Quaternions are saved back in the same folder as a .mat file.
                      You need to provide the flags --get_pred_only and --pred_folder_name `<insert test folder path>`


## Task 1 - Learn a DNN to regress the pose of the molecule

The code corresponding to this is provided in the folder "angle_regression". There are 2 modes of operation here: a) with Euler angles b) with Quaternions.
There are a total of 5 or 6 quantities that the neural network learns to regress depending on which mode (Euler vs Quaternions)
For Euler angles it is 3+2 = 5 quantities and for Quaternions it is 4+2 = 6 quantities. The 2 other values are image scale and pixel gain respectively. 

The loss curves corresponding to the three datasets are shown below thus indicating that the DNN is successfully able to learn a mapping from the images to the corresponding angles.

### Data without noise
![Train_curve](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/angle_regression/outputs/Haar/train_loss.png)

![Validation_curve](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/angle_regression/outputs/Haar/val_loss.png)

### Data with noise - SNR 1.0
![Train_curve](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/angle_regression/outputs/Haar_1/train_loss.png)

![Validation_curve](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/angle_regression/outputs/Haar_1/val_loss.png)

### Data with noise - SNR 0.5
![Train_curve](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/angle_regression/outputs/Haar_0.5/train_loss.png)

![Validation_curve](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/angle_regression/outputs/Haar_0.5/val_loss.png)

## Task 2 - Learn a VAE to jointly perform pose estimation and reconstruction of the molecule

Here the goal is the learn a generative model, specifically a VAE which can learn interpretable euler angles/quaternions in its latent space as well as be able to learn a decoder to map these angles back to the image space. 
The neural network architecture we use is as follows: 
### Network architecture 
![VAE_network_architecture](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/disentangled_vae/outputs_Haar/vae_arch.png)

The euler angles/quaternions are learned in a supervised manner at the latent space. 
### Original Images
![Original_image_grid](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/disentangled_vae/outputs_Haar/org_image.png)

### Reconstructed images
![Reconstructed_image_grid](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/disentangled_vae/outputs_Haar/vae_recon.png)

### Latent space interpolations
These represent VAE latent space interpolations. Every row represents the interpolation between the images in the first column and last column. The interpolation is done as follows: 

Let image1 and image2 be two images. Let (zf1, ze1) and (zf2, ze2) be the latent representations for image1 and image2 respectively. We then calculate z_interp as follows:

z_interp = alpha*zf1 + (1-alpha)*zf2, where alpha is varied between 0 and 1 in steps of 0.1. 

Now, each of these z_interp is concatenated with zf1 and fed to the decoder to obtain an "interpolated" image. It is interesting to note that zf1 is fixed across all images. 
This shows that the architecture is able to successfully disentangle explcit euler angles from their feature representations. 


![Latent_space_interpolations](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/disentangled_vae/outputs_Haar/interpolations.png)

### Training curves
Below shown are the loss curves for the Angle loss, KL divergence loss and L2 reconstruction loss for the VAE on the training set:

![Train_angle_loss](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/disentangled_vae/outputs_Haar/train_angle_loss.png)

![Train_kl_loss](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/disentangled_vae/outputs_Haar/train_kl_loss.png)

![Train_recon_loss](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/disentangled_vae/outputs_Haar/train_recon_loss.png)

### Validation loss
Below shown are the loss curves for the Angle loss, KL divergence loss and L2 reconstruction loss for the VAE on the validation set:

![Val_angle_loss](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/disentangled_vae/outputs_Haar/val_angle_loss.png)

![Val_kl_loss](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/disentangled_vae/outputs_Haar/val_kl_loss.png)

![Val_recon_loss](https://github.com/PhysicsOfAI/PhysicsAI4Imaging/blob/master/Method2/disentangled_vae/outputs_Haar/val_recon_loss.png)
