import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import loadmat
from random import randint

def matplotlib_imshow(img, name, one_channel=False):
    
    if one_channel:
        img = img.mean(dim=0)

    npimg = img.cpu().numpy()
    str_name = "./outputs/" + name
    plt.imsave(str_name, npimg, cmap="Greys")

def euler2R(abc):
    
    cosabc=torch.cos(abc)
    sinabc=torch.sin(abc)

    R=torch.zeros((abc.shape[0],3,3), device=abc.device)
    
    R[:,0,0] = cosabc[:,0]*cosabc[:,1]*cosabc[:,2] - sinabc[:,0]*sinabc[:,2]
    R[:,0,1] = sinabc[:,0]*cosabc[:,1]*cosabc[:,2] + cosabc[:,0]*sinabc[:,2]
    R[:,0,2] = -1*sinabc[:,1]*cosabc[:,2]

    R[:,1,0] = -1*cosabc[:,0]*cosabc[:,1]*sinabc[:,2] - sinabc[:,0]*cosabc[:,2]
    R[:,1,1] = -1*sinabc[:,0]*cosabc[:,1]*sinabc[:,2] + cosabc[:,0]*cosabc[:,2]
    R[:,1,2] = sinabc[:,1]*sinabc[:,2]

    R[:,2,0] = cosabc[:,0]*sinabc[:,1]
    R[:,2,1] = sinabc[:,0]*sinabc[:,1]
    R[:,2,2] = cosabc[:,1]

    return R

def quaternion2R(qq):

    R=torch.zeros((qq.shape[0],3,3), device=qq.device)
    
    criterion = nn.Softmax(dim=1)
    qq_intermediate = criterion(qq)
    qqq = torch.sqrt(qq_intermediate)
    
    R[:,0,0] = 1 - 2*(qqq[:,2]*qqq[:,2] + qqq[:,3]*qqq[:,3])
    R[:,0,1] = 2*(qqq[:,1]*qqq[:,2] - qqq[:,3]*qqq[:,0])
    R[:,0,2] = 2*(qqq[:,1]*qqq[:,3] + qqq[:,2]*qqq[:,0])
    R[:,1,0] = 2*(qqq[:,1]*qqq[:,2] + qqq[:,3]*qqq[:,0])
    R[:,1,1] = 1 - 2*(qqq[:,1]*qqq[:,1] + qqq[:,3]*qqq[:,3])
    R[:,1,2] = 2*(qqq[:,2]*qqq[:,3] - qqq[:,1]*qqq[:,0])
    R[:,2,0] = 2*(qqq[:,1]*qqq[:,3] - qqq[:,2]*qqq[:,0])
    R[:,2,1] = 2*(qqq[:,2]*qqq[:,3] + qqq[:,1]*qqq[:,0])
    R[:,2,2] = 1 - 2*(qqq[:,1]*qqq[:,1] + qqq[:,2]*qqq[:,2])

    return R

def getRbeta():
    #compute the 60 rotation matrices in the coordinate system of Yibin Zheng and Peter C. Doerschuk, Computers in Physics, vol. 9, no. 4, July/August 1995.
    S=np.array([[np.cos(2*np.pi/5), -np.sin(2*np.pi/5), 0], [np.sin(2*np.pi/5), np.cos(2*np.pi/5), 0], [0, 0, 1]])
    U=np.array([[1/np.sqrt(5), 0, 2/np.sqrt(5)], [0, 1, 0], [-2/np.sqrt(5), 0, 1/np.sqrt(5)]])
    P=np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    T=np.dot(U,np.dot(S,np.linalg.inv(U)))

    Rbeta=np.zeros((60,3,3))

    Rbeta[0,:,:]=np.eye(3)
    Rbeta[1,:,:]=S
    Rbeta[2,:,:]=np.dot(S,S) #S^2
    Rbeta[3,:,:]=np.dot(Rbeta[2,:,:],S) #S^3
    Rbeta[4,:,:]=np.dot(Rbeta[3,:,:],S) #S^4
    Rbeta[5,:,:]=np.dot(S,T)
    Rbeta[6,:,:]=np.dot(T,Rbeta[5,:,:])
    Rbeta[7,:,:]=np.dot(T,Rbeta[6,:,:])
    Rbeta[8,:,:]=np.dot(np.linalg.inv(T),Rbeta[5,:,:])
    Rbeta[9,:,:]=np.dot(np.linalg.inv(T),Rbeta[8,:,:])
    for ii in range(5,15): #5,6,...,14
        Rbeta[ii+5,:,:]=np.dot(S,Rbeta[ii,:,:])
    Sinv=np.linalg.inv(S)
    for ii in range(5,10): #5,6,...,9
        Rbeta[ii+15,:,:]=np.dot(Sinv,Rbeta[ii,:,:])
    for ii in range(20,25): #20,21,...,24
        Rbeta[ii+5,:,:]=np.dot(Sinv,Rbeta[ii,:,:])
    for ii in range(0,30): #0,1,...,29
        Rbeta[ii+30,:,:]=np.dot(P,Rbeta[ii,:,:])

    return Rbeta

def getlossrotationsymmetry(testBool, Rest,Rtarget,Rbeta):
    
    Nbatch = Rest.shape[0]
    Ng = Rbeta.shape[0]
    
    tripleproduct=torch.zeros((Nbatch,Ng,3,3),device=Rest.device)
    RtargetT=torch.transpose(Rtarget,1,2) #leave dim 0 unaltered and transpose dims 1 and 2
    prod1 = torch.matmul(torch.unsqueeze(Rbeta,0), torch.unsqueeze(RtargetT,1))
    tripleproduct = torch.matmul(torch.unsqueeze(Rest,1), prod1) 

    '''
    for batch in range(0,Nbatch):
        tripleproduct[batch,:,:,:] = torch.matmul(RtargetT[batch,:,:], torch.matmul(Rbeta, Rest[batch,:,:]))
        
        for beta in range(0,Ng):
            tripleproduct[batch,beta,:,:] = torch.chain_matmul( Rest[batch,:,:], Rbeta[beta,:,:], RtargetT[batch,:,:] )
        
    '''
    tmp = torch.add(tripleproduct,-1*torch.eye(3, device=Rest.device)) #subtract the identity matrix

    (values, indices) = torch.min( torch.norm(tmp,p='fro',dim=(2, 3)), 1) #compute 3x3 matrix norms and select the smallest value
    
    loss = torch.sum(values)/Rest.shape[0] #average over batchsize to avg. batch loss

    gt = torch.zeros((3,3,3))
    pred = torch.zeros((3,3,3))

    if testBool:
        rand_nums = [randint(0,32),randint(0,32),randint(0,32)]
        for i in range(len(rand_nums)):
            gt[i,:,:] = prod1[rand_nums[i],indices[rand_nums[i]],:,:]
            pred[i,:,:] = Rest[rand_nums[i],:,:]
   
    return gt, pred, loss
    
def getlossrotation(testBool, R_est, R_target):
    
    R_targetT = torch.transpose(R_target,1,2) #leave dim 0 unaltered and transpose dims 1 and 2

    tmp = torch.add(torch.bmm(R_est,R_targetT),-1*torch.eye(3, device=R_est.device))# , requires_grad=True
    
    #First compute norms of the 3x3 matrices occupying the 1 and 2 indices and then sum over batchsize
    loss = torch.sum(torch.norm(tmp,p='fro',dim=(1, 2)))/R_est.shape[0]

    gt = torch.zeros((3,1,3,3))
    pred = torch.zeros((3,1,3,3))

    if testBool:
        rand_nums = [randint(0,32),randint(0,32),randint(0,32)]
        for i in range(len(rand_nums)):
            gt[i,:,:] = R_target[i,:,:]
            pred[i,:,:] = R_est[i,:,:]

    return gt, pred, loss

def getlossspacescale(Sest,Starget):
    #Sest and Starget: batchsize x 1 where "1" are the space scalings

    criterion = nn.MSELoss()
    return criterion(Sest, Starget)

def getlossgain(Gest,Gtarget):
    #Gest and Gtarget: batchsize x 1 where "1" are the gains

    criterion = nn.MSELoss()
    return criterion(Gest, Gtarget)

def getlosstotal(output,target,EulerN,QuaternionN,UseScaleSpaceAndGain,UseQuaternionNotEuler,UseSymmetryInvariantLoss,testBool,Rbeta=None):
    gt = 0
    pred = 0
    if UseScaleSpaceAndGain:
        if UseQuaternionNotEuler:
            Rest=quaternion2R(output[:,0:QuaternionN]) #quaternion
            Rtarget=euler2R(target[:,0:EulerN]) #even for quaternion calculations, the target is given in Euler angles because that is how hetero works
            if UseSymmetryInvariantLoss:
                gt, pred, sym_loss = getlossrotationsymmetry(testBool, Rest,Rtarget,Rbeta) 
                loss = sym_loss + getlossspacescale(output[:,QuaternionN],target[:,EulerN]) + getlossgain(output[:,QuaternionN+1],target[:,EulerN+1])
            else:
                gt, pred, sym_loss = getlossrotation(testBool, Rest,Rtarget)
                loss = sym_loss + getlossspacescale(output[:,QuaternionN],target[:,EulerN]) + getlossgain(output[:,QuaternionN+1],target[:,EulerN+1])
        else:
            Rest=euler2R(output[:,0:EulerN]) #Euler angles
            Rtarget=euler2R(target[:,0:EulerN]) #Euler angles
            if UseSymmetryInvariantLoss:
                gt, pred, sym_loss = getlossrotationsymmetry(testBool, Rest,Rtarget,Rbeta) 
                loss = sym_loss + getlossspacescale(output[:,EulerN],target[:,EulerN]) + getlossgain(output[:,EulerN+1],target[:,EulerN+1])
            else:
                gt, pred, sym_loss = getlossrotation(testBool, Rest,Rtarget)
                loss = sym_loss + getlossspacescale(output[:,EulerN],target[:,EulerN]) + getlossgain(output[:,EulerN+1],target[:,EulerN+1])
    else:
        if UseQuaternionNotEuler:
            Rest=quaternion2R(output) #quaternion
            Rtarget=euler2R(target) #even for quaternion calculations, the target is given in Euler angles because that is how hetero works
        else:
            Rest=euler2R(output) #Euler angles
            Rtarget=euler2R(target) #Euler angles
        
        if UseSymmetryInvariantLoss:
            gt, pred, loss = getlossrotationsymmetry(testBool, Rest,Rtarget,Rbeta)
        else:
            gt, pred, loss = getlossrotation(testBool, Rest,Rtarget)

    return gt, pred, loss

def getlossPhysicsPixel_precalc(args, device=None):

    verts = loadmat('dodecahedron.vertex.mat')['verts']
    Neta = 1
    Npulse = verts.shape[0]
    covar_value = args.covar_value #default is from inst_fw_2D_GaussianPulse.m
    delta_chi = torch.tensor((args.deltachia, args.deltachib), device=device) #default is from inst_fw_2D_GaussianPulse.m

    means4eachclass = torch.zeros((Neta, Npulse, 3), device=device)
    means4eachclass[0,:,:] = torch.tensor(verts, device=device)
    
    covar4eachclass = torch.zeros((Neta, Npulse, 3, 3), device=device)
    for npulse in range(0,Npulse):
        covar4eachclass[0,npulse,:,:] = covar_value * torch.eye(3, device=device)
    
    UseConstantPeak4eachclass = torch.full((Neta, 1), args.UseConstantPeak, device=device)

    return (delta_chi, means4eachclass, covar4eachclass, UseConstantPeak4eachclass)

def getlossPhysicsPixel(deltachi, means4eachclass, covar4eachclass, UseConstantPeak4eachclass, data, output, EulerN, QuaternionN, UseQuaternionNotEuler):
    #deltachi, means4eachclass, covar4eachclass, UseConstantPeak4eachclass come from getlossPhysicsPixel_precalc
    #data[0:Nbatch,0,0:Na,0:Nb] is the forward model input batch of images
    #output[0:Nbatch,0:EulerN+1] or output[0:Nbatch,0:QuaternionN+1] is the forward model output
    #EulerN, QuaternionN are global constants describing indexing
    #UseQuaternionNotEuler comes from args

    Nbatch=data.shape[0]
    Na=data.shape[2]
    Nb=data.shape[3]
    etas=torch.zeros((Nbatch,1),dtype=torch.long,device=data.device) #everything is class 0

    #Seek to estimate rotation (Euler angles or quaternion), gain, and xyzscale.
    #So they should be torch.tensors with requires_grad=True.
    #Everything else should be torch.tensors with requires_grad=False which is the default.

    if UseQuaternionNotEuler:
        Rests=quaternion2R(output[:,0:QuaternionN])
        gains=output[:,QuaternionN]
        xyzscales=output[:,QuaternionN+1]
    else:
        Rests=euler2R(output[:,0:EulerN])
        gains=output[:,EulerN]
        xyzscales=output[:,EulerN+1]

    img = fw_GaussianPulses(etas,means4eachclass,covar4eachclass,UseConstantPeak4eachclass,Rests,gains,xyzscales,deltachi,Na,Nb)

    return F.l1_loss(img, data)

def fw_GaussianPulses(etas,means4eachclass,covar4eachclass,UseConstantPeak4eachclass,Rests,gains,xyzscales,deltachi,Na,Nb):

    #everything is for a batch of Nbatch images.
    #different images are from one of Neta different classes of Gaussian pulse models.
    #etas[0:Nbatch]: class indices
    #means4eachclass[0:Neta,0:Npulse,0:3]
    #covar4eachclass[0:Neta,0:Npulse,0:3,0:3]
    #UseConstantPeak4eachclass[0:Neta]
    #Rests[0:Nbatch,0:3,0:3]
    #Might also include translations.
    #gains[0:Nbatch]
    #xyzscales[0:Nbatch]
    #deltachi[0:2]: float sampling intervals in the two directions
    #Na,Nb: integer dimensions in pixels of the image
    #Would like Npulse to change as a function of eta in 0:Neta.  Did that in Matlab via cell arrays.  Don't know how to do that in Python.
    #Don't do tilt series.

    Neta=means4eachclass.shape[0]
    Nbatch=etas.shape[0]

    #2-D real-space images.
    img=torch.zeros((Nbatch,Na,Nb),device=Rests.device)

    #Prepare for evaluating the 2-D Gaussian for the real-space image.
    xa=torch.arange(-1*np.floor(Na/2),-1*np.floor(Na/2)+Na,device=Rests.device)*deltachi[0]
    xb=torch.arange(-1*np.floor(Nb/2),-1*np.floor(Nb/2)+Nb,device=Rests.device)*deltachi[1]
    Xb, Xa = torch.meshgrid(xa,xb) #Matlab: ndgrid, Python: meshgrid, note reversal of Xb and Xa
    #Xa and Xb match same variables in Matlab
    vecXa=torch.reshape(Xa.T,(1,Na*Nb)) #C ordering, column changes fastest
    vecXb=torch.reshape(Xb.T,(1,Na*Nb))

    allpixels = torch.cat((vecXa, vecXb))#should be 2 x Na*Nb, "torch.as_tensor" avoids the copy that occurs with "torch.tensor".
    #allpixels matches same variable in Matlab    

    for batch in range(0,Nbatch):
        eta=etas[batch]
        Rabc=torch.squeeze(Rests[batch,:,:])
        gain=gains[batch]
        xyzscale=xyzscales[batch]
        Npulse=means4eachclass.shape[1] #might eventually depend on eta
        for npulse in range(0,Npulse):

            #Compute the rotated Gaussian
            rmean=torch.mv(Rabc,torch.squeeze(means4eachclass[eta,npulse,:])) #in math, means4eachclass is a column vector
            rcovar=torch.chain_matmul(Rabc, torch.squeeze(covar4eachclass[eta,npulse,:,:]), torch.t(Rabc))

            #Project in the z direction = get marginal on x-y = extract subblock and perform space scaling
            mean2=xyzscale*rmean[0:2]
            covar2=(xyzscale**2)*rcovar[0:2,0:2]

            #Evaluate this Gaussian at all pixels
            if UseConstantPeak4eachclass[eta]:
                normalizer=1.0
            else:
                normalizer=1/(2*np.pi*torch.sqrt(torch.det(covar2)))

            tmp=torch.t(torch.t(allpixels) - mean2)

            #reversal of Na,Nb and transpose following reshape makes img match Matlab variable
            img[batch,:,:]=img[batch,:,:]+torch.transpose(torch.reshape(normalizer*torch.exp(-0.5*torch.sum(tmp * torch.mm(torch.inverse(covar2), tmp), dim=0)), (Nb,Na)), 0,1)

        img[batch,:,:]=gain*img[batch,:,:]

    return img

    

