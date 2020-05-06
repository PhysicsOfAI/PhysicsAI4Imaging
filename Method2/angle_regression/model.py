import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EulerGainVGG(nn.Module):

    def __init__(self, args, init_weights=True):
        super(EulerGainVGG, self).__init__()

        self.UseQuaternion = args.UseQuaternionNotEuler
        self.shift4spacescaling=(args.ScaleSpaceMax+args.ScaleSpaceMin)/(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.scaling4spacescaling=0.5*(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.shift4gain=(args.GainMax+args.GainMin)/(args.GainMax-args.GainMin)
        self.scaling4gain=0.5*(args.GainMax-args.GainMin)

        self.EulerN=3
        self.QuaternionN=4
        self.ScaleSpaceAndGainN=2

        self.features = self.__make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        classifier = [
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True)
            ]
        num_out = self.ScaleSpaceAndGainN + (self.QuaternionN*int(args.UseQuaternionNotEuler) + self.EulerN*int(not args.UseQuaternionNotEuler))
        classifier.append(nn.Linear(4096,num_out))
        self.classifier = nn.Sequential(*classifier)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        if self.UseQuaternion:
                x_new = x.clone()
                x_new[:,self.QuaternionN:] = (F.tanh(x[:,self.QuaternionN:]) + torch.cuda.FloatTensor((self.shift4spacescaling, self.shift4gain)))*torch.cuda.FloatTensor((self.scaling4spacescaling, self.scaling4gain))
                
                return x_new # quaternion, space scaling, gain

        else:
            x_new = x.clone()
            x_new = (F.tanh(x) + torch.cuda.FloatTensor((1.0, 1.0, 1.0, self.shift4spacescaling, self.shift4gain))) * torch.cuda.FloatTensor((np.pi, np.pi/2.0, np.pi, self.scaling4spacescaling, self.scaling4gain))
            
            return x_new 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    
    def __make_layers(self, batch_norm=False):
        layers = []
        in_channels = 1
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
                else:
                    layers += [conv2d, nn.ReLU(True)]
                in_channels = v
        return nn.Sequential(*layers)

class EulerGainMLP(nn.Module):

    def __init__(self, args, init_weights=True):
        super(EulerGainMLP, self).__init__()

        self.UseQuaternion = args.UseQuaternionNotEuler
        self.shift4spacescaling=(args.ScaleSpaceMax+args.ScaleSpaceMin)/(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.scaling4spacescaling=0.5*(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.shift4gain=(args.GainMax+args.GainMin)/(args.GainMax-args.GainMin)
        self.scaling4gain=0.5*(args.GainMax-args.GainMin)
        

        self.EulerN=3
        self.QuaternionN=4
        self.ScaleSpaceAndGainN=2

        classifier = [nn.Linear(113*113, 4096), 
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(), 
        nn.Linear(4096, 1024),
        nn.ReLU(), 
        nn.Linear(1024, 512),
        nn.ReLU()]
        num_out = self.ScaleSpaceAndGainN + (self.QuaternionN*int(args.UseQuaternionNotEuler) + self.EulerN*int(not args.UseQuaternionNotEuler))
        classifier.append(nn.Linear(512,num_out)) 
        
        
        self.classifier = nn.Sequential(*classifier)
        if init_weights:
            self._initialize_weights()
        

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        if self.UseQuaternion:
            x_new = x.clone()
            x_new[:,self.QuaternionN:] = (F.tanh(x[:,self.QuaternionN:]) + torch.cuda.FloatTensor((self.shift4spacescaling, self.shift4gain))) * torch.cuda.FloatTensor((self.scaling4spacescaling, self.scaling4gain))
            
            return x_new # quaternion, space scaling, gain

        else:
            x_new = x.clone()
            x_new = (F.tanh(x) + torch.cuda.FloatTensor((1.0, 1.0, 1.0, self.shift4spacescaling, self.shift4gain))) * torch.cuda.FloatTensor((np.pi, np.pi/2.0, np.pi, self.scaling4spacescaling, self.scaling4gain))
            
            return x_new 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0, 0.01) # variance was 0.01 before
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class QuaternionGainMLP(nn.Module):

    def __init__(self, args, init_weights=True):
        super(QuaternionGainMLP, self).__init__()

        self.UseQuaternion = args.UseQuaternionNotEuler
        self.shift4spacescaling=(args.ScaleSpaceMax+args.ScaleSpaceMin)/(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.scaling4spacescaling=0.5*(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.shift4gain=(args.GainMax+args.GainMin)/(args.GainMax-args.GainMin)
        self.scaling4gain=0.5*(args.GainMax-args.GainMin)
        

        self.EulerN=3
        self.QuaternionN=4
        self.ScaleSpaceAndGainN=2
        
        classifier = [nn.Linear(113*113, 4096), 
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(), 
        nn.Linear(4096, 1024),
        nn.ReLU(), 
        nn.Linear(1024, 512),
        nn.ReLU()]
        num_out = self.ScaleSpaceAndGainN + (self.QuaternionN*int(args.UseQuaternionNotEuler) + self.EulerN*int(not args.UseQuaternionNotEuler))
        classifier.append(nn.Linear(512,num_out)) 

        self.classifier = nn.Sequential(*classifier)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        if self.UseQuaternion:
            x_new = x.clone()
            x_new[:,self.QuaternionN:] = (F.tanh(x[:,self.QuaternionN:]) + torch.cuda.FloatTensor((self.shift4spacescaling, self.shift4gain))) * torch.cuda.FloatTensor((self.scaling4spacescaling, self.scaling4gain))
            
            return x_new # quaternion, space scaling, gain
            
        else:
            x_new = x.clone()
            x_new = (F.tanh(x) + torch.cuda.FloatTensor((1.0, 1.0, 1.0, self.shift4spacescaling, self.shift4gain))) * torch.cuda.FloatTensor((np.pi, np.pi/2.0, np.pi, self.scaling4spacescaling, self.scaling4gain))
            
            return x_new 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)