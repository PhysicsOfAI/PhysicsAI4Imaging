import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EulerGainConvVAE(nn.Module):

    def __init__(self, args, init_weights=True):
        super(EulerGainConvVAE, self).__init__()

        self.UseQuaternion = args.UseQuaternionNotEuler
        self.shift4spacescaling=(args.ScaleSpaceMax+args.ScaleSpaceMin)/(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.scaling4spacescaling=0.5*(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.shift4gain=(args.GainMax+args.GainMin)/(args.GainMax-args.GainMin)
        self.scaling4gain=0.5*(args.GainMax-args.GainMin)

        self.EulerN=3
        self.QuaternionN=4
        self.ScaleSpaceAndGainN=2
        
        self.ch_factor_1out6 = 128
        self.ch_factor_2out6 = 128
        self.ch_factor_3out6 = 128
        self.ch_factor_4out6 = 256
        self.ch_factor_5out6 = 256
        self.ch_factor_6out6 = 256
        num_channels = 1
        
        encoder = [nn.Conv2d(num_channels, num_channels*self.ch_factor_1out6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_1out6),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels*self.ch_factor_1out6, num_channels*self.ch_factor_2out6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_2out6),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels*self.ch_factor_2out6, num_channels*self.ch_factor_3out6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_3out6),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels*self.ch_factor_3out6, num_channels*self.ch_factor_4out6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_4out6),
            nn.ReLU(inplace=True)] 
        encoder += self.ResBlock(num_channels*self.ch_factor_4out6, num_channels*self.ch_factor_5out6, bn=True)
        encoder.append(nn.BatchNorm2d(num_channels*self.ch_factor_5out6)) 
        encoder += self.ResBlock(num_channels*self.ch_factor_5out6, num_channels*self.ch_factor_6out6, bn=True)
        self.encoder = nn.Sequential(*encoder)

        num_out = self.ScaleSpaceAndGainN + (self.QuaternionN*int(args.UseQuaternionNotEuler) + self.EulerN*int(not args.UseQuaternionNotEuler)) 

        self.mu = nn.Linear(num_channels*self.ch_factor_6out6 * 7 * 7, 128)
        self.logvar = nn.Linear(num_channels*self.ch_factor_6out6 * 7 * 7, 128)
        self.angle_params = nn.Linear(num_channels*self.ch_factor_6out6 * 7 * 7, num_out)

        ## for decoder
        self.dec_in = nn.Linear(128+num_out, num_channels*self.ch_factor_6out6 * 7 * 7)

        decoder = self.ResBlock(num_channels*self.ch_factor_6out6, num_channels*self.ch_factor_5out6, bn=True)
        decoder.append(nn.BatchNorm2d(num_channels*self.ch_factor_5out6))
        decoder += self.ResBlock(num_channels*self.ch_factor_5out6, num_channels*self.ch_factor_4out6, bn=True)
        decoder.append(nn.BatchNorm2d(num_channels*self.ch_factor_4out6))
        decoder += [nn.ConvTranspose2d(num_channels*self.ch_factor_4out6, num_channels*self.ch_factor_3out6, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_3out6),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_channels*self.ch_factor_3out6, num_channels*self.ch_factor_2out6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_2out6),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_channels*self.ch_factor_2out6 , num_channels*self.ch_factor_1out6, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_1out6),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_channels*self.ch_factor_1out6, num_channels, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False)]

        self.decoder = nn.Sequential(*decoder)

        if init_weights:
            self._initialize_weights()


    def forward(self, x_first):
        x = self.encoder(x_first)
        x = torch.flatten(x, 1)
        
        angle_params = self.angle_params(x)
        mu = self.mu(x)
        logvar = self.logvar(x)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        reparametrized = mu + eps*std
        
        if self.UseQuaternion:
            #modify only the Space Scaling and the Gain columns in x
            x_new = angle_params.clone()
            x_new[:,self.QuaternionN:] = (F.tanh(angle_params[:,self.QuaternionN:]) + torch.cuda.FloatTensor((self.shift4spacescaling, self.shift4gain))) * torch.cuda.FloatTensor((self.scaling4spacescaling, self.scaling4gain))
            
            feat_vec = torch.cat([x_new, reparametrized],dim=1)
            
        else:
            x_new = angle_params.clone()
            x_new = (F.tanh(angle_params) + torch.cuda.FloatTensor((1.0, 1.0, 1.0, self.shift4spacescaling, self.shift4gain))) * torch.cuda.FloatTensor((np.pi, np.pi/2.0, np.pi, self.scaling4spacescaling, self.scaling4gain))
            feat_vec = torch.cat([x_new, reparametrized],dim=1)

        lin_out = self.dec_in(feat_vec)
        lin_out = torch.reshape(lin_out, (-1,self.ch_factor_6out6,7,7))
        dec_out = self.decoder(lin_out)

        return mu, logvar, x_new, x_first, dec_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
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

    def ResBlock(self, in_channels, out_channels, mid_channels=None, bn=False):
        if mid_channels is None:
            mid_channels = out_channels
        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))

        return layers 

'''
STILL IN DEVELOPMENT

class EulerGainVAE(nn.Module):

    def __init__(self, args, init_weights=True):
        super(EulerGainVAE, self).__init__()

        self.UseQuaternion = args.UseQuaternionNotEuler
        self.shift4spacescaling=(args.ScaleSpaceMax+args.ScaleSpaceMin)/(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.scaling4spacescaling=0.5*(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.shift4gain=(args.GainMax+args.GainMin)/(args.GainMax-args.GainMin)
        self.scaling4gain=0.5*(args.GainMax-args.GainMin)

        self.EulerN=3
        self.QuaternionN=4
        self.ScaleSpaceAndGainN=2
        
        encoder = [nn.Linear(91*91, 4096), 
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(), 
        nn.Linear(4096, 1024),
        nn.ReLU(), 
        nn.Linear(1024, 512),
        nn.ReLU()]
        self.encoder = nn.Sequential(*encoder)

        num_out = self.ScaleSpaceAndGainN + (self.QuaternionN*int(args.UseQuaternionNotEuler) + self.EulerN*int(not args.UseQuaternionNotEuler)) 

        self.angle_params = nn.Linear(512, num_out)
        self.mu = nn.Linear(512, 128)
        self.logvar = nn.Linear(512, 128)

        decoder = [nn.Linear(num_out+128, 512), 
        nn.ReLU(),
        nn.Linear(512,1024), 
        nn.ReLU(),
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 91*91)]
        self.decoder = nn.Sequential(*decoder)
        if init_weights:
            self._initialize_weights()

    def forward(self, x_first):
        x_first = torch.flatten(x_first, 1)
        x = self.encoder(x_first)

        angle_params = self.angle_params(x)
        mu = self.mu(x)
        logvar = self.logvar(x)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        reparametrized = mu + eps*std
        
        if self.UseQuaternion:
            #modify only the Space Scaling and the Gain columns in x
            x_new = angle_params.clone()
            x_new[:,self.QuaternionN:] = (F.tanh(angle_params[:,self.QuaternionN:]) + torch.cuda.FloatTensor((self.shift4spacescaling, self.shift4gain))) * torch.cuda.FloatTensor((self.scaling4spacescaling, self.scaling4gain))
            
            feat_vec = torch.cat([x_new, reparametrized],dim=1)

        else:

            x_new = angle_params.clone()
            x_new = (F.tanh(angle_params) + torch.cuda.FloatTensor((1.0, 1.0, 1.0, self.shift4spacescaling, self.shift4gain))) * torch.cuda.FloatTensor((np.pi, np.pi/2.0, np.pi, self.scaling4spacescaling, self.scaling4gain))
            feat_vec = torch.cat([x_new, reparametrized],dim=1)

        dec_out = self.decoder(feat_vec)

        return mu, logvar, x_new, x_first, dec_out

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
    
class EulerGainConvVAE2(nn.Module):

    def __init__(self, args, init_weights=True):
        super(EulerGainConvVAE2, self).__init__()

        self.UseQuaternion = args.UseQuaternionNotEuler
        self.shift4spacescaling=(args.ScaleSpaceMax+args.ScaleSpaceMin)/(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.scaling4spacescaling=0.5*(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.shift4gain=(args.GainMax+args.GainMin)/(args.GainMax-args.GainMin)
        self.scaling4gain=0.5*(args.GainMax-args.GainMin)

        self.EulerN=3
        self.QuaternionN=4
        self.ScaleSpaceAndGainN=2
        
        self.ch_factor_1out6 = 128
        self.ch_factor_2out6 = 128
        self.ch_factor_3out6 = 128
        self.ch_factor_4out6 = 256
        self.ch_factor_5out6 = 256
        self.ch_factor_6out6 = 256
        self.ch_factor_7out6 = 256
        num_channels = 1
        
        encoder = [nn.Conv2d(num_channels, num_channels*self.ch_factor_1out6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_1out6),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels*self.ch_factor_1out6, num_channels*self.ch_factor_2out6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_2out6),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels*self.ch_factor_2out6, num_channels*self.ch_factor_3out6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_3out6),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels*self.ch_factor_3out6, num_channels*self.ch_factor_4out6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_4out6),
            nn.ReLU(inplace=True),
            ResBlock(num_channels*self.ch_factor_4out6, num_channels*self.ch_factor_5out6, bn=True),
            nn.BatchNorm2d(num_channels*self.ch_factor_5out6),
            ResBlock(num_channels*self.ch_factor_5out6, num_channels*self.ch_factor_6out6, bn=True),
            nn.BatchNorm2d(num_channels*self.ch_factor_6out6),
            ResBlock(num_channels*self.ch_factor_6out6, num_channels*self.ch_factor_7out6, bn=True)]

        self.encoder = nn.Sequential(*encoder)

        num_out = self.ScaleSpaceAndGainN + (self.QuaternionN*int(args.UseQuaternionNotEuler) + self.EulerN*int(not args.UseQuaternionNotEuler)) 

        self.mu = nn.Linear(num_channels*self.ch_factor_6out6 * 7*7, 128)
        self.logvar = nn.Linear(num_channels*self.ch_factor_6out6 * 7*7, 128)
        self.angle_params = nn.Linear(num_channels*self.ch_factor_6out6 * 7*7, num_out)

        ## for decoder
        self.dec_in = nn.Linear(128+num_out, num_channels*self.ch_factor_6out6 * 7*7)

        decoder = [ResBlock(num_channels*self.ch_factor_7out6, num_channels*self.ch_factor_6out6, bn=True),
            nn.BatchNorm2d(num_channels*self.ch_factor_6out6),
            ResBlock(num_channels*self.ch_factor_6out6, num_channels*self.ch_factor_5out6, bn=True),
            nn.BatchNorm2d(num_channels*self.ch_factor_5out6),
            ResBlock(num_channels*self.ch_factor_5out6, num_channels*self.ch_factor_4out6, bn=True),
            nn.BatchNorm2d(num_channels*self.ch_factor_4out6),
            nn.ConvTranspose2d(num_channels*self.ch_factor_4out6, num_channels*self.ch_factor_3out6, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_3out6),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_channels*self.ch_factor_3out6, num_channels*self.ch_factor_2out6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_2out6),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_channels*self.ch_factor_2out6 , num_channels*self.ch_factor_1out6, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_channels*self.ch_factor_1out6),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_channels*self.ch_factor_1out6, num_channels, kernel_size=4, stride=2, padding=1, output_padding=1, bias=False)]

        self.decoder = nn.Sequential(*decoder)

        if init_weights:
            self._initialize_weights()


    def forward(self, x_first):
        x = self.encoder(x_first)
        x = torch.flatten(x, 1)
        
        angle_params = self.angle_params(x)
        mu = self.mu(x)
        logvar = self.logvar(x)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        reparametrized = mu + eps*std
        
        if self.UseQuaternion:
            #modify only the Space Scaling and the Gain columns in x
            x_new = angle_params.clone()
            x_new[:,self.QuaternionN:] = (F.tanh(angle_params[:,self.QuaternionN:]) + torch.cuda.FloatTensor((self.shift4spacescaling, self.shift4gain))) * torch.cuda.FloatTensor((self.scaling4spacescaling, self.scaling4gain))
            
            feat_vec = torch.cat([x_new, reparametrized],dim=1)
 
        else:
            
            x_new = angle_params.clone()
            x_new = (F.tanh(angle_params) + torch.cuda.FloatTensor((1.0, 1.0, 1.0, self.shift4spacescaling, self.shift4gain))) * torch.cuda.FloatTensor((np.pi, np.pi/2.0, np.pi, self.scaling4spacescaling, self.scaling4gain))
            feat_vec = torch.cat([x_new, reparametrized],dim=1)

        lin_out = self.dec_in(feat_vec)
        lin_out = torch.reshape(lin_out, (-1,self.ch_factor_6out6,5,5))
        dec_out = self.decoder(lin_out)

        return mu, logvar, x_new, x_first, dec_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0, 0.01) # variance was 0.01 before
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

class ResBlock(nn.Module):
    def __init__(self, in_channels, channels, bn=False):
        super(ResBlock, self).__init__()
        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)
'''





















































class EulerGainVGGVAE(nn.Module):

    def __init__(self, args, init_weights=True):
        super(EulerGainVGGVAE, self).__init__()

        self.UseQuaternion = args.UseQuaternionNotEuler
        self.shift4spacescaling=(args.ScaleSpaceMax+args.ScaleSpaceMin)/(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.scaling4spacescaling=0.5*(args.ScaleSpaceMax-args.ScaleSpaceMin)
        self.shift4gain=(args.GainMax+args.GainMin)/(args.GainMax-args.GainMin)
        self.scaling4gain=0.5*(args.GainMax-args.GainMin)
        self.UseScaleSpaceAndGain=args.UseScaleSpaceAndGain

        self.EulerN=3
        self.QuaternionN=4
        self.ScaleSpaceAndGainN=2
        
        encoder = [nn.Conv2d(1, 64, kernel_size=3),
        nn.ReLU(True),
        nn.Conv2d(64, 64, kernel_size=3),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(64, 128, kernel_size=3),
        nn.ReLU(True),
        nn.Conv2d(128, 128, kernel_size=3),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(128, 256, kernel_size=3),
        nn.ReLU(True),
        nn.Conv2d(256, 256, kernel_size=3),
        nn.ReLU(True),
        nn.Conv2d(256, 256, kernel_size=3),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2),
        nn.AdaptiveAvgPool2d((7, 7))
        ]
        self.encoder = nn.Sequential(*encoder)

        num_out = int(args.UseScaleSpaceAndGain)*self.ScaleSpaceAndGainN + (self.QuaternionN*int(args.UseQuaternionNotEuler) + self.EulerN*int(not args.UseQuaternionNotEuler)) 

        self.angle_params = nn.Linear(256*7*7, num_out)
        self.mu = nn.Linear(256*7*7, 128)
        self.logvar = nn.Linear(256*7*7, 128)

        linLayer = [nn.Linear(num_out+128, 256*7*7), 
        nn.ReLU(True)]
        self.linLayer = nn.Sequential(*linLayer)

        decoder = [ #7*7
        nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1), #13*13
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0), #15*15
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0), #17*17
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0), #19*19
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0), #39*39
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0), #41*41
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0), #43*43
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0), #87*87
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0), #89*89
        nn.ReLU(True),
        nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=0), #91*91
        ]
        self.decoder = nn.Sequential(*decoder)

        if init_weights:
            self._initialize_weights()

    def forward(self, x_first):
        x = self.encoder(x_first)
        x = torch.flatten(x, 1)
        
        angle_params = self.angle_params(x)
        mu = self.mu(x)
        logvar = self.logvar(x)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        reparametrized = mu + eps*std
        
        if self.UseQuaternion:
            if self.UseScaleSpaceAndGain:
                #modify only the Space Scaling and the Gain columns in x
                x_new = angle_params.clone()
                x_new[:,self.QuaternionN:] = (F.tanh(angle_params[:,self.QuaternionN:]) + torch.cuda.FloatTensor((self.shift4spacescaling, self.shift4gain))) * torch.cuda.FloatTensor((self.scaling4spacescaling, self.scaling4gain))
                
                feat_vec = torch.cat([x_new, reparametrized],dim=1)
            else:
                x_new = angle_params.clone()
                feat_vec = torch.cat([x_new, reparametrized],dim=1) 
        else:
            if self.UseScaleSpaceAndGain:
                x_new = angle_params.clone()
                x_new = (F.tanh(angle_params) + torch.cuda.FloatTensor((1.0, 1.0, 1.0, self.shift4spacescaling, self.shift4gain))) * torch.cuda.FloatTensor((np.pi, np.pi/2.0, np.pi, self.scaling4spacescaling, self.scaling4gain))
                feat_vec = torch.cat([x_new, reparametrized],dim=1)
            else:
                x_new = angle_params.clone()
                x_new = (F.tanh(angle_params) + torch.cuda.FloatTensor((1.0, 1.0, 1.0))) * torch.cuda.FloatTensor((np.pi, np.pi/2.0, np.pi))
                feat_vec = torch.cat([x_new, reparametrized],dim=1)

        lin_out = self.linLayer(feat_vec)
        lin_out = torch.reshape(lin_out, (-1,256,7,7))
        dec_out = self.decoder(lin_out)

        return mu, logvar, x_new, x_first, dec_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.ConvTranspose2d):
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