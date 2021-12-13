import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os

# This class is only for debugging purposes - it is used to print the shape of the layer and added to the sequential model
class PrintLayer(nn.Module): 
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x

class Generator(nn.Module):
    def __init__(self, NC, NEF, BTLS, NDF):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 96 x 96
            nn.Conv2d(NC,NEF,4,2,1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        
            # state size: (nef) x 48 x 48
            nn.Conv2d(NEF,NEF,4,2,1, bias=False),
            nn.BatchNorm2d(NEF),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (nef) x 24 x 24
            nn.Conv2d(NEF,NEF*2,4,2,1, bias=False),
            nn.BatchNorm2d(NEF*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (nef*2) x 12 x 12
            nn.Conv2d(NEF*2,NEF*4,4,2,1, bias=False),
            nn.BatchNorm2d(NEF*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (nef*4) x 6 x 6
            nn.Conv2d(NEF*4,NEF*8,4,2,1, bias=False),
            nn.BatchNorm2d(NEF*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (nef*8) x 3 x 3
            nn.Conv2d(NEF*8, BTLS, 3, bias=False), # NOTE: The kernel size was 4 in the original paper since their input size was 128x128
            
            # state size: (nBottleneck) x 1 x 1
            nn.BatchNorm2d(BTLS),
            nn.LeakyReLU(0.2, inplace=True),
            
            # input is Bottleneck, going into a convolution
            nn.ConvTranspose2d(BTLS, NDF * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.ReLU(True),
            
            # state size. (ngf*8) x 3 x 3
            nn.ConvTranspose2d(NDF * 8, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.ReLU(True),
            
            # state size. (ngf*4) x 6 x 6
            nn.ConvTranspose2d(NDF * 4, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.ReLU(True),
            
            # state size. (ngf*2) x 12 x 12
            nn.ConvTranspose2d(NDF * 2, NDF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF),
            nn.ReLU(True),
            
            # state size. (ngf) x 24 x 24
            nn.ConvTranspose2d(NDF, NC, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 48 x 48
            
            # TODO: Right now the assumption is that the crop_size is 48x48
            # This could be modifiable for the next versions of the code
        )
        
    def forward(self, input):
        output = self.main(input)
        return output

