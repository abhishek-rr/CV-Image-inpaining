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
import numpy as np

# Code for the discriminator

class Discriminator(nn.Module):
    def __init__(self, NC, NDIF):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 48 x 48 - TODO: Again, this is dependent on the crop size!!
            nn.Conv2d(NC, NDIF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf) x 24 x 24
            nn.Conv2d(NDIF, NDIF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDIF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*2) x 12 x 12
            nn.Conv2d(NDIF * 2, NDIF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDIF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*4) x 6 x 6
            nn.Conv2d(NDIF * 4, NDIF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDIF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*8) x 3 x 3
            nn.Conv2d(NDIF * 8, 1, 3, 1, 0, bias=False), # NOTE: Kernel size here was 4 but we changed it to 3 since the input size is 48
            nn.Sigmoid(),
        
        )
            

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1)
