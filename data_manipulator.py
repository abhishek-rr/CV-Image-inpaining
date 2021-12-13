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

# Method to create the mask matrix to crop the images however we want
class DataManipulator:
    def __init__(self, device, image_size=(96,96), crop_center=(48,48), crop_size=(48,48)):
        self.device = device
        self.image_size = image_size
        self.crop_center = crop_center
        self.crop_size = crop_size

        # Create masks to crop the images and calculate the loss

        self.crop_mask = torch.ones(image_size).to(self.device)
        self.loss_mask = torch.zeros(image_size).to(self.device)

        self.x1 = self.crop_center[0]-int(self.crop_size[0]/2)
        self.x2 = self.crop_center[0]+int(self.crop_size[0]/2)

        self.y1 = self.crop_center[1]-int(self.crop_size[1]/2)
        self.y2 = self.crop_center[1]+int(self.crop_size[1]/2)
        self.crop_mask[self.x1:self.x2, self.y1:self.y2] = 0.
        self.loss_mask[self.x1:self.x2, self.y1:self.y2] = 1.

        self.crop_mask = self.crop_mask.expand(3,-1,-1)
        self.loss_mask = self.loss_mask.expand(3,-1,-1)

    def mask_image(self, image):

        # Add noise to the cropped part
        for i in range(self.x1,self.x2):
            for j in range(self.y1,self.y2):
                self.loss_mask[:,i,j] = torch.ones([1])[0]
        
        return torch.mul(image, self.crop_mask).add(self.loss_mask)
    
    def crop_image(self, image):
        # Crops the image according to the loss_mask
        return image[:, :, self.x1:self.x2, self.y1:self.y2]
    
    # Blurs the middle part of the image
    def blur_mid_image(self, image, sigma):
        # TODO: try pytorch transforms for blurring
        cropped_image = self.crop_image(image)
        blurred = np.zeros(image.shape)
        blurred[:,:, self.x1:self.x2, self.y1:self.y2] = gaussian_filter(cropped_image, sigma=sigma)
        blurred = torch.FloatTensor(blurred)
        return torch.mul(image, self.crop_mask).add(blurred)
