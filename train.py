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

from data_manipulator import DataManipulator
from generator import Generator
from discriminator import Discriminator

NC = 3 # Number of channels
NEF = 64 # Number of encoder filters in the first conv layer - TODO: maybe change? 
NDF = 64 # Size of the decoder filters
NDIF = 64 # Size of the discriminator filters
BTLS = 4000 # Bottleneck size - Size of the flattened feature map in the middle of encoders and decoders TODO: Make sure this is okay
NGPU = 2 # NOTE: Take a look at this, not sure if this is okay or not

# Implement the Training class for the GAN
class GANTrainer:
    def __init__(self, device, dataset_object, epochs, batch_size, gen_learning_rate, disc_learning_rate, file_name=None):
        '''
        Constructor for the training class. It initializes Discriminator and
        Generator and starts training both, according to the given data
        Args:
        - discriminator_epochs: Number k mentioned in Algorithm 1 of the official GAN paper.
        Steps to train Discriminator in each training iteration
        - plot_graphs: Parameter to show if we'd like to plot the outputs of each NN
        - plotting_interval: If plot_graphs is True, number of training steps to plot the graphs in
        '''
        self.device = device
        
        self.file_name = file_name
        if file_name != None:
            self.gen_file = open("{}_gen.txt".format(file_name), "w+")
            self.disc_file = open("{}_disc.txt".format(file_name), "w+")
        
        self.epochs = epochs
        self.batch_size = batch_size
#         self.learning_rate = learning_rate

        self.generator = Generator(NC, NEF, BTLS, NDF)
        self.discriminator = Discriminator(NC, NDIF)

        # Set up optimizers for both discriminator and generator
        self.optimizer_gen = optim.Adam(self.generator.parameters(), lr=gen_learning_rate)
        self.optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=disc_learning_rate, weight_decay=0.001)

        # Initialize the loss functions
        self.adversarial_loss = torch.nn.MSELoss()
        self.pixelwise_loss = torch.nn.L1Loss()
        # Get the data loader and manipulator objects
        self.dataset_object = dataset_object
        self.data_loader = torch.utils.data.DataLoader(dataset=self.dataset_object,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)
        self.data_iter = iter(self.data_loader)
        self.data_man = DataManipulator(device)
        
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.adversarial_loss.to(self.device)
        self.pixelwise_loss.to(self.device)
            
    def train(self):
        
        for epoch in range(self.epochs + 1):
            for i, (x_original, _) in enumerate(self.data_loader):
                label_real = torch.full((x_original.shape[0],1), 1., dtype=torch.float, requires_grad=False).cuda()
                label_fake = torch.full((x_original.shape[0],1), 0., dtype=torch.float, requires_grad=False).cuda()
                
                x_original = x_original.to(self.device)

                x_cropped = self.data_man.crop_image(x_original) #small cropped square in the middle that is the ground truth
                x_masked = self.data_man.mask_image(x_original) #entire image with black box in the middle

                #Train Discriminator

                self.optimizer_disc.zero_grad()
                self.optimizer_gen.zero_grad()
                generated_img = self.generator(x_masked)

                real_loss = 0.5 * self.adversarial_loss(self.discriminator(x_cropped), label_real)
                real_loss.backward()
                
                fake_loss = 0.5 * self.adversarial_loss(self.discriminator(generated_img.detach()), label_fake)
                fake_loss.backward()
                
                d_loss = real_loss + fake_loss
#                 d_loss.backward()
                self.optimizer_disc.step()
                
                #Train Generator

#                 self.optimizer_gen.zero_grad()

#                 generated_img = self.generator(x_masked)

                # Adversarial and pixelwise loss
                g_adv = self.adversarial_loss(self.discriminator(generated_img), label_real)
                g_pixel = self.pixelwise_loss(generated_img, x_cropped)
                # Total loss
                g_loss = 0.002 * g_adv + 0.998 * g_pixel

                g_loss.backward()
                self.optimizer_gen.step()    
                
                if self.file_name != None: # Save the losses in a file
#                     print('g_loss.item(): {}'.format(g_loss.item()))
#                     print('g_loss: {}, d_loss: {}'.format(g_loss.item(), d_loss.item()))
                    self.gen_file.write('{}\n'.format(g_loss.item()))
                    self.disc_file.write('{}\n'.format(d_loss.item()))
        
                    self.gen_file.flush()
                    self.disc_file.flush()
                
            
            print('| {} | GEN_LOSS: {} | DISC_LOSS: {} | Real_LOSS: {} | Fake_LOSS: {} |'.format(
                epoch, g_loss, d_loss, real_loss, fake_loss
            ))
            if epoch % 10 == 0: # Print the losses   
                torch.save(self.generator.state_dict(), 'model_weights/model_gen_{}.pth'.format(epoch))