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
import sys

from train import GANTrainer

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_data(path_to_data):
    """
    :param path_to_data: path to the binary file containing data from the STL-10 dataset
    :return: an array containing the images in column-major order
    """
    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.
        images = np.reshape(everything, (-1, 3, 96, 96))
        return images

def plot_image(encoded_image):
    import matplotlib.pyplot as plt
    encoded_image_cpu = encoded_image.cpu()
    image = np.transpose(encoded_image_cpu, (2, 1, 0))
    plt.imshow(image)
    plt.show()
    

def main():
    
    file_name = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--save":
            file_name = sys.argv[2]
        else:
            print("Usage: python3.9 main.py <--save file_path>")
            return

    transform = transforms.Compose([transforms.ToTensor()])
    dataset_object = dset.STL10(root ='./root' , split='unlabeled', download = False, transform=transform)
        
    trainer = GANTrainer(
        device=device,
        file_name=file_name,
        dataset_object=dataset_object,
        epochs=100,
        batch_size=256,
        gen_learning_rate = 0.01,
        disc_learning_rate=0.001, # TODO: test different learning rates for generator and discriminator
    )

    trainer.train()
    
main()
