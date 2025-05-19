
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import cv2

from torch.utils.data import DataLoader, Dataset, ConcatDataset

import numpy as np
import matplotlib.pyplot as plt
# from stm_utils import Sxm_Image
# from CITS_Class import CITS_Analysis
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import atomai as aoi
from atomai.utils import get_coord_grid, extract_patches_and_spectra, extract_patches, extract_subimages


class im2spec_Dataset(Dataset):

    def __init__(self, images, spectra, transform = None, norm = True):
        self.images = images
        self.spectra = spectra
        self.transform = transform
#         self.norm = norm
#         self.normalize = transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize in range[-1 to 1]

    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, idx):
        
        image = torch.tensor(self.images[idx], dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        spectra = torch.tensor(self.spectra[idx], dtype=torch.float32)
        
#         if self.norm:
#             image = self.normalize(image)
#             spectra = self.normalize(spectra)

        return image, spectra


def augmented_dataset(images, spectra):
    
    dataset1 = im2spec_Dataset(images, spectra)
    
    
    # Define the transform1
    transform1 = transforms.Compose([
        AddGaussianNoise(mean=0., std=0.1),
        transforms.RandomHorizontalFlip(p = 1)
    ])
    
    dataset2 = im2spec_Dataset(images, spectra, transform = transform1)
    
    
    # Define the transform2
    transform2 = transforms.Compose([
        AddGaussianNoise(mean=0.0, std=0.5),
        transforms.RandomVerticalFlip(p = 1)
    ])
    
    dataset3 = im2spec_Dataset(images, spectra, transform = transform2)
    
    
    # Combine the datsets
    dataset = ConcatDataset([dataset1, dataset2, dataset3])
    
    return dataset



class Error_Dataset(Dataset):

    def __init__(self, images, err_vector, transform = None):
        self.images = images
        self.error = err_vector
        self.transform = transform

    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, idx):
        
        image = torch.tensor(self.images[idx], dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        # if error is output of a single error model
        if len(self.error.shape) == 1:
            error_data = self.error[:, np.newaxis]
            error = torch.tensor(error_data[idx], dtype=torch.float32)
            
        # if error is output of an ensemble model
        else:            
            error = torch.tensor(self.error[idx], dtype=torch.float32).unsqueeze(1)

        return image, error


def paired_images_spectra(image, hyperspectra, window_size = 30, coordinate_step = 10, image_norm = False, spectra_norm = False):
 

    """
    Extracts patches from the image and the corresponding spectra from the hyperspectra at the center of each patch.

    Inputs:
        image: 2D numpy array is the morphology
        hyperspectra: 3D numpy array is the hyperspectra
        window_size: int, the width (in pixels) of the patch considered as a feature-example for training
        coordinate_step: int, the distance between the center of two adjacent image patches

    Outputs:
        patches: 3D numpy array. the extracted image patches at index i
        training_spectra: 2D numpy array, the spectrum at the center of each patch at index 1
        coordinates: 2D numpy array, the coordinates of the center of each patch at index i
        
    """
    
    coords = get_coord_grid(image, step = coordinate_step, return_dict= False)
    #print('initial coordinates = ',coords[:, 0].shape)


    # Extract patches (or features) and the center coordinates of each patch.
    extracted_features = extract_subimages(image, coordinates = coords, window_size = window_size)
    patches, coordinates, _ = extracted_features
    patches = patches.squeeze()

    if image_norm:
        for i in range(len(patches)):
            patches[i] = norm_0to1(patches[i])

    #total number of pathces that are extracted
    n, _, _ = patches.shape

    
    n_dim = int(n**0.5)
    points = hyperspectra.shape[-1]

    #Reshape the training spectra to the same training set as the image patches
    training_spectra = np.zeros((n_dim, n_dim, points)) 

    for i in range(points):

        # Extract the spectra at the center of each patch
        training_spectra[:, :, i] = cv2.resize(hyperspectra[:, :, i], (n_dim, n_dim))

    # Reshape the training spectra so that each row is a spectra
    training_spectra = training_spectra.reshape(n, -1)
    
    if spectra_norm:
        for i in range(len(training_spectra)):
            training_spectra[i] = norm_0to1(training_spectra[i])


    return patches, training_spectra, coordinates


def paired_images_spectra_1(image, cits_obj, hyperspectra, window_size = 30, coordinate_step = 10, image_norm = False, spectra_norm = False):
 

    """
    Extracts patches from the image and the corresponding spectra from the hyperspectra at the center of each patch.
    The spectra is the closest point (present in the dataset) at the center of the patch

    Inputs:
        image: 2D numpy array is the morphology
        hyperspectra: 3D numpy array is the hyperspectra
        window_size: int, the width (in pixels) of the patch considered as a feature-example for training
        coordinate_step: int, the distance between the center of two adjacent image patches

    Outputs:
        patches: 3D numpy array. the extracted image patches at index i
        training_spectra: 2D numpy array, the spectrum at the center of each patch at index 1
        coordinates: 2D numpy array, the coordinates of the center of each patch at index i
        
    """
    
    coords = get_coord_grid(image, step = coordinate_step, return_dict= False)
    #print('initial coordinates = ',coords[:, 0].shape)


    # Extract patches (or features) and the center coordinates of each patch.
    extracted_features = extract_subimages(image, coordinates = coords, window_size = window_size)
    patches, coordinates, _ = extracted_features
    patches = patches.squeeze()
    

    if image_norm:
        for i in range(len(patches)):
            patches[i] = norm_0to1(patches[i])

    #total number of pathces that are extracted
    n, _, _ = patches.shape

 

    scan_frame = cits_obj.get_frame_size()
    
    image_pixels = image.shape[0]
    training_spectra = []

    for i in range(len(coordinates)):
        coordinate_point = coordinates[i]*scan_frame/image_pixels
        coord_val, cits_coord = cits_obj.nearest_point(coordinate_point)
        spectra = hyperspectra[cits_coord[0], cits_coord[1], :]
        
        if spectra_norm:
            spectra = norm_0to1(spectra)
        training_spectra.append(spectra)
    

    training_spectra = np.asarray(training_spectra)
    
    #Normalized globally over the label set
    training_spectra = norm_0to1(training_spectra)
    
    
    # Reshape the training spectra so that each row is a spectra
    training_spectra = training_spectra.reshape(n, -1)
    

    return patches, training_spectra, coordinates



def norm_0to1(arr):
    arr = np.asarray(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr


class AddGaussianNoise():
    """
    Adds Gaussian noise to the input tensor

    Input:
        noise_factor: float, optional, default: 0.1
        mean: float, optional, default: 0.0
        std: float, optional, default: 1.0

    Output:
        tensor: tensor with added noise
    """    
    
    def __init__(self, noise_factor=0.1, mean=0.0, std=1.0):

        self.noise_factor = noise_factor
        self.mean = mean
        self.std = std
        

    def __call__(self, tensor):

        return tensor + self.noise_factor * torch.normal(mean=self.mean, std=self.std, size=tensor.size())
