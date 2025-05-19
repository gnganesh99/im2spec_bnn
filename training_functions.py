

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset
#from im2spec_dataset import augmented_dataset

 

def spectral_mismatch_error(pred_spectra, spectra):

    mse = np.mean(np.abs((pred_spectra - spectra)), axis=1)

    return mse

def norm_0to1(arr):
    arr = np.asarray(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr

def distance_acq_fn(distances, beta = 0.5, lambda_ = 1, optimize = "custom_fn", sample_next_points = 10, exclude_indices = []):

    distances = np.ravel(np.asarray(distances))
    acq_vals = norm_0to1(distances)



    if optimize == "minimize":

        acq_vals[exclude_indices] = 2
        aq_ind = np.argsort(acq_vals)


    elif optimize == "maximize":

        acq_vals[exclude_indices] = -1
        aq_ind = np.argsort(acq_vals)[::-1]

    elif optimize == "custom_fn":

                    # EXPLORATION + EXPLOITATION
        acq_vals = (1-np.exp(-lambda_ * np.abs(acq_vals-(1-beta))))
        acq_vals = norm_0to1(acq_vals)

        #acq_vals = beta*(1- np.exp(-lambda_ * distances)) + (1-beta)*np.exp(-lambda_ * distances)

        acq_vals[exclude_indices] = -1
        aq_ind = np.argsort(acq_vals)[::-1]

    else:
        raise ValueError('Invalid optimization type')


    aq_ind = aq_ind[:sample_next_points]


    return aq_ind, acq_vals

def append_training_set(images, spectra, next_index, imgs_train, spectra_train, indices_train):

    imgs_train = np.append(imgs_train, images[next_index].reshape(1, images.shape[1], images.shape[2], 1), axis = 0)

    spectra_train = np.append(spectra_train, spectra[next_index].reshape(1, spectra.shape[1]), axis = 0)

    indices_train = np.append(indices_train, next_index)

    return imgs_train, spectra_train, indices_train







def err_estimation(model, images, spectra):

    images = torch.tensor(images, dtype=torch.float32)
    spectra = torch.tensor(spectra, dtype=torch.float32)

    model.eval()
    
    outputs = model.predict(images)
    #print(outputs.shape)
    error_vector = np.abs(outputs.squeeze(1) - spectra)
    #print(error_vector.shape)

    error_vector = error_vector.detach().squeeze().numpy()
    error_mean = np.mean(error_vector, axis = -1)
    error_std = np.std(error_vector, axis = -1)

    return error_mean, error_std, error_vector





def error_dataset(model, images, spectra, norm = True):

    error_vector = []
    for submodel in model.models:

        error_mean, _, _ = err_estimation(submodel, images, spectra)

        if norm:
            error_mean = norm_0to1(error_mean)
        
        error_vector.append(error_mean)


    error_vector = np.asarray(error_vector).T
    return error_vector
   

def predict_error(error_ensemble_model, images):

    errors = []
    for image in images:
        
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        error_vector = error_ensemble_model.predict(image)

        errors.append([error.detach().numpy() for error in error_vector])

    errors = np.asarray(errors).squeeze()
    error_mean = np.mean(errors, axis = -1)
    error_std = np.std(errors, axis = -1)
    
    return error_mean, error_std, errors