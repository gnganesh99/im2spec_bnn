

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset

from im2spec_models import *

def norm_0to1(arr):
    arr = np.asarray(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr


def train_model(model, imgs_train, spectra_train, n_epochs = 100):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100

    train_loss = []

    model.train()

    train_images = torch.tensor(imgs_train, dtype=torch.float32)
    train_spectra = torch.tensor(spectra_train, dtype=torch.float32)


    for epoch in range(n_epochs):
        
        optimizer.zero_grad()
        outputs = model(train_images)
        loss = criterion(outputs, train_spectra)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())


    model.eval()

    return model, train_loss


def train_model_ensemble(model, dataset, n_epochs = 100):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    
    optimizers = [torch.optim.Adam(submodel.parameters(), lr=0.1) for submodel in model.models]

    train_loss = []

    model.train()

    dataloader = DataLoader(dataset, batch_size = len(dataset)//3, shuffle = True)


    for epoch in range(n_epochs):

        for train_images, train_spectra in dataloader:
            
            train_images, train_spectra = train_images.to(device), train_spectra.to(device)
            
            loss_vector = []
            
            for idx, submodel in enumerate(model.models):

                optimizers[idx].zero_grad()
                
                output = submodel(train_images)
            
                loss = criterion(output, train_spectra)

                loss.backward()
                optimizers[idx].step()

                loss_vector.append(loss.item())


        train_loss.append(loss_vector)


    model.eval()

    return model, train_loss



def train_error_ensemble(model, dataset, n_epochs = 100):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    
    optimizers = [torch.optim.Adam(submodel.parameters(), lr=0.1) for submodel in model.models]

    train_loss = []

    model.train()

    dataloader = DataLoader(dataset, batch_size = 1 + len(dataset)//3, shuffle = True)


    for epoch in range(n_epochs):

        for train_images, error_vector in dataloader:

            train_images, error_vector = train_images.to(device), error_vector.to(device)
        
            loss_vector = []
            
            for idx, submodel in enumerate(model.models):

                optimizers[idx].zero_grad()
                
                output = submodel(train_images)
            
                loss = criterion(output, error_vector[:, idx])

                loss.backward()
                optimizers[idx].step()

                loss_vector.append(loss.item())


        train_loss.append(loss_vector)


    model.eval()

    return model, train_loss


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


def acquisition_fn(error_mean, error_std, beta = 1, index_exclude = [], sample_next_point = 1):
    
    aq_fn = beta * error_mean + error_std

    aq_fn = np.asarray(aq_fn)
    
    aq_fn[index_exclude] = 1

    # acq_val_min = aq_fn.min()

    # acq_ind = [index for index, acq_val in enumerate(aq_fn) if acq_val == acq_val_min]

    aq_ind = np.argsort(aq_fn)[:sample_next_point]  # sort indices in ascending order of the acquisition function and select the first sample_next_point indices

    return aq_ind, aq_fn


def append_training_set(images, spectra, next_index, imgs_train, spectra_train, indices_train):

    imgs_train = np.append(imgs_train, images[next_index].reshape(1, images.shape[1], images.shape[2]), axis = 0)

    spectra_train = np.append(spectra_train, spectra[next_index].reshape(1, spectra.shape[1]), axis = 0)
    
    indices_train = np.append(indices_train, next_index)

    return imgs_train, spectra_train, indices_train


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