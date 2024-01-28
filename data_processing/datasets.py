import pickle

import torch
#from spectral import *
from torch.utils.data import Dataset
import os

#spectral.settings.envi_support_nonlowercase_params = True

import utils
from data_processing.generate_dataset import generate_spectrogram


class SpectraDataset(Dataset):
    """
    Dataset class for loading a synthetic dataset created by the methods in generate_dataset.py

    The parameter <continuous_sample_num> come from the idea of a continuous (infinite) data generation during training,
    by creating a new dataset for each epoch. However, we noticed that 10.000 to 100.000 samples are always enough and don't take long to generate.
    """

    def __init__(self, path, continuous_sample_num=None):

        if continuous_sample_num is None:
            with open(path + "spectra.pkl", 'rb') as spectra_file:
                self.spectra = pickle.load(spectra_file)
            with open(path + "params.pkl", 'rb') as params_file:
                self.params = pickle.load(params_file)
            self.length = len(self.params)

            mean, std = utils.get_mean_std(self.spectra)
            self.spectra = (self.spectra - mean) / std

        else:
            self.spectra = None
            self.length = continuous_sample_num

    def __len__(self):

        return self.length

    def __getitem__(self, idx):
        if self.spectra is None:
            x, syn_data, syn_params = generate_spectrogram(interpolate=True)
            return syn_data, syn_params

        else:
            return self.spectra[idx], self.params[idx]

## Change for monte carlo data, see PigletDatasetSim
class PigletDataset(Dataset):
    """
    Dataset class for loading a dataset created by the optimisation.py script
    """

    def __init__(self, path, version="", range=(475, 512)):
        self.spectra, self.params = None, None
        for folder in os.listdir(path):
            if folder != '.DS_Store':
                if not (int(folder[3:6]) >= range[0] and int(folder[3:6]) <= range[1]): continue  
                with open(path + "/" + folder + "/spectra_list" + version + ".pt", 'rb') as spectra_file:
                    self.spectra = (torch.load(spectra_file) if self.spectra is None else torch.cat((self.spectra, torch.load(spectra_file)), axis=0))
                with open(path + "/" + folder + "/coef_list" + version + ".pt", 'rb') as params_file:
                    self.params = (torch.load(params_file) if self.params is None else torch.cat((self.params, torch.load(params_file)), axis=0))
            
        self.length = len(self.params)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.spectra[idx], self.params[idx]
    
## For monte carlo data, simulated data acquisition on a replica of a region of exposed brain cortex of a mouse
class MouseDataset(Dataset):
    """
    Dataset class for loading a dataset created by the optimisation.py script
    Treating each Mouse data as a seperate entiry instead of concatenating al
    """

    def __init__(self, path, folders, version=""):
        self.spectra, self.params = [], []
        for folder in folders:
            if folder != '.DS_Store':
                with open(path + "/" + folder + "/spectra" + version + ".pt", 'rb') as spectra_file:
                    self.spectra.append(torch.load(spectra_file))
                with open(path + "/" + folder + "/coef" + version + ".pt", 'rb') as params_file:
                    self.params.append(torch.load(params_file))
            
        self.length = len(self.params)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.spectra[idx], self.params[idx]
    
    
class PigletDataset2(Dataset):
    """
    Dataset class for loading a dataset created by the optimisation.py script
    """

    def __init__(self, path, version="", range=(475, 512)):
        self.spectra, self.params = None, None
        for folder in os.listdir(path):
            if folder != '.DS_Store':
                with open(path + "/" + folder + "/spectra_list" + version + ".pt", 'rb') as spectra_file:
                    self.spectra = (torch.load(spectra_file) if self.spectra is None else torch.cat((self.spectra, torch.load(spectra_file)), axis=0))
                # with open(path + "/" + folder + "/coef_list" + version + ".pt", 'rb') as params_file:
                #     self.params = []
            
        self.length = len(self.spectra)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.spectra[idx]
