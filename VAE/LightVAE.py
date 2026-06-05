import numpy as np
from torch.autograd import Variable
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, Dataset, DataLoader
import os
import pandas as pd

class VAE(nn.Module):
    def __init__(self, input_dim=17409, latent_dim=1024, p_dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encode_block = nn.Sequential(
            nn.Linear(input_dim, 2**12),
            nn.BatchNorm1d(2**12),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            
            nn.Linear(2**12, 2**11),
            nn.BatchNorm1d(2**11),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            
            nn.Linear(2**11, self.latent_dim*2)
        )
        
        self.decode_block = nn.Sequential(
            nn.Linear(self.latent_dim, 2**11),
            nn.BatchNorm1d(2**11),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            
            nn.Linear(2**11, 2**12),
            nn.BatchNorm1d(2**12),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            
            nn.Linear(2**12, input_dim)
        )
         
    def encode(self, x): 
        mu, logvar = torch.split(self.encode_block(x), self.latent_dim, dim=-1)   
        return mu, logvar 

    def gaussian_sampler(self, mu, logvar):
        if self.training: 
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * torch.clamp(logvar, -20, 20))
            sample = mu + (eps * std)
            return sample
        else:
            return mu

    def decode(self, z):
        reconstruction = self.decode_block(z)
        return reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.gaussian_sampler(mu, logvar)
        reconstruction = self.decode(z)
        return mu, logvar, reconstruction
    
    def get_latent(self, x):
        mu, logvar = self.encode(x)
        z = self.gaussian_sampler(mu, logvar)
        return z