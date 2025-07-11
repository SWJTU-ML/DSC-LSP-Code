import h5py
import torch
import numpy as np
import scipy.io as scio
import numpy as np
import os
import pickle
from sklearn.datasets import make_moons
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from utils import *
from keras.datasets import reuters
import argparse
from sklearn.decomposition import PCA
from kymatio import Scattering2D
import torch
from torchvision import datasets, transforms
import pandas as pd

def load_mydata(dataname) -> tuple:
    # Pendigits, fashion_mnist, mnist,MSRA25_uni,mnist_test,CNAE,mfeat_kar,isolet1234
    d_path = "./dataset/"+ dataname + ".mat"  
    data = scio.loadmat(d_path)
    if dataname in ['Pendigits','mfeat_kar','isolet1234','CNAE']:
        x=data['data']
        y=data['labels']
        s_x=StandardScaler()
        x=s_x.fit_transform(x)
        y=y.reshape(y.shape[0])
        cluster_num=len(np.unique(y)) 
    elif dataname in ['fashion_mnist','MSRA25_uni']:
        x= data['X']
        y= data['Y'].T
        y=y.flatten()
        x=x/255
        cluster_num=len(np.unique(y))
        y=y.astype(int)
    elif dataname == "mnist":
        x=data['data'].T
        y=data['label'].T
        x=x/255
        y=y.reshape(len(y))
        cluster_num=len(np.unique(y))
    else:   # mnist-test
        x = data['data']
        y = data['label']
        x = x/255
        y = y.reshape(len(y))  

    xy_zip = list(zip(x, y))
    xy_train, xy_test = train_test_split(xy_zip, train_size=0.9 , shuffle=True) 
    x_train, y_train = zip(*xy_train)
    x_test, y_test = zip(*xy_test)

    x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
    return x_train, y_train, x_test, y_test

def load_data(dataset: str) -> tuple:
    
    x_train, y_train, x_test, y_test = load_mydata(dataset)
        
    return x_train, x_test, y_train, y_test