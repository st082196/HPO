import os
import itertools
import csv
import torch
from math import pi, sqrt
from time import time
from torch import nn
from torch.func import vmap, hessian
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from matplotlib.pyplot import figure, subplots, plot, semilogy, title, xlabel, ylabel, legend, savefig, close

# Compare loss(time) plots
directory = 'models/ndims=5'
ndims = [5]
bounds = [0, 1]
N = [512] # number of nodes per hidden layer
L = [2] # number of hidden layers
activation = ['sin'] # activation function
num_points = [1000000]
use_sobol = [False, True] # whether to generate training points using Sobol sequence (True) or uniformly (False)
batch_size = [4096]
lr = [0.01]
lr_scheduler = ['ReduceLROnPlateau-0.1-2', 'ReduceLROnPlateau-0.5-2']

figure()
for ndims, N, L, activation, num_points, use_sobol, batch_size, lr, lr_scheduler in itertools.product(ndims, N, L, activation, num_points, use_sobol, batch_size, lr, lr_scheduler):
    name = f'{ndims},{bounds},{N},{L},{activation},{num_points},{use_sobol},{batch_size},{lr},{lr_scheduler}'
    if os.path.isdir(os.path.join(directory, name)):
        with open(os.path.join(directory, name, 'loss(time).csv')) as datafile:
            data = list(csv.reader(datafile, quoting=csv.QUOTE_NONNUMERIC))
        del data[0]
        epoch_data, time_data, MSE_f_train_data, MSE_f_data, RMSE_u_data, lr_data = [list(x) for x in zip(*data)]
        semilogy(time_data, MSE_f_data, label=f'use_sobol={use_sobol}, lr_scheduler={lr_scheduler}')
title(f'ndims={ndims}, N={N}, L={L}, σ={activation}, num_points={num_points}, batch_size={batch_size}, lr={lr}', fontsize=10)
xlabel('time [sec]')
ylabel('MSE_f')
legend()
savefig(os.path.join(directory, 'loss(time)-numpoints=1000000.pdf'), bbox_inches='tight')