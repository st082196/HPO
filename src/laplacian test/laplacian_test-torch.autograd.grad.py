import os
import itertools
import csv
import torch
from math import pi, sqrt
from time import time
from torch import nn
from torch.func import vmap, grad, hessian
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from matplotlib.pyplot import figure, subplots, plot, xlabel, ylabel, title, savefig, close
device = 'cuda'


class HelmholtzSolver(nn.Module):
    def __init__(self, ndims, N, L, activation, bounds, g=lambda x: 0):
        super(HelmholtzSolver, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(ndims, N), activation,
            *[nn.Linear(N, N), activation]*(L-1),
            nn.Linear(N, 1),
        )
        self.bounds = bounds
        self.g = g


    def forward(self, x):
        # enforce boundary condition
        return self.g(x) + torch.prod((x-self.bounds[0])*(self.bounds[1]-x), -1, True)*self.layers(x)


class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class Atan(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.atan(x)/(pi/2)


def SSE_f_fn(model, x, u):
    d1 = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2 = torch.stack([torch.autograd.grad(d1[:,i], x, grad_outputs=torch.ones_like(d1[:,i]), create_graph=True)[0][:,i] for i in range(d1.size(1))], 1)
    return torch.sum((torch.sum(d2, -1, True) + u + (ndims*4*pi**2 - 1)*torch.prod(torch.sin(2*pi*x), -1, True))**2)


def train(model, optimizer, num_points, use_sobol=True, batch_size=1024, max_time=None, MSE_f=0, i=0):
    start_time = time()
    if max_time is None:
        max_time = float('inf')
    model.train()
    torch.manual_seed(2022)
    torch.rand(i, ndims, device=device)
    if use_sobol:
        sobolengine = torch.quasirandom.SobolEngine(ndims, True, 2022)
        sobolengine.fast_forward(i)
    SSE_f = MSE_f*i
    while i < num_points and time() - start_time < max_time:
        n = min(batch_size, num_points - i)
        if use_sobol:
            x = (sobolengine.draw(n)*(bounds[1] - bounds[0]) + bounds[0]).to(device).requires_grad_()
        else:
            x = (torch.rand(n, ndims, device=device)*(bounds[1] - bounds[0]) + bounds[0]).requires_grad_()
        pred = model(x)
        loss = SSE_f_fn(model, x, pred)
        SSE_f += loss.item()
        loss /= n
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += n
    MSE_f = SSE_f/i
    return MSE_f, i


def test(model, num_points, batch_size=1):
    model.eval()
    SSE_f, SSE_u = 0, 0
    torch.manual_seed(2023)
    for i in range(0, num_points, batch_size):
        x = (torch.rand(min(batch_size, num_points - i), ndims, device=device)*(bounds[1] - bounds[0]) + bounds[0]).requires_grad_()
        u = torch.prod(torch.sin(2*pi*x), -1, True)
        pred = model(x)
        SSE_f += SSE_f_fn(model, x, pred).item()
        SSE_u += torch.sum((pred-u)**2).item()
    MSE_f = SSE_f/num_points
    RMSE_u = sqrt(SSE_u/num_points)
    return MSE_f, RMSE_u


# Parameters
directory = 'models/laplacian test'
ndims = [2, 5, 10]
bounds = [0, 1]
N = [512] # number of nodes per hidden layer
L = [2] # number of hidden layers
activation = ['sin'] # activation function
num_points = [100000]
use_sobol = [True] # whether to generate training points using Sobol sequence (True) or uniformly (False)
batch_size = [4096]
lr = [0.01]
lr_scheduler = ['ReduceLROnPlateau-0.5-2']
max_time = 60
load_model = False # whether to load the model (True) or overwrite with a new one (False)

for ndims, N, L, activation, num_points, use_sobol, batch_size, lr, lr_scheduler in itertools.product(ndims, N, L, activation, num_points, use_sobol, batch_size, lr, lr_scheduler):
    name = f'{ndims},{bounds},{N},{L},{activation},{num_points},{use_sobol},{batch_size},{lr},{lr_scheduler},torch.autograd.grad'
    savedir = os.path.join(directory, name)

    # Initialization
    activation = {
        'ELU': nn.ELU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'sin': Sin,
        'atan': Atan,
    }[activation]()
    torch.manual_seed(82196)
    model = HelmholtzSolver(ndims, N, L, activation, bounds).to(device)
    optimizer = Adam(model.parameters(), lr)
    lr_scheduler = lr_scheduler.split('-')
    if lr_scheduler[0] == 'ExponentialLR':
        lr_scheduler = ExponentialLR(optimizer, float(lr_scheduler[1]))
    elif lr_scheduler[0] == 'ReduceLROnPlateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', float(lr_scheduler[1]), int(lr_scheduler[2]))

    if load_model and os.path.exists(os.path.join(savedir, 'model.pt')):
        checkpoint = torch.load(os.path.join(savedir, 'model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_point = checkpoint['last_point']
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        with open(os.path.join(savedir, 'loss(time).csv')) as datafile:
            data = list(csv.reader(datafile, quoting=csv.QUOTE_NONNUMERIC))
        del data[0]
        epoch, elapsed_time, MSE_f_train, MSE_f, RMSE_u, lr = data[-1]
        if last_point > 0:
            del data[-1]
        epoch_data, time_data, MSE_f_train_data, MSE_f_data, RMSE_u_data, lr_data = [list(x) for x in zip(*data)]
    else:
        epoch, elapsed_time, MSE_f_train, last_point = 0, 0, 0, 0
        epoch_data, time_data, MSE_f_train_data, MSE_f_data, RMSE_u_data, lr_data = [], [], [], [], [], []

    # Training
    start_time = time() - elapsed_time
    output_timestamp = save_timestamp = time()
    print(f'\n{name}\nepoch | time  | train MSE_f | test MSE_f | test RMSE_u | lr')
    while elapsed_time < max_time:
        MSE_f_train, last_point = train(model, optimizer, num_points, use_sobol, batch_size, max_time - elapsed_time, MSE_f_train, last_point)
        MSE_f, RMSE_u = test(model, num_points, 10000)
        lr = optimizer.param_groups[0]['lr']
        epoch += last_point/num_points
        if last_point >= num_points:
            if isinstance(lr_scheduler, ExponentialLR):
                lr_scheduler.step()
            elif isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(MSE_f)
            last_point = 0
        elapsed_time = time() - start_time
        epoch_data.append(epoch)
        time_data.append(elapsed_time)
        MSE_f_train_data.append(MSE_f_train)
        MSE_f_data.append(MSE_f)
        RMSE_u_data.append(RMSE_u)
        lr_data.append(lr)
        if time() - output_timestamp >= 1 and elapsed_time < max_time:
            print(f'{epoch:5.0f} | {elapsed_time:5.0f} | {MSE_f_train:11.3e} | {MSE_f:10.3e} | {RMSE_u:11.4f} | {lr:.1e}')
            output_timestamp = time()
        if time() - save_timestamp >= 600 or elapsed_time >= max_time:
            save_timestamp = time()
            
            # Saving
            os.makedirs(savedir, exist_ok=True)
            with open(os.path.join(savedir, 'loss(time).csv'), 'w+') as output:
                output.write('"epoch","time","train MSE_f","test MSE_f","test RMSE_u","lr"\n')
                output.writelines([f'{epoch_data[i]:.2f},{time_data[i]:.2f},{MSE_f_train_data[i]:.3e},{MSE_f_data[i]:.3e},{RMSE_u_data[i]:.3e},{lr_data[i]:.1e}\n' for i in range(len(time_data))])
            state_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            if lr_scheduler is not None:
                state_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            state_dict['last_point'] = last_point
            torch.save(state_dict, os.path.join(savedir, 'model.pt'))

            # Plot loss(time) and RMSE(time)
            fig, axs = subplots(1, 2, figsize=(20,7))
            axs[0].semilogy(time_data, MSE_f_data)
            axs[0].set_xlabel('time [sec]')
            axs[0].set_ylabel('MSE_f')
            axs[1].semilogy(time_data, RMSE_u_data)
            axs[1].set_xlabel('time [sec]')
            axs[1].set_ylabel('RMSE_u')
            savefig(os.path.join(savedir, 'loss(time).pdf'), bbox_inches='tight')
            close()

            start_time += time() - save_timestamp # subtract time spent on saving from measured time
            save_timestamp = time()

    print(f'{epoch:5.2f} | {elapsed_time:5.0f} | {MSE_f_train:11.3e} | {MSE_f:10.3e} | {RMSE_u:11.4f} | {lr:.1e}')
    with open('results.csv', 'a') as output:
        output.write(f'{name},{epoch:.2f},{elapsed_time:.0f},{MSE_f_train:.3e},{MSE_f:.3e},{RMSE_u:.3e},{lr:.1e}\n')

    # Plot sin(x)
    if ndims == 1:
        x = torch.arange(0, 1, 0.001).to(device)
    else:
        x = torch.cartesian_prod(*[torch.arange(0, 1, 0.005)]*2, *[torch.tensor([1/4])]*(ndims-2)).to(device)
    u = torch.prod(torch.sin(2*pi*x), -1, True) # analytical solution
    x, u, pred = x.cpu().detach(), u.cpu().detach(), model(x).cpu().detach()
    if ndims == 1:
        figure(figsize=(10,3))
        plot(x, u, '.', markersize=1)
        plot(x, pred, '.', markersize=1)
        xlabel('x')
        ylabel('u')
        savefig(os.path.join(savedir, 'sin(x).png'), bbox_inches='tight')
        close()
    else:
        fig = figure(figsize=(20,10))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(x[:,0], x[:,1], pred, c=pred, cmap='coolwarm', antialiased=False)
        title('u(x), NN solution')
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(x[:,0], x[:,1], pred-u, c=pred-u, cmap='coolwarm', antialiased=False)
        title('NN solution - analytical solution')
        savefig(os.path.join(savedir, 'sin(x).png'), bbox_inches='tight')
        close()