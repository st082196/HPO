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
from ray import tune
from ray.air import session, RunConfig, CheckpointConfig
from ray.air.checkpoint import Checkpoint
from ray.tune import CLIReporter
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
from matplotlib.pyplot import figure, subplots, plot, semilogy, title, xlabel, ylabel, legend, savefig, close
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
    d2 = torch.diagonal(vmap(lambda I: torch.autograd.grad(d1, x, grad_outputs=torch.ones_like(d1)*I, create_graph=True)[0], 1)(torch.eye(ndims, device=device)), 0, 0, 2)
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
    if i > 0:
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


def trainable(params):
    N = params['N'] if 'N' in params else 512
    L = params['L'] if 'L' in params else 2
    activation = params['activation'] if 'activation' in params else 'tanh'
    num_points = params['num_points'] if 'num_points' in params else 1000000
    use_sobol = params['use_sobol'] if 'use_sobol' in params else True
    batch_size = params['batch_size'] if 'batch_size' in params else 1024
    lr = params['lr'] if 'lr' in params else 0.01
    lr_scheduler = params['lr_scheduler'] if 'lr_scheduler' in params else 'None'
    name = f'{ndims},{bounds},{N},{L},{activation},{num_points},{use_sobol},{batch_size},{lr},{lr_scheduler}'

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
    if lr_scheduler[0] == 'None':
        lr_scheduler = None
    elif lr_scheduler[0] == 'ExponentialLR':
        lr_scheduler = ExponentialLR(optimizer, float(lr_scheduler[1]))
    elif lr_scheduler[0] == 'ReduceLROnPlateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', float(lr_scheduler[1]), int(lr_scheduler[2]))

    checkpoint = session.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint = torch.load(os.path.join(checkpoint_dir, 'model.pt'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_point = checkpoint['last_point']
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            with open(os.path.join(checkpoint_dir, 'loss(time).csv')) as datafile:
                data = list(csv.reader(datafile, quoting=csv.QUOTE_NONNUMERIC))
        del data[0]
        epoch, elapsed_time, MSE_f_train, MSE_f, RMSE_u, lr = data[-1]
        epoch = int(epoch)
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
        checkpoint = None
        MSE_f_train, last_point = train(model, optimizer, num_points, use_sobol, batch_size, max_time - elapsed_time, MSE_f_train, last_point)
        MSE_f, RMSE_u = test(model, last_point if epoch < 1 else num_points, 10000)
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
        if time() - save_timestamp >= save_period or elapsed_time >= max_time:
            save_timestamp = time()
            
            # Saving
            os.makedirs('data', exist_ok=True)
            with open(os.path.join('data', 'loss(time).csv'), 'w+') as output:
                output.write('"epoch","time","train MSE_f","test MSE_f","test RMSE_u","lr"\n')
                output.writelines([f'{epoch_data[i]:.2f},{time_data[i]:.2f},{MSE_f_train_data[i]:.3e},{MSE_f_data[i]:.3e},{RMSE_u_data[i]:.3e},{lr_data[i]:.1e}\n' for i in range(len(time_data))])
            state_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
            if lr_scheduler is not None:
                state_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            state_dict['last_point'] = last_point
            torch.save(state_dict, os.path.join('data', 'model.pt'))
            checkpoint = Checkpoint.from_directory('data')

            # Plot loss(time) and RMSE(time)
            fig, axs = subplots(1, 2, figsize=(20,7))
            axs[0].semilogy(time_data, MSE_f_data)
            axs[0].set_xlabel('time [sec]')
            axs[0].set_ylabel('MSE_f')
            axs[1].semilogy(time_data, RMSE_u_data)
            axs[1].set_xlabel('time [sec]')
            axs[1].set_ylabel('RMSE_u')
            savefig(os.path.join('data', 'loss(time).pdf'), bbox_inches='tight')
            close()

            start_time += time() - save_timestamp # subtract time spent on saving from measured time
            save_timestamp = time()

        session.report({'epoch': epoch, 'elapsed_time': elapsed_time, 'MSE_f_train': MSE_f_train, 'MSE_f': MSE_f, 'RMSE_u': RMSE_u, 'lr': lr}, checkpoint=checkpoint)


df_dir = 'HPO_dataframes'
local_dir = 'ray_results'
os.makedirs(df_dir, exist_ok=True)

# ndims = 2
# bounds = [0, 2]
# name = f'{ndims=},{bounds=},HyperOptSearch+ASHAScheduler'
# max_time = 30
# save_period = 30
# tuner = tune.Tuner(
    # tune.with_resources(
        # trainable,
        # resources={'cpu': 2, 'gpu': 1}
    # ),
    # param_space={
        # 'N': tune.qlograndint(64, 2048, 32),
        # 'L': tune.randint(1, 5),
        # 'activation': tune.choice(['ELU', 'sigmoid', 'tanh', 'sin', 'atan']),
        # 'num_points': tune.choice([100, 1000, 10000]),
        # 'batch_size': tune.qlograndint(64, 32768, 32),
        # 'lr': tune.qloguniform(1e-4, 1, 1e-4),
        # 'lr_scheduler': tune.choice(['None', 'ExponentialLR-0.95', 'ReduceLROnPlateau-0.1-10', 'ReduceLROnPlateau-0.5-2']),
    # },
    # tune_config=tune.TuneConfig(
        # mode="min",
        # metric="MSE_f",
        # search_alg=HyperOptSearch(random_state_seed=2023),
        # scheduler=ASHAScheduler(time_attr='elapsed_time', max_t=max_time),
        # num_samples=-1,
        # time_budget_s=3600,
    # ),
    # run_config = RunConfig(
        # name=name,
        # local_dir=local_dir,
        # progress_reporter=CLIReporter(max_report_frequency=60),
    # )
# )
# results = tuner.fit()
# df = results.get_dataframe()
# df.to_csv(os.path.join(df_dir, f'{name}.csv'))

ndims = 5
bounds = [0, 2]
name = f'{ndims=},{bounds=},HyperOptSearch+ASHAScheduler'
max_time = 300
save_period = 300
tuner = tune.Tuner(
    tune.with_resources(
        trainable,
        resources={'cpu': 2, 'gpu': 1}
    ),
    param_space={
        'N': tune.qlograndint(64, 2048, 32),
        'L': tune.randint(1, 5),
        'activation': tune.choice(['ELU', 'sigmoid', 'tanh', 'sin', 'atan']),
        'num_points': tune.choice([10000, 100000, 1000000]),
        'batch_size': tune.qlograndint(64, 32768, 32),
        'lr': tune.qloguniform(1e-4, 1, 1e-4),
        'lr_scheduler': tune.choice(['None', 'ExponentialLR-0.95', 'ReduceLROnPlateau-0.1-10', 'ReduceLROnPlateau-0.5-2']),
    },
    tune_config=tune.TuneConfig(
        mode="min",
        metric="MSE_f",
        search_alg=HyperOptSearch(random_state_seed=2023),
        scheduler=ASHAScheduler(time_attr='elapsed_time', max_t=max_time),
        num_samples=-1,
        time_budget_s=24*3600,
    ),
    run_config = RunConfig(
        name=name,
        local_dir=local_dir,
        progress_reporter=CLIReporter(max_report_frequency=60),
    )
)
results = tuner.fit()
df = results.get_dataframe()
df.to_csv(os.path.join(df_dir, f'{name}.csv'))