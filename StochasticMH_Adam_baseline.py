import torch
import urllib.request as urllib
from argparse import ArgumentParser

import sys
sys.path.append('src')

from util.loss_util import L2_Bern_loss
from util.model_util import RegressionNet
from datasets.toy import load_1d, RegressionData
from datasets.samplers import naive_Bernoulli_Dataloader
from uncertimators.non_bayes_unc import non_bayes_uncertimator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = ArgumentParser()
parser.add_argument('--folder', type=str, default="sigma_data_002/n10000/lambda1/rho0.1/0/")
parser.add_argument('--n_points', type=int, default=10000)
parser.add_argument('--rho', type=float, default=0.1)
parser.add_argument('--lambda_factor', type=float, default=1)
parser.add_argument('--sigma_data', type=float, default=0.02)
parser.add_argument('--num_dataloader_workers', type=int, default=0)

runargs = parser.parse_args()

folder = f'./figs/StochasticMH/{runargs.folder}/'
n_points = runargs.n_points
p = runargs.rho
lambda_factor = runargs.lambda_factor
n = 170000

load = False
sigma_data = runargs.sigma_data

X_tmp, Y_tmp = load_1d(folder, n_points, sigma_data)
X = torch.Tensor(X_tmp).view(-1,1)
Y = torch.Tensor(Y_tmp).view(-1,1)

X_tmp, Y_tmp = load_1d(folder, n_points, sigma_data)
X_val = torch.Tensor(X_tmp).view(-1,1)
Y_val = torch.Tensor(Y_tmp).view(-1,1)

n_l2_int = 10000
X_test, Y_test = load_1d(folder, n_l2_int, 0)
X_test = torch.cat((torch.linspace(-2,2,500).view(-1,1), torch.Tensor(X_test).view(-1,1)), 0) #first 500 points are for plotting

data_tr = RegressionData(X, Y)
data_val = RegressionData(X_val, Y_val)

dataloader_tr = naive_Bernoulli_Dataloader(data_tr, p)
if runargs.num_dataloader_workers == 0:
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=len(X_val),
                        shuffle=False, num_workers=0, pin_memory=True, persistent_workers=False)
else:
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=len(X_val),
                        shuffle=False, num_workers=runargs.num_dataloader_workers, pin_memory=True, persistent_workers=True)
    
data = {'train': dataloader_tr, 'val': dataloader_val}
loss = L2_Bern_loss(n_points, p, use_mean=False)
net = RegressionNet(dim_in=1, dim_out=1, ndf=100, dropout=0, activation=torch.nn.ReLU, layers=3*[torch.nn.Linear], layer_kwargs=3*[{}], device=device).to(device)

warm_up_unc = non_bayes_uncertimator(net, loss, data, device)

warm_up_unc.train(2000, {'opt': torch.optim.SGD(net.parameters(), lr = 1e-3)})

warm_up_unc.train(n, {'opt': torch.optim.Adam(net.parameters(), lr = 1e-3)})
Y_sampled_test = warm_up_unc.get_samples(X_test.to(device))

warm_up_unc.save(folder +'/Adam_baseline/')
torch.save(Y_sampled_test, folder +'/Adam_baseline/'+'sampled_points.pt')