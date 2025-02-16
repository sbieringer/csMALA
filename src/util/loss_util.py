from typing import Any
import torch
from numpy import pi, log

class L2_Bern_loss():
    '''
    Returns L2 distance of the true values and the prediction y_pred normalized by the expected batchsize n*rho of drawing samples with propability rho
    '''
    def __init__(self, n, rho=1, use_mean = False):
        self.n = n
        self.rho = rho
        self.use_mean = use_mean
        
    def __call__(self, y_pred, y):
        '''                
        args:   y_pred - torch.Tensor: y_pred.shape = (batchsize, model.n_dims_out), predictions for every datapoint, here: y_pred[:,0] being means and y_pred[:,1] being std dev
                y -  torch.Tensor: y.shape = (batchsize, 1): target Tensor of true datapoints
        '''
        assert y_pred.shape == y.shape
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1,1)
            y = y.reshape(-1,1)

        if self.use_mean:
            ret = torch.mean(((y_pred[:,0]-y[:,0]))**2)                   
        else:
            ret = torch.sum(((y_pred[:,0]-y[:,0]))**2)/(self.n*self.rho)     
        return ret
    
class L2_Bern_loss_corrected():
    '''
    Returns L2 distance of the true values and the prediction y_pred normalized by the expected batchsize n*rho of drawing samples with propability rho
    including the correction
    '''
    def __init__(self, n, rho=1, lamb = 1, use_mean = False, k=1):
        self.n = n
        self.rho = rho
        self.log_rho = log(rho)
        self.lamb = lamb
        self.use_mean = use_mean
        self.k = k
        self.corr_factor = None
        
    def __call__(self, y_pred, y):
        '''                
        args:   y_pred - torch.Tensor: y_pred.shape = (batchsize, model.n_dims_out), predictions for every datapoint, here: y_pred[:,0] being means and y_pred[:,1] being std dev
                y -  torch.Tensor: y.shape = (batchsize, 1): target Tensor of true datapoints
        '''
        assert y_pred.shape == y.shape
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1,1)
            y = y.reshape(-1,1)

        if self.corr_factor is None:
            self.set_corr_factor(y_pred, y)

        if self.use_mean:
            ret = torch.mean(((y_pred[:,0]-y[:,0]))**2) + self.corr_factor*(self.log_rho/self.lamb)*len(y.detach())
        else:
            ret = torch.sum(((y_pred[:,0]-y[:,0]))**2)/(self.n) + self.corr_factor*(self.log_rho/self.lamb)*len(y.detach())
        return ret
    
    def set_corr_factor(self, y_pred, y):
        if self.k == 'run_avg':
            mean_loss = torch.mean(((y_pred[:,0]-y[:,0]))**2).detach()
            if self.log_rho !=0:
                self.corr_factor = -mean_loss*self.lamb/(self.n*self.log_rho)
            else:
                self.corr_factor = 1
        else:
            self.corr_factor = self.k

class Gaussian_Liklihood_loss():
    '''
    Returns negative log likelihood (loss) of the true values for y_pred giving the mean and std deviation of a Gaussian distribution approximating the likelihood
    '''
    def __init__(self):
        pass
        
    def __call__(self, y_pred, y):
        '''                
        args:   y_pred - torch.Tensor: y_pred.shape = (batchsize, model.n_dims_out), predictions for every datapoint, here: y_pred[:,0] being means and y_pred[:,1] being std dev
                y -  torch.Tensor: y.shape = (batchsize, 1): target Tensor of true datapoints
        '''
        #sp = torch.nn.Softplus()
        std = y_pred[:,1] #sp(y_pred[:,1])+1e-6
        ret = torch.mean(-0.5*((y_pred[:,0]-y[:,0])/std)**2-torch.log(std)-0.5*torch.log(torch.Tensor([2*pi]).to(y_pred.device))) #should be a sum, but MSEloss is also mean reduced
        return -ret

class Gaussian_Liklihood_loss_corrected():
    '''
    Like Gaussian_Liklihood_loss but contains correction factor log(p)/lambda*|Z| to regain contraction behavior of full Metropolis-Hastings corrected MCMC-Adam
    '''
    def __init__(self, p, lam):
        self.p = p
        self.lam = lam

    def __call__(self, y_pred, y):
        '''
        args:   y_pred - torch.Tensor: y_pred.shape = (batchsize, model.n_dims_out), predictions for every datapoint, here: y_pred[:,0] being means and y_pred[:,1] being std dev
                y -  torch.Tensor: y.shape = (batchsize, 1): target Tensor of true datapoints
        '''
        std = y_pred[:,1] #sp(y_pred[:,1])+1e-6
        ret = torch.mean(-0.5*((y_pred[:,0]-y[:,0])/std)**2-torch.log(std)-0.5*torch.log(torch.Tensor([2*pi]).to(y_pred.device))) #should be a sum, but MSEloss is also mean reduced
        return -ret + log(self.p)/self.lam*len(std)