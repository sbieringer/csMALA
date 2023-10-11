from typing import Any
import torch
from blitz.modules import BayesianLinear

from util.VBLinear import VBLinear
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

def NCP_loss(model, x, y, **kwargs):
    '''implements an noise contrastive prior loss as in 1807.09289. If more then one (the last) layer is Bayesian and point_estimate = False the model[:,0] liklihood is assumed to be gaussian for simplicity.
    
    args: model - model_util.RegressionNet: model to evaluate
          x - torch.Tensor: x.shape = (batchsize, model.n_dims), input tensor
          y - torch.Tensor: y.shape = (batchsize, model.n_dims_out), target tensor
    
    kwargs: ncp_noise_std - FLOAT: size of noise to add to x to sample out-of-distribution (OOD) values, default = 0.5
            ncp_center_at_target - BOOL: if True data prior is centered at corresponding target valuus, default = True
            ood_std_prior - FLOAT: standard deviation of the data prior    
            point_estimate - BOOL: if True the KL between proposed posterior and the data prior is calculated, 
                                   if False the KL between model[:,0] liklihood (over weight posteriors) abd data prior is calculated,
                                   default FALSE
            sample_nbr - INT: number of samples in mean and standard deviation, if more then one (the last) layer is Bayesian and point_estimate = False
    '''
    
    N = model.n_dims_out
    
    ood_inputs = x[:,0] + torch.normal(0.0, kwargs.get('ncp_noise_std', 0.5), size=(len(x),))    
    ood_prior_means = y[:,0] if kwargs.get('ncp_center_at_target', True) else 0
    ood_prior_std = kwargs.get('ood_std_prior', 0.1)
    
    ood_inputs = ood_inputs.view(-1,1)
    
    if kwargs.get('point_estimate', False):
        assert model.n_dims_out==2
        out = model(ood_inputs) 
        ood_model_means = out[:,0]
        ood_model_std = out[:,1]
    
    #else:
    elif sum([isinstance(layer, VBLinear) or isinstance(layer, BayesianLinear) for layer in model.modules()])>1:
        out = []
        for i in range(kwargs.get('sample_nbr',1)):
            _ = [layer.reset_random() for layer in model.modules() if isinstance(layer, VBLinear)]
            out.append(model(ood_inputs).view(1,-1,model.n_dims_out))
        out = torch.cat(out, 0)
        ood_model_means = torch.mean(out[:,:,0], 0)
        ood_model_std = torch.std(out[:,:,0], 0)
    
    else:
        b_layer = model.last_layer
        if isinstance(b_layer, VBLinear):
            w_mean = b_layer.mu_w
            w_var = torch.exp(b_layer.logsig2_w)
            b_mean = b_layer.bias            
        elif isinstance(b_layer, BayesianLinear):
            w_mean = b_layer.weight_mu
            w_var = torch.log1p(torch.exp(b_layer.weight_rho))**2
            b_mean = b_layer.bias_mu
        
        hidden = model.hidden_forward(x)
        ood_model_means = torch.matmul(hidden, w_mean.T) + b_mean.T
        ood_model_std = torch.sqrt(torch.matmul(hidden ** 2, w_var.T))
        ood_model_means = ood_model_means[:,0]
        ood_model_std = ood_model_std[:,0]
                                                                      
    dkl = 0.5*(torch.sum(torch.log(ood_model_std**2/ood_prior_std**2),0)
               - N
               + ood_prior_std**2/torch.sum(ood_model_std**2,0) 
               + torch.sum((ood_model_means-ood_prior_means)**2/ood_model_std**2, 0))
        
    return dkl
