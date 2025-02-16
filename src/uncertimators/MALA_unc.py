import torch
import numpy as np
from copy import deepcopy as dc
from time import time
from psutil import virtual_memory

import sys
sys.path.append("../")

from uncertimators.base_uncertimator import uncertimator
from MALA import MCMC_by_bp
from datasets.samplers import Bernoulli_batch_sampler, naive_Bernoulli_Dataloader
from util.loss_util import L2_Bern_loss_corrected

class MALA_uncertimator(uncertimator):
    '''
    Class to run all the MALA code 
    '''

    def __init__(self, model, loss, data, device, loss_full = None, loss_train = None, loss_val = None):
        super().__init__(model, loss, data, device)
        dict_tmp = {'steplenght': [], 'lr': [], 'acceptance_prob': [], 'sigma': [], 
            'step_over_sigma': [], 'accepted': [], 'prob_diff': [], 'loss_diff': [], 
            'old_loss': [], 'prop_loss': [],
            'old_new_sqe': [], 'prop_next_sqe': [], #'diff_diff': [], 'diff_prob': [],
            'full_train_loss': [], 'correction_factor': []}
        if isinstance(self.data['train'], naive_Bernoulli_Dataloader):
            dict_tmp['correction_diff'] = []
            dict_tmp['batch_size_acc'] = []
        self.doc_dict.update(dict_tmp)
        self.loss_full = loss_full if loss_full is not None else loss
        self.loss_train = loss_train if loss_train is not None else loss
        self.loss_val = loss_val if loss_val is not None else loss

    def run(self, epochs, start, **kwargs):
        '''
        Loop for the MALA algorithm
        
        args:   epochs - INT: Number of repetitions over the train_dataloader
                start - bool: Start new optimization or reuse old optimzers & schedulers

        kwargs: full_loss - BOOL: stochastic or non-stochastic setting
                MH - Bool: Use metropolis-hastings correction 
                sigma - 'dynamic', None, FLOAT: if float standard deviation of the noise added to the update step
                                                if 'dynamic' the size of the update step is used
                                                if None optimzer.defaults.lr is used, if float sigma is used
                                                default = None
                sigma_factor - FLOAT: multiplicative constant on teh noise, default 0.99
                max_reps_per_step - INT: Maximum number of updates attempted on one batch, default 100
        '''

        self.kwargs = kwargs
        extended_doc_dict = kwargs.get('extended_doc_dict', True)
        bern_sampled_data = isinstance(self.data['train'], naive_Bernoulli_Dataloader)

        if bern_sampled_data:
            old_correction = None
            self.last_accepted_batch = None
        if start:
            if kwargs.get('opt', 'SGD') == 'SGD':
                self.base_opt = torch.optim.SGD(self.model.parameters(), lr = kwargs.get('lr_start', 1e-2))
            else:
                self.base_opt = torch.optim.Adam(self.model.parameters(), lr = kwargs.get('lr_start', 1e-2), betas = kwargs.get('betas', (0.9,0.999)))
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.base_opt, kwargs.get('gamma_scheduler',1))
            self.optimizer = MCMC_by_bp(self.model, self.base_opt, temperature = kwargs.get('temperature', 1), sigma = kwargs.get('sigma'))

        self.model.train()
        verbose = kwargs.get('verbose', False)
        b, sigma_fact = False, dc(kwargs['sigma_factor'])
        
        epochs = range(epochs) if isinstance(epochs, int) else range(*epochs)
        maxed_out_mbb_batches  = 0
        for epoch in epochs:
            if verbose:
                print(epoch, ' epochs', end='\r')

            if epoch%100==0 and isinstance(self.loss, L2_Bern_loss_corrected):
                self.loss.set_corr_factor(self.model(self.data['train'].dataset.x_train.reshape(-1,1).to(device = self.device, dtype = torch.float32)), 
                            self.data['train'].dataset.y_train.reshape(-1,1).to(device = self.device, dtype = torch.float32))

            for data_ep in self.data['train']: #SLOW
                self.optimizer.zero_grad()
                old_state = torch.nn.utils.parameters_to_vector(self.model.parameters())
                x = data_ep[0].reshape(-1,1).to(device = self.device, dtype = torch.float32)
                y = data_ep[1].reshape(-1,1).to(device = self.device, dtype = torch.float32)

                if extended_doc_dict:
                    training_obj = self.loss_train(self.model(x), y) #SLOW 
                    self.doc_dict['train_loss'].append(training_obj.detach().cpu().item())
                    if training_obj < -10000 or training_obj > 10000 and epoch != epochs[0]:
                        print(f'training obj {training_obj} overflow')
                    
                #define the calling functions for batchwise loss (and the full loss term)
                loop_kwargs = dc(kwargs)
                batch_loss = lambda: self.loss(self.model(x), y)
                if kwargs.get('full_loss', False):
                    def full_loss(model): #this is where hamiltorch uses the fmodel functional
                        n_data = len(self.data['train'].dataset)
                        losses = torch.zeros(n_data).to(model.device)
                        bs = self.data['train'].batch_size if self.data['train'].batch_size is not None else 100
                        n_batches = (n_data+bs-1)//bs
                        for i in range(n_batches):
                            b = self.data['train'].dataset[i*bs:(i+1)*bs]
                            loss = self.loss_full(self.model(b[0].reshape(-1,1).to(device = self.device, dtype = torch.float32).detach()), 
                                                b[1].reshape(-1,1).to(device = self.device, dtype = torch.float32).detach())
                            losses[i] = loss
                            if loss>1e6:
                                print('high loss values encountered')
                        out = torch.sum(losses)
                        return out
                    loop_kwargs['full_loss'] = full_loss

                if 'sigma_factor' in kwargs and b:
                    sigma_fact *= kwargs.get('sigma_factor_decay', 1)
                loop_kwargs['sigma_factor'] = sigma_fact
                    
                maxed_out_mbb_batches += 1
                t1 = time() #the loss is called in the optimizer.step 
                _,a,b,sigma,stop_dict = self.optimizer.step(batch_loss, **loop_kwargs)
                self.doc_dict['time'].append(time()-t1)
                if b: 
                    maxed_out_mbb_batches  = 0
                if maxed_out_mbb_batches > 100:
                    print('MBB sampling is not convergent, reinitializing the chain')
                    self.optimizer.start = True #This is a hot fix to not get the optimizer stuck to often

                if bern_sampled_data:
                    if old_correction is None:
                        old_correction = (torch.log(self.data['train'].sampler.p[0])/self.kwargs['temperature'])*len(y)
                    new_correction = (torch.log(self.data['train'].sampler.p[0])/self.kwargs['temperature'])*len(y)
                    if extended_doc_dict:
                        self.doc_dict['correction_diff'].append(self.kwargs['temperature']*(old_correction-new_correction).detach().cpu().item())
                    old_correction = new_correction
                    if isinstance(self.loss, L2_Bern_loss_corrected) and extended_doc_dict:
                        if isinstance(self.loss.corr_factor, torch.Tensor):
                            self.doc_dict['correction_factor'].append(dc(self.loss.corr_factor).detach().cpu().item())
                        else:
                            self.doc_dict['correction_factor'].append(self.loss.corr_factor)
                    if b:
                        self.doc_dict['batch_size_acc'].append(len(y))
                        self.last_accepted_batch = data_ep
                self.doc_dict['acceptance_prob'].append(a.detach().cpu().item())
                self.doc_dict['accepted'].append(b)
                if extended_doc_dict:
                    self.doc_dict['sigma'].append(sigma.detach().cpu().item())
                    self.doc_dict['step_over_sigma'].append(stop_dict['old_new_sqe'].detach().cpu().item()/sigma.detach().cpu().item())
                    self.doc_dict['steplenght'].append(torch.sqrt(torch.sum((old_state - torch.nn.utils.parameters_to_vector(self.model.parameters()))**2)).detach().cpu().item())   
                    self.doc_dict['GPU_allocated'].append(torch.cuda.memory_allocated(self.device))
                    self.doc_dict['memory_allocated'].append(virtual_memory()[2])
                for key in stop_dict.keys():
                    try:
                        self.doc_dict[key].append(stop_dict[key].detach().cpu().item())
                    except:
                        self.doc_dict[key].append(stop_dict[key])

                if self.scheduler is not None:
                    lr = self.scheduler.get_last_lr()[0]
                    #assert lr == self.optimizer.print_lr()

                    self.doc_dict['lr'].append(lr)
                    if lr >= kwargs.get('min_lr', 1e-10):
                        #if b:
                        self.scheduler.step()
                        if kwargs['sigma'] != 'dynamic' and kwargs.get('gamma_sigma_decay') != 'constant':
                            kwargs['gamma_sigma_decay'] =  self.scheduler.get_last_lr()[0]/lr if lr!=0 else 1
                    else:
                        kwargs['gamma_sigma_decay'] = None
                
                self.doc_dict['steps'] +=1
                if 'val' in self.data.keys():
                    val_losses = []
                    for d in self.data['val']:
                        val_loss = self.loss_val(self.model(d[0].detach().to(self.model.device).reshape(-1,1).float()), 
                                                d[1].detach().to(self.model.device).reshape(-1,1).float())
                        val_losses.append(val_loss)
                    self.doc_dict['val_loss'].append(torch.sum(torch.Tensor(val_losses)).detach().cpu().item())
        self.model.eval()
        print("    ", end='\r')

    def warm_up(self, epochs, **kwargs):
        self.run(epochs, True, **kwargs)

    def get_samples(self, x_eval, n_samples, gap_length, save_path = None, return_samples=True, **kwargs):
        kwargs_in = self.kwargs if kwargs == {} else kwargs

        kwargs_in['gamma_sigma_decay'] = None
        if self.kwargs.get('gamma_scheduler',1) != 1:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.base_opt, 1)
        #self.scheduler = None

        if return_samples: 
            samples = []
            samples.append(self.model(x_eval).unsqueeze(0).detach().cpu())

        if save_path is not None:
            self.save(save_path+'/sample_0/')
        self.doc_dict['steps_sampling'].append(dc(self.doc_dict['steps']))
        for i in range(n_samples-1):
            for l in [self.loss, self.loss_full]:
                if isinstance(l, L2_Bern_loss_corrected):
                    l.set_corr_factor(self.model(self.data['train'].dataset.x_train.reshape(-1,1).to(device = self.device, dtype = torch.float32)), 
                                      self.data['train'].dataset.y_train.reshape(-1,1).to(device = self.device, dtype = torch.float32))
            self.run(gap_length, False, **kwargs_in)
            if save_path is not None:
                self.save(save_path+f'/sample_{i+1}/')
            if return_samples:
                samples.append(self.model(x_eval).unsqueeze(0).detach().cpu())
            self.doc_dict['steps_sampling'].append(dc(self.doc_dict['steps']))
        
        if return_samples:
            return torch.cat(samples, 0)

    def get_mean_and_std(self, x_eval, n_samples, gap_length, **kwargs):
        samples = self.get_samples(x_eval, n_samples, gap_length, **kwargs)
        return samples.mean(0), samples.std(0)