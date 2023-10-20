import torch
import numpy as np
from time import time

import sys
sys.path.append("../")

from uncertimators.base_uncertimator import uncertimator

class non_bayes_uncertimator(uncertimator):
    '''
    Class to run classical training
    '''

    def __init__(self, model, loss, data, device):
        '''
        model: model_util.RegressionNet with VBLinear layers
        '''
        super().__init__(model, loss, data, device)
        dict_tmp = {'time_fit': []}
        self.doc_dict.update(dict_tmp)

    def train(self, epochs, start, **kwargs):
        '''
        Loop that performs the variational inference of self.loss
        
        args:   epochs - INT: Number of repetitions over the train_dataloader
                start - bool: Start new optimization or reuse old optimzers & schedulers

        kwargs: opt - torch.optim.Optimizer, 'SGD', else: defines Optimizer, defaults to ADAM

        '''

        self.kwargs = kwargs
        self.model.train()

        if start:
            if not isinstance(kwargs.get('opt', 'SGD'), torch.optim.Optimizer):
                if kwargs.get('opt', 'SGD') == 'SGD':
                    self.optimizer = torch.optim.SGD(self.model.parameters(), lr = kwargs.get('lr_start', 1e-2))
                else:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr = kwargs.get('lr_start', 1e-2), betas = kwargs.get('betas', (0.9,0.999)))
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, kwargs.get('gamma_scheduler',1))

        self.model.train()
        verbose = kwargs.get('verbose', False)
        
        epochs = range(epochs) if isinstance(epochs, int) else range(*epochs)
        for epoch in epochs:
            if verbose:
                print(epoch, ' epochs', end='\r')

            for data_ep in self.data['train']: #SLOW
                self.optimizer.zero_grad()
                          
                t1 = time()
            
                self.optimizer.zero_grad()
                x = data_ep[0].reshape(-1,1).to(device = self.device, dtype = torch.float32)
                y = data_ep[1].reshape(-1,1).to(device = self.device, dtype = torch.float32)

                training_obj = self.loss(self.model(x), y)

                if training_obj < -10000 or training_obj > 10000 and epoch != epochs[0]:
                    print(f'training obj {training_obj} overflow')

                self.doc_dict['train_loss'].append(training_obj.detach().cpu().numpy())

                training_obj.backward()
                self.optimizer.step()
                t2 = time()
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                self.doc_dict['time_fit'].append(t2-t1)

                if 'val' in self.data.keys():
                    val_losses = []
                    for d in self.data['val']:
                        val_loss = self.loss(self.model(d[0].detach().to(self.model.device).reshape(-1,1).float()), 
                                                d[1].detach().to(self.model.device).reshape(-1,1).float())
                        val_losses.append(val_loss)
                    self.doc_dict['val_loss'].append(torch.sum(torch.Tensor(val_losses)).detach().cpu().numpy())
        self.model.eval()

    def warm_up(self, epochs, **kwargs):
        self.train(epochs, True, **kwargs)

    def run(self, epochs, **kwargs):
        self.train(epochs, False, **kwargs)

    def get_samples(self, x_eval):
        return np.array([self.model(x_eval.reshape(-1,1)).detach().cpu().numpy().reshape(len(x_eval), self.model.n_dims_out)])

    def get_mean_and_std(self, x_eval):
        y_fit = self.get_samples(x_eval)
        return y_fit, np.zeros_like(y_fit)
