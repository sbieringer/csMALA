import torch
import pickle
from copy import deepcopy as dc
import os

def add_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# class ensemble:
#     '''
#     Calculate an Ensemble of uncertimators
#     '''
#     def __init__(self, models, loss, data, device, experiment = None):
#         '''
#         models - list of model_util.RegressionNet: model to evaluate
#         data - dict: - keys "train", "val" and "test" and and values torch.utils.data.DataLoader
#         loss - callable: Loss function instance which takes the arguments: y_prediction and y_target
#         '''
#         self.loss = loss

#         self.data = data
#         self.n_data_total = sum([len(data[key].data) for key in data.keys()])
#         self.n_data = {key: len(data[key]) for key in data.keys()}
#         self.data_total = torch.concat(data.values().data, 0)
#         assert len(self.data_total) == self.n_data_total

#         self.models = models
#         self.n_ensemble = len(models)

#         self.device = device
#         self.experiment = experiment

#         perms_tmp = [torch.randperm(self.n_data_total).to(self.data.device) for _ in range(self.n_ensemble)]

#         self.idcs = []
#         for perm in perms_tmp:
#             n_tmp = 0
#             perms_dict_tmp = {}
#             for key in data.keys():
#                 n_tmp_next = n_tmp + self.n_data[key]
#                 perms_dict_tmp[key] = perm[n_tmp:n_tmp_next]
#                 n_tmp = n_tmp_next
#             self.idcs.append(perms_dict_tmp)

#         del perms_tmp, data_tmp

#     def __getitem__(self, i):
#         '''
#         only initialize the uncertimators when called to save memory
#         '''
#         for key in self.data:
#             self.data[key].data = self.data_total[self.idcs[i][key]]
#         return uncertimator


class uncertimator:
    '''
    Baseclass for all uncertainty estimation methods
    '''

    def __init__(self, model, loss, data, device, experiment = None):
        '''
        Baseclass for all uncertainty estimation methods
        model - model_util.RegressionNet: model to evaluate
        data - dict: - keys "train", "val" and "test" and and values torch.utils.data.DataLoader
        loss - callable: Loss function instance which takes the arguments: y_prediction and y_target
        '''
        
        self.loss = loss
        self.data = data
        self.model = model
        self.device = device
        self.experiment = experiment

        self.doc_dict = {'time': [], 'steps': 0, 'steps_sampling': [], 'memory_allocated': [], 'GPU_allocated': []}
        for key in self.data.keys():
            self.doc_dict[f'{key}_loss'] = []

        self.model.to(self.device)

    def run(self):
        pass

    def train(self):
        pass
    
    def warm_up(self):
        pass

    def get_samples(self):
        pass

    def get_mean_and_std(self):
        samples = self.get_samples()
        return samples.mean(0), samples.std(0)
    
    #write basic saving and loading routines for model and doc dict 
    def save(self, path): 
        torch.save(self.model.state_dict(), add_path(path)+'/model.pt')
        with open(path+'/doc_dict.pkl', 'wb') as file:
            pickle.dump(self.doc_dict, file)


    def load(self, path):
        self.model.load_state_dict(torch.load(path+'/model.pt'))
        with open(path+'/doc_dict.pkl','rb') as file:
            self.doc_dict = pickle.load(file)

    # def SBC():
    #     '''
    #     does simulation based calibration (https://arxiv.org/pdf/1804.06788.pdf and https://github.com/stefanradev93/BayesFlow) -- FUTURE WORK
    #     '''