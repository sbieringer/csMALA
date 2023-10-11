# variation on the regression taks from https://arxiv.org/pdf/1807.09289.pdf

import numpy as np
import torch

# def noise_std(inputs, noise_slope=0.2):
#     return np.maximum(1e-3, (inputs + 1) * noise_slope)

# class vargrad_dataset():
#     def __init__(self, noise_slope=0.2, frequency=25, amplitude=0.5, slope=0.5):
#         self.noise_slope = noise_slope
#         self.frequency = frequency
#         self.amplitude = amplitude
#         self.slope = slope
        
#     def __call__(self, inputs):
#         targets = self.amplitude * + np.sin(self.frequency * inputs)
#         targets += self.slope * inputs
#         return targets
    
#     def std_dev(self, inputs):
#         return noise_std(inputs, self.noise_slope)
        
#     def generate(self, length=1000, x_empty=0):
#         random = np.random.RandomState(0)
#         inputs = (1-np.random.rand(length)*(1-x_empty))*(2*np.random.randint(2, size=length)-1)
#         targets = self(inputs)+np.random.normal(0, self.std_dev(inputs))
        
#         return inputs, targets 
    
# even simpler dataset generation for 1d regresssion
import os

def add_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def generate_1d_maximilian(n):
    x = np.random.uniform(0,1, n)
    y = np.sin(2*np.pi*x**2)
    y_out = y + np.random.normal(0, 0.1, n)

    x_out = x.reshape(-1,1) 
    y_out = y_out.reshape(-1,1) 

    return np.concatenate([x_out,y_out],1)

def load_1d_maximilian(base_dir, n, *args):
    if not os.path.exists(base_dir + f'/agw_data/{n}/maximilian/data.npy'):
        add_path(base_dir + '/agw_data')
        data = generate_1d_maximilian(n)
        np.save(base_dir + f'/agw_data/{n}/maximilian/data.npy', data)

    else:
        data = np.load(base_dir + f'/agw_data/{n}/maximilian/data.npy')

    X, Y = data[:, 0].astype(np.float32), data[:, 1].astype(np.float32)
    Y = Y[:, None]

    return X[:, None], Y

def generate_1d(n, sigma_data = 0.1):
    n2 = n//2
    ns = np.random.permutation(np.array([n2,n-n2]))
    if sigma_data !=0:
        x1 = np.random.uniform(-0.8, -0.2, ns[0])
        x2 = np.random.uniform(0.2, 0.8, ns[1])
    else:
        x1 = np.linspace(-0.8, -0.2, ns[0])
        x2 = np.linspace(0.2, 0.8, ns[1])

    y1 = 1.5*(x1+0.5)**2
    y2 = 0.3*np.sin(x2*10-2)+0.5

    x_out = np.concatenate([x1,x2]).reshape(-1,1)
    y_out = np.concatenate([y1,y2])+ np.random.normal(0, sigma_data, n)
    y_out = y_out.reshape(-1,1) 

    return np.concatenate([x_out,y_out],1)

# def generate_1d(n, sigma_data = 0.1):
#     n3 = n//3
#     ns = np.random.permutation(np.array([n3,n3,n-2*n3]))
#     x1 = np.random.uniform(-1.4, -1, ns[0])
#     x2 = np.random.uniform(-0.25, 0.25, ns[1])
#     x3 = np.random.uniform(1, 1.4, ns[2])

#     y1 = -0.4*x1-2
#     y2 = 1.5*x2+0.5
#     y3 = 0.3*np.sin(x3*10-2)+0.5

#     x_out = np.concatenate([x1,x2,x3]).reshape(-1,1)
#     y_out = np.concatenate([y1,y2,y3])+ np.random.normal(0, sigma_data, n)
#     y_out = y_out.reshape(-1,1) 

#     return np.concatenate([x_out,y_out],1)

def load_1d(base_dir, n, sigma_data=0.1):
    if not os.path.exists(base_dir + f'/agw_data/{n}/sigma{sigma_data}/data.npy'):
        add_path(base_dir + f'/agw_data/{n}/sigma{sigma_data}/')
        data = generate_1d(n, sigma_data)
        np.save(base_dir + f'/agw_data/{n}/sigma{sigma_data}/data.npy', data)
    else:
        data = np.load(base_dir + f'/agw_data/{n}/sigma{sigma_data}/data.npy')

    X, Y = data[:, 0].astype(np.float32), data[:, 1].astype(np.float32)
    Y = Y[:, None]

    return X[:, None], Y

from torch.utils.data import Dataset

class RegressionData(Dataset):
    """Simple Regression Dataset"""

    def __init__(self, x_train, y_train, transform=None):
        """
        Args:
            path (string): Path to the Acoustic folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.x_train[idx], self.y_train[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

        