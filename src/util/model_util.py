import torch
import torch.nn as nn

class RegressionNet(torch.nn.Module):
    def __init__(self, dim_in=1, dim_out=2, ndf=256, dropout=0, activation=nn.LeakyReLU, layers=4*[nn.Linear], layer_kwargs=4*[{}], device=None):
        """
        Simple fully convolutional NN for the regression task. Last layer is accesible via self.last_layer

        kwargs: dim_int - INT: input dimension, default = 1
                dim_out - INT: output dimension, default = 2
                ndf - INT: dimension of hidden layers, default = 256
                dropout - FLOAT: amount of dropout to use, default = 0
                activation - torch.Module: activation function to use inbetween layers, default = nn.LeakyReLU
                layers - list: list of layers to use, len(layers) >=2
                layer_kwargs - list: kwargs passed to every layer on construction
        """
        super(RegressionNet, self).__init__()
        self.ndf=ndf   
        self.n_dims = dim_in
        self.dropoutChance = dropout
        self.n_dims_out = dim_out
        self.device = device
        
        self.last_layer = layers[-1](ndf,dim_out, **layer_kwargs[-1])
    
        architecture = [layers[0](dim_in, ndf, **layer_kwargs[0]),
            activation(inplace=True),
            nn.Dropout(p=self.dropoutChance)]

        for i in range(len(layers)-2):
            architecture.append(layers[i+1](ndf,ndf, **layer_kwargs[i]))
            architecture.append(activation(inplace=True))
            architecture.append(nn.Dropout(p=self.dropoutChance))
                
        if self.n_dims_out == 2:
            self.sp = torch.nn.Softplus()
            
        self.hidden_layers = nn.Sequential(*architecture)
        #self.main = nn.Sequential(self.hidden_layers, self.last_layer)

    def forward(self, x):
        """
        Performs a forward pass of the model.
        args: x - torch.Tensor: x.shape = (batchsize, model.n_dims), input tensor
        """
        output = self.last_layer(self.hidden_layers(x))
        if self.n_dims_out == 2:
            output = torch.cat((output[:,0].view(-1,1), self.sp(output[:,1]).view(-1,1)+1e-3), 1)
        return output
    
    def hidden_forward(self, x):
        """
        Performs a forward pass of the model, without the last layer for calculating the likelihood distribution when using NCP.
        args: x - torch.Tensor: x.shape = (batchsize, model.n_dims), input tensor
        """
        output = self.hidden_layers(x)
        return output

