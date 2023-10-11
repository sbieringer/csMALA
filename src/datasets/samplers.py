import torch

class Bernoulli_batch_sampler(torch.utils.data.Sampler):
    """
    Implements a batch_sampler that give indices based on a Bernoulli distribution (drawing with probability p).
    Batchsizes thus differ with mean p*len(data)
    If passed as batch_sampler, only one optimization per epoch is performed
    """
    def __init__(self, p, n_data):
        self.p = p*torch.ones(n_data)
        self.n_data = n_data
        self.indices = torch.arange(n_data)

    def __iter__(self):
        #bern = torch.zeros_like(self.p)
        #while bern.sum() == 0:
        bern = torch.bernoulli(self.p).to(bool)
        yield torch.masked_select(self.indices, bern)

    def __len__(self):
        return self.n_data
    

class naive_Bernoulli_Dataloader():
    '''
    Naive Dataloader equivalent that allows the use of batchsize 0
    '''
    def __init__(self, data, p):
        self.dataset = data
        self.batch_size = int(p * len(data))
        self.sampler = Bernoulli_batch_sampler(p, len(data))
        self._index = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index == 0:
            self._index += 1
            return self.dataset[next(iter(self.sampler))]
        else:
            self._index = 0
            raise StopIteration