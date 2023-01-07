import numpy as np
import torch

class RunningMeanStd:
    def __init__(self,shape):
        self.n = 0
        self.mean = torch.zeros(shape)
        self.S = torch.zeros(shape)
        self.std = torch.sqrt(self.S)

    def update(self, x):
        x = torch.array(x).mean()
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n )
    
    def reset():
        self.mean=0
        self.n=0
        self.S=0

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def normalization(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x
    
    def reset():
        self.running_ms.reset()