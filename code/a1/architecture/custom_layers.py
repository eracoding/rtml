import torch.nn as nn
import torch.nn.functional as F


__all__ = ["LocalResponseNormalize"]


class LocalResponseNormalize(nn.Module):
    __constants__ = ["size", "alpha", "beta", "k"]
    size: int
    alpha: float
    beta: float
    k: float
        
    def __init__(self, size, alpha=0.0001, beta=0.75, k=1.0):
        super(LocalResponseNormalize, self).__init__()

        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
    
    def forward(self, x):
        return F.local_response_norm(x, k=self.k, alpha=self.alpha, beta=self.beta, size=self.size)

