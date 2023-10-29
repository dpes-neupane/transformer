import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.WQ = torch.nn.Parameter(torch.randn(()))
        self.WK = torch.nn.Parammeter(torch.randn(()))
        self.WV = torch.nn.Parameter(torch.randn(()))
        self.dim = dim
    def forward(self, emb):
        Q = torch.matmul(emb, self.WQ)
        K = torch.matmul(emb, self.WK)
        V = torch.matmul(emb,self.WV)
        return torch.matmul(torch.nn.functional.softmax(torch.matmul(Q, torch.t(K))/self.dim)), V)
