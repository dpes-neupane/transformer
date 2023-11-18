import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.WQ = torch.nn.Parameter(torch.randn((dim, dim)))
        self.WK = torch.nn.Parameter(torch.randn((dim, dim)))
        self.WV = torch.nn.Parameter(torch.randn((dim, dim)))
        self.dim = dim
    def forward(self, emb):
        Q = torch.matmul(emb, self.WQ)
        K = torch.matmul(emb, self.WK)
        V = torch.matmul(emb,self.WV)
        dots = torch.matmul(Q, torch.t(K)) /  torch.sqrt(torch.tensor(self.dim))
        softmax  = torch.nn.functional.softmax(dots , dim=-1)
        attention = torch.matmul(softmax, V)
        return attention, softmax
