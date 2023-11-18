import torch 
from torch import nn
class Multihead(nn.Module):
    def __init__(self, input_dim, emb_dim, heads):
        super().__init__()
        assert emb_dim % heads == 0, "embedding dimension should be divisible by no. of heads"
        self.qkv = nn.Linear(input_dim, 3*emb_dim)
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.emb_dim = emb_dim
        self.heads = heads
        self._reset_params()
    def _reset_params(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        self.qkv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)
        
    def forward(self, x, att=True):
        batch, seq_length, _ = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_length, 3, self.heads, self.emb_dim//self.heads) #(batch, seq_length, qkv, heads, dim_k)
        qkv = qkv.permute(0, 3, 2, 1, -1) #(batch, heads, qkv, seq_length, dim_k)
        q, k, v = torch.chunk(qkv, dim=2, chunks=3)
        softmax, values =  self.selfAttention(q, k, v)
        values = values.permute(0, 3, 2, 1, -1) #(batch, seq_length, 1, heads, dim_k)
        values = values.reshape(batch, seq_length, self.emb_dim)
        if att==True:
            return softmax, values
        else:
            return values

    def selfAttention(self, q, k, v):
        dot = torch.matmul(q, k.transpose(-1, -2))
        scaled_dots = dot / torch.sqrt(torch.tensor(q.size()[-1]))
        softmax = torch.nn.functional.softmax(scaled_dots, dim=-1)
        values = torch.matmul(softmax, v)
        return softmax, values


            
