import torch 
from torch import nn
class Multihead(nn.Module):
    def __init__(self, input_dim, emb_dim, heads, decoder=False):
        super().__init__()
        assert emb_dim % heads == 0, "embedding dimension should be divisible by no. of heads"
        if not decoder:
            self.qkv = nn.Linear(input_dim, 3*emb_dim)
        else:
            self.qkv = nn.Linear(input_dim, emb_dim)
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.emb_dim = emb_dim
        self.heads = heads
        self.decoder=decoder
        self._reset_params()
        
    def _reset_params(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        self.qkv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)
        
    def forward(self, x, mask=None, masked=False, ret_att=False):
        batch, seq_length, _ = x.size()
        qkv = self.qkv(x)
        if not self.decoder:
            qkv = qkv.reshape(batch, seq_length, 3, self.heads, self.emb_dim//self.heads) #(batch, seq_length, qkv, heads, dim_k)
            qkv = qkv.permute(0, 3, 2, 1, -1) #(batch, heads, qkv, seq_length, dim_k)
            q, k, v = torch.chunk(qkv, dim=2, chunks=3)
            softmax, values =  self.maskedSelfAttention(q, k, v, mask) 
        else:
            seq_length /= 3
            qkv = qkv.reshape(batch, 3, seq_length, self.heads, self.emb_dim//self.heads)
            qkv = qkv.permute(0, 1, 3, 2, -1)
            q, k, v = qkv.chunk(3, dim=1)
            if masked:
                m = torch.ones(seq_length, seq_length)
                torch.tril(m, diagonal=0, out=m)
                mask += m
            softmax, values = self.maskedSelfAttention(q, k, v, mask)
        #may need to change this too
        values = values.permute(0, 3, 2, 1, -1) #(batch, seq_length, 1, heads, dim_k)
        values = values.reshape(batch, seq_length, self.emb_dim)
        vals = self.linear(values)
        if ret_att:
            return softmax, vals
        return vals
    
    def selfAttention(self, q, k, v):
        dot = torch.matmul(q, k.transpose(-1, -2))
        scaled_dots = dot / torch.sqrt(torch.tensor(q.size()[-1]))
        softmax = torch.nn.functional.softmax(scaled_dots, dim=-1)
        values = torch.matmul(softmax, v)
        return softmax, values

    def maskedSelfAttention(self, q, k, v, mask=None):
        dot = torch.matmul(q, k.transpose(-1, -2))
        scaled_dots = dot / torch.sqrt(torch.tensor(q.size()[-1]))
        if mask is not None:
            mask = torch.unsqueeze(mask, 0)
            scaled_dots = scaled_dots.masked_fill(mask==0, -9e15)
        softmax = torch.nn.functional.softmax(scaled_dots, dim=-1)
        values = torch.matmul(softmax, v)
        return softmax, values
    
    
    
            
