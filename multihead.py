import torch 
from torch import nn, Tensor
from typing import Union, Tuple
class Multihead(nn.Module):
    '''
    Implementation of Multihead attention according the the paper "Attention is all you need".
    
    Args:
        input_dim: size of the input dimension 
        emb_dim: dimension of the embeddings inside the transformer
        heads: no of heads for multihead selfAttention calculation.
    
    Shape:
        - Input: (seq_length, input_dim) 
        - Output: (seq_length, emb_dim) | tuple
    '''
    def __init__(self, input_dim: int, emb_dim: int, heads: int) -> None:
        super().__init__()
        assert emb_dim % heads == 0, "embedding dimension should be divisible by no. of heads"
        self.qkv = nn.Linear(input_dim, 3*emb_dim)
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.emb_dim = emb_dim
        self.heads = heads
        self._reset_params()
        
    def _reset_params(self) -> None:
        nn.init.xavier_uniform_(self.qkv.weight)
        self.qkv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)
        
    def forward(self, x: Tensor, mask=None, ret_att=False) -> Union[tuple, Tensor]:
        batch, seq_length, _ = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_length, 3, self.heads, self.emb_dim//self.heads) #(batch, seq_length, qkv, heads, dim_k)
        qkv = qkv.permute(0, 3, 2, 1, -1) #(batch, heads, qkv, seq_length, dim_k)
        q, k, v = torch.chunk(qkv, dim=2, chunks=3)
        softmax, values = self.maskedSelfAttention(q, k, v, mask)
        values = values.permute(0, 3, 2, 1, -1) #(batch, seq_length, 1, heads, dim_k)
        values = values.reshape(batch, seq_length, self.emb_dim)
        vals = self.linear(values)
        if ret_att:
            return softmax, vals
        return vals
    
    def selfAttention(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        dot = torch.matmul(q, k.transpose(-1, -2))
        scaled_dots = dot / torch.sqrt(torch.tensor(q.size()[-1]))
        softmax = torch.nn.functional.softmax(scaled_dots, dim=-1)
        values = torch.matmul(softmax, v)
        return softmax, values

    def maskedSelfAttention(self, q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tuple[Tensor, Tensor]:
        dot = torch.matmul(q, k.transpose(-1, -2))
        scaled_dots = dot / torch.sqrt(torch.tensor(q.size()[-1]))
        if mask is not None:
            mask = torch.unsqueeze(mask, 0)
            mask = torch.unsqueeze(mask, 2)
            scaled_dots = scaled_dots.masked_fill(mask==0, -9e15)
        softmax = torch.nn.functional.softmax(scaled_dots, dim=-1)
        values = torch.matmul(softmax, v)
        return softmax, values
    
    
    
class DecoderMultihead(Multihead):
    ''' 
    Class for the applying the multihead attention for the decoder of the transformer

    The input for this multihead takes the output of the encoder. 
    
    Args:
        input_dim: size of the input dimension 
        emb_dim: dimension of the embeddings inside the transformer
        heads: no of heads for multihead selfAttention calculation
    
    Shape:
        - Input: (seq_length, input_dim) and the output of the encoder (seq_length, emb_dim)
        - Output: (seq_length, emb_dim)
    '''
    def __init__(self, input_dim: int, emb_dim: int, heads: int) -> None:
        super().__init__(input_dim, emb_dim, heads)
        self.kv = nn.Linear(input_dim, 2*emb_dim)
        self.q = nn.Linear(input_dim, emb_dim)
        
    def forward(self, kv: Tensor, x: Tensor, mask=None, ret_att=False) -> Union[tuple, Tensor]:
        batch_x, seq_length_x, _ = x.size()
        batch_kv, seq_length_kv, _ = kv.size()
        kv = self.kv(kv)
        kv = kv.reshape(batch_kv, seq_length_kv, 2, self.heads, self.emb_dim//self.heads)
        kv = kv.permute(0, 3, 2, 1, -1) # batch, heads, kv, seq_length, dk 
        k, v = torch.chunk(kv, chunks=2, dim=2)
        q = self.q(x)
        q = q.reshape(batch_x, self.heads, 1, seq_length_x, self.emb_dim//self.heads)
        softmax, values =  super().maskedSelfAttention(q, k, v, mask) 
        values = values.permute(0, 3, 2, 1, -1) #(batch, seq_length, 1, heads, dim_k)
        values = values.reshape(batch_kv, seq_length_kv, self.emb_dim)
        vals = self.linear(values)
        if ret_att:
            return softmax, vals
        return vals
