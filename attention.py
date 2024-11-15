import torch 
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_head:int, d_emb:int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()

        self.in_proj = nn.Linear(d_emb, 3 * d_emb, bias =in_proj_bias)
        self.out_proj = nn.Linear(d_emb, d_emb, bias = out_proj_bias)

        self.n_heads = n_head 
        self.d_heads = d_emb // n_head  # value of dk


    def forward(self, x: torch.Tensor, casual_mask = False):
        # x: (batch_size, seq_len, d_emb)

        input_shape = x.shape

        batch_size, sequence_length, d_embd = input_shape

        intermim_dim = (batch_size, sequence_length, self.n_heads, self.d_heads)

        #(batch_size, seq_len, d_emb) -> (batch_size, sequence_length, d_emb*3) -> 3 tensor of (batch_size, sequence_length, d_emb)
        q,k,v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, sequence_length, d_emb) -> (batch_size, sequence_length, n_heads, d_heads) -> (batch_size, n_heads,sequence_length, d_heads)
        q = q.view(intermim_dim).transpose(1,2)
        k = k.view(intermim_dim).transpose(1,2)
        v = v.view(intermim_dim).transpose(1,2)

        # (batch_size, n_heads, sequence_length, sequence_length)
        weight  = q @ k.transpose(-1,-2)

        if casual_mask:
            mask =  torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_heads)

        weight = F.softmax(weight, dim=-1)

        # (batch_size, n_heads, sequence_length, sequence_length) @ (batch_size, n_heads, sequence_length, d_heads) -> (batch_size, n_heads, sequence_length, d_heads)
        output = weight @ v

        # (batch_size, n_heads, sequence_length, d_heads) -> (batch_size, sequence_length, n_heads,d_heads)
        output = output.transpose(1,2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (BatchSize, sequence_length, d_emb)
        return output