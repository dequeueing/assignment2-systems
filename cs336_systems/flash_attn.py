import math

import triton
import triton.language as tl

import torch
import torch.nn as nn

import numpy as np

from einops import rearrange 
    
class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        # get the shapes
        n_queries = q.shape[1]
        n_keys = k.shape[1]
        D = q.shape[-1]
        
        # 1. S @ K^T  / sqrt(D)
        s = torch.einsum("... q d, ...k d -> ... q k", q, k)
        s /= math.sqrt(D)
        
        # 2. softmax for each row 
        p = torch.softmax(s, dim=-1)  # (batch_size, n_queries, n_keys)
        
        # 3. O = P V
        o = torch.einsum("... q k, ... k d -> ... q d", p, v)
        
        # 4. Li
        s = torch.exp(s) 
        s = torch.sum(s, dim=-1)
        l = torch.log(s)   # (batch_size, n_queries, )
        
        # save input in context
        ctx.save_for_backward(l, q, k, v, o)
        
        return o
    
    

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError
    
    
if __name__ == '__main__':
    pass