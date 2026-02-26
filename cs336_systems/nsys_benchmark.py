from __future__ import annotations

import torch
import torch.cuda.nvtx as nvtx  # 导入 NVTX 模块 [cite: 116]



import functools
import json
import logging
import math
import os
from einops import rearrange, einsum
import einx

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float, Bool, Int

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

def softmax(x, dim=-1):
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]
    exponentiated_rescaled_input = torch.exp(rescaled_input)
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)


# 使用装饰器标记整个函数 [cite: 118]
@nvtx.range("annotated_attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        result =  einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
        
    return result


# 在你的脚本中替换原始实现 [cite: 144-145]
import cs336_basics
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


# 自动检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


warmup = True
time_forward = True
time_backward = True

batch_size = 4
vocab_size = 10000
context_length = 256
seq_len = 100

warmup_iter = 5
benchmark_iter = 10

# 定义不同大小的模型配置
model_configs = {
    'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
    # 'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    # 'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    # '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32},
}

def run_benchmark(config):
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=10000.0
    ).to(device)
    
    optimizer = AdamW(
        params=model.parameters(),
        lr=1e-3,              
        betas=(0.9, 0.999),   
        eps=1e-8,             
        weight_decay=0.01,    
    )
    
    # ... 初始化模型和数据 ...
    data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    

    # --- 热身阶段 ---
    # 我们不给热身阶段加 NVTX 标记，或者给它一个不同的标记
    # 这样在 Nsight Systems 里就能轻松过滤掉这部分数据 
    for _ in range(5):
        logits = model(data)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # --- 正式测试阶段 ---
    # 使用 "benchmark_iteration" 包裹你真正想要分析的步骤
    with nvtx.range("benchmark_iteration"):
        for i in range(benchmark_iter):
            # 也可以针对 Forward/Backward/Optimizer 分别标记
            with nvtx.range(f"step_{i}"):
                with nvtx.range("forward"):
                    logits = model(data)
                
                with nvtx.range("backward"):
                    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
                    loss.backward()
                
                with nvtx.range("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()
            
            torch.cuda.synchronize()

if __name__ == '__main__':
    run_benchmark(model_configs['small'])




