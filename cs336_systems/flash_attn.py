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
        """
        Shape of input tensors:
        q: (batch_size, n_queres, D)
        k: (batch_size, n_keys  , D)
        v: (batch_size, n_keys  , D)
        
        Shape of output tensors:
        O: (batch_size, n_queries, D)
        L: (batch_size, n_queries, )
        """
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
        o = torch.einsum("... q k, ... k d -> ... q d", p, v) # (batch_size, n_queries, D)
        
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
    
    
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # get the tile numbers 
    # FIXME: cdiv already has the right logic!s
    T_Q = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    T_K = tl.cdiv(N_KEYS, K_TILE_SIZE)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),   # Note: iterate K pointer at inner loop
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),   # Note: iterate V pointer at inner loop
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    # buffer 
    m = tl.full((Q_TILE_SIZE,), value=float('-inf'), dtype=tl.float32)
    s = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
    p = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    
    # q needs to be loaded once 
    q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    
    # The inner loop
    for key_tile_index in range(T_K):
        # load the tile contents
        k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Shape note:
        # q_i: (Q_TILE_SIZE, D)
        # k_j: (K_TILE_SIZE, D)
        # v_j: (K_TILE_SIZE, D)
        # s:   (Q_TILE_SIZE, K_TILE_SIZE)
        # m:   (Q_TILE_SIZE,)
        # l:   (Q_TILE_SIZE,)
        # p:   (Q_TILE_SIZE, K_TILE_SIZE)
        # o:   (Q_TILE_SIZE, D)
        
        # compute tile of pre-sofrmax attention scores S
        s = tl.dot(q_i, tl.trans(k_j))
        s *= scale
        
        # prepare mask if is_causal is True
        if is_causal:
            q_arange = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_arange = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_arange[:, None] >= k_arange[None, :]
            s = tl.where(mask, s, float('-inf'))
            
        
        # compute m_j
        m_j_minus1 = m
        m = tl.maximum(m_j_minus1, tl.max(s, axis = -1))
        
        # compute pij
        # FIXME: broadcast is m[:, None] not m[:][None]
        p = tl.exp(s - m[:, None])
        
        # computer lij
        l = tl.exp(m_j_minus1 - m) * l + tl.sum(p, axis=-1)
        
        # compute oij
        p_cast = p.to(v_j.dtype)
        o = tl.exp(m_j_minus1 - m)[:, None] * o + tl.dot(p_cast, v_j)
        
        # advance the pointers
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    o = o / l[:, None]
    l = m + tl.log(l)
    
    # write O and L
    tl.store(O_block_ptr, o.to(O_ptr.type.element_ty), boundary_check=(0,1))
    tl.store(L_block_ptr, l, boundary_check=(0,))
    
    
class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        # get the shapes
        batch_size = q.shape[0]
        n_queries = q.shape[1]
        n_keys = k.shape[1]
        D = q.shape[-1]        
        # define hyperparameters: tile size
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        
        # validation check 
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Expected CUDA tensors"
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Our pointer arithmetic will assume contiguous x"
        
        
        # prepare output buffers
        l = torch.empty((batch_size, n_queries, ),  device=q.device)
        o = torch.empty((batch_size, n_queries, D), device=q.device)
        
        
        grid = ((n_queries + Q_TILE_SIZE - 1) // Q_TILE_SIZE, batch_size)
        flash_fwd_kernel[grid](
            q, k, v, o, l, 
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            l.stride(0), l.stride(1),
            n_queries, n_keys,
            1 / math.sqrt(D),
            D,
            Q_TILE_SIZE, K_TILE_SIZE, is_causal
        )
        
        # save in context 
        ctx.save_for_backward(l, q, k, v, o)
        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE
        ctx.q_shape = q.shape
        ctx.k_shape = k.shape
        ctx.is_causal = is_causal
        
        # return result
        return o
    
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError

    
    
if __name__ == '__main__':
    pass