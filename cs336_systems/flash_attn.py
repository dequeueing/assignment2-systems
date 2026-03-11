import argparse
import math

import triton
import triton.language as tl

import torch
import torch.nn as nn

import numpy as np

from einops import rearrange 

def flash_backward_fn(grad_out, l, q, k, v, o, is_causal):
    # get the shapes 
    n_q = q.shape[1]
    n_k = k.shape[1]
    d_model = q.shape[-1]
    
    # compute D
    D = torch.sum(o * grad_out, dim=-1)   # (batch_size, n_q)
    
    # get s
    s = torch.einsum("... q d, ... k d -> ... q k", q, k)
    s /= math.sqrt(d_model)   # (batch_size, n_q, n_k)
    
    # 👑 Causal Mask
    if is_causal:
        q_idx = torch.arange(n_q, device=q.device).unsqueeze(1)
        k_idx = torch.arange(n_k, device=k.device).unsqueeze(0)
        mask = q_idx >= k_idx
        s = s.masked_fill(~mask, float('-inf'))
    
    # get pij 
    p = torch.exp(s - l.unsqueeze(-1))  
    
    # get dV, dP
    dV = torch.einsum('... q k, ... q d -> ... k d', p, grad_out)   
    dP = torch.einsum('... q d, ... k d -> ... q k', grad_out, v)   
    
    # get dS 
    dS = p * (dP - D.unsqueeze(-1))        
    
    # get dQ and DK
    dQ = torch.einsum('... q k, ... k d -> ... q d', dS, k) / math.sqrt(d_model)
    dK = torch.einsum('... q k, ... q d -> ... k d', dS, q) / math.sqrt(d_model)
    
    return dQ, dK, dV

# 🔥 核心：在这里全局编译一次，整个训练过程复用！
compiled_flash_backward = torch.compile(flash_backward_fn)


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
        
        # 1.5 Masking 
        if is_causal:
            q_arange = torch.arange(0, n_queries, device='cuda')  # (q, )
            k_arange = torch.arange(0, n_keys, device = 'cuda')     # (k, )
            mask = q_arange[..., None] >= k_arange[None, ...]
            s = torch.where(mask, s, float('-inf'))
        
        # 2. softmax for each row 
        p = torch.softmax(s, dim=-1)  # (batch_size, n_queries, n_keys)
        
        # 3. O = P V
        o = torch.einsum("... q k, ... k d -> ... q d", p, v) # (batch_size, n_queries, D)
        
        # 4. Li
        # s = torch.exp(s) 
        # s = torch.sum(s, dim=-1)
        # l = torch.log(s)   # (batch_size, n_queries, )
        # FIXME: use logsumexp, a safer low-level function
        l = torch.logsumexp(s, dim=-1)
        
        
        # save input in context
        ctx.save_for_backward(l, q, k, v, o)
        ctx.is_causal = is_causal
        
        return o
    
    @staticmethod
    def backward(ctx, grad_out):
        # load tensors from context 
        l, q, k, v, o = ctx.saved_tensors
        is_causal = ctx.is_causal
    
        layer_compiled = compiled_flash_backward
        dQ, dK, dV = layer_compiled(grad_out, l, q, k, v, o, is_causal)
        return dQ, dK, dV, None  # FIXME: the backward should return the same number of args

    
@triton.autotune(
    configs=[
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_warps=8),
        triton.Config({'Q_TILE_SIZE': 64,  'K_TILE_SIZE': 128}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64},  num_warps=4),
        triton.Config({'Q_TILE_SIZE': 64,  'K_TILE_SIZE': 64},  num_warps=4),
        triton.Config({'Q_TILE_SIZE': 32,  'K_TILE_SIZE': 32},  num_warps=2),
        triton.Config({'Q_TILE_SIZE': 16,  'K_TILE_SIZE': 16},  num_warps=1),
    ],
    key=['N_QUERIES', 'N_KEYS', 'D'],
)
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

        # validation check
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Expected CUDA tensors"
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        # prepare output buffers
        l = torch.empty((batch_size, n_queries,),  device=q.device)
        o = torch.empty((batch_size, n_queries, D), device=q.device)

        # grid uses meta dict so autotune-chosen Q_TILE_SIZE is accessible
        grid = lambda meta: (triton.cdiv(n_queries, meta['Q_TILE_SIZE']), batch_size)
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
            is_causal=is_causal,
        )

        # save in context
        ctx.save_for_backward(l, q, k, v, o)
        ctx.is_causal = is_causal
        
        # return result
        return o
    
    @staticmethod
    def backward(ctx, grad_out):
        # load tensors from context 
        l, q, k, v, o = ctx.saved_tensors
        is_causal = ctx.is_causal
    
        layer_compiled = compiled_flash_backward
        dQ, dK, dV = layer_compiled(grad_out, l, q, k, v, o, is_causal)
        return dQ, dK, dV, None  # FIXME: the backward should return the same number of args
    
    
def benchmark_attention(fwd_only: bool = False):
    batch_size = 1
    is_causal = True
    device = "cuda"

    # fwd_only mode enables longer sequences (no N² backward allocation)
    if fwd_only:
        seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
        d_models = [64, 128]
    else:
        seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        d_models = [16, 32, 64, 128]

    dtypes = [torch.float32, torch.bfloat16]
    impls = [FlashAttentionTriton, FlashAttentionPytorch]

    for seq_len in seq_lens:
        for d_model in d_models:
            for dtype in dtypes:
                for impl in impls:
                    tag = f"[{impl.__name__}] seq_len={seq_len:>7}, d_model={d_model:>3}, dtype={str(dtype).split('.')[-1]:>10}"
                    try:
                        if fwd_only:
                            q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
                            k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
                            v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)

                            def run_forward(_impl=impl, _q=q, _k=k, _v=v):
                                with torch.no_grad():
                                    return _impl.apply(_q, _k, _v, is_causal)

                            ms_fwd = triton.testing.do_bench(run_forward, warmup=25, rep=100)
                            print(f"{tag}  |  fwd={ms_fwd:.3f} ms")
                        else:
                            q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                            k = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
                            v = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

                            out = impl.apply(q, k, v, is_causal)
                            dout = torch.randn_like(out)

                            def run_forward(_impl=impl, _q=q, _k=k, _v=v):
                                return _impl.apply(_q, _k, _v, is_causal)

                            def run_backward(_out=out, _dout=dout):
                                _out.backward(_dout, retain_graph=True)

                            def run_e2e(_impl=impl, _q=q, _k=k, _v=v, _dout=dout):
                                _out = _impl.apply(_q, _k, _v, is_causal)
                                _out.backward(_dout)

                            ms_fwd = triton.testing.do_bench(run_forward, warmup=25, rep=100)
                            ms_bwd = triton.testing.do_bench(run_backward, warmup=25, rep=100)
                            ms_e2e = triton.testing.do_bench(run_e2e, warmup=25, rep=100)
                            print(f"{tag}  |  fwd={ms_fwd:.3f} ms  bwd={ms_bwd:.3f} ms  e2e={ms_e2e:.3f} ms")

                    except torch.cuda.OutOfMemoryError:
                        print(f"{tag}  |  OOM")
                    finally:
                        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flash Attention benchmark")
    parser.add_argument(
        "--fwd-only",
        action="store_true",
        help="Benchmark forward pass only (no backward). Enables longer sequence lengths.",
    )
    args = parser.parse_args()
    benchmark_attention(fwd_only=args.fwd_only)