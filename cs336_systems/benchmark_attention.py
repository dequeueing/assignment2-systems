import statistics
import timeit

from cs336_basics.model import RotaryEmbedding, Linear
from cs336_basics.nn_utils import softmax

import einx
import math

import torch
import torch.nn as nn
from torch import Tensor

from jaxtyping import Float, Int, Bool
from einops import rearrange, einsum



def scaled_dot_product_attention(
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
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")




class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention

    This function implements section 3.2.2 of the Transformer paper. In particular,
    given an input tensor of shape `(batch_size, sequence_length, d_model)`, we project
    it to create queries, keys, and values, and then perform causal multi-headed attention with
    those queries, keys, and values.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        positional_encoder: RotaryEmbedding
            The RoPE module to use.

    Returns:
        Tensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)

        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

        self.positional_encoder = positional_encoder  # RoPE

    def forward(self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None) -> Float[Tensor, " ... seq d_v"]:
        """
        Args:
            x: The input to perform multi-headed self-attention on.
            positional_ids: The positional indices along the sequence dimension of the input embeddings.

        Returns:
            Self-attention outputs.
        """
        *b, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Take apart each head from the embedding dimension of Q, K, V to shape (..., num_heads, seq_len, d_k).
        Q, K, V = (
            rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for X in (Q, K, V)
        )  # fmt: skip

        if token_positions is None:
            token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))

        # Duplicate token positions for each head
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        Q = self.positional_encoder(Q, token_positions)
        K = self.positional_encoder(K, token_positions)

        # Construct causal mask
        seq = torch.arange(sequence_length, device=x.device)
        qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
        kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
        causal_mask = qi >= kj  # (query, key)

        # Shape: (..., num_heads, sequence_length, d_k)
        attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)

        # Concatenate the attention output from all heads.
        # (..., sequence_length, num_heads * d_v).
        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

        # Apply the output projection
        output = self.output_proj(attn_output)
        return output
    
    
class CausalNoHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention

    This function implements section 3.2.2 of the Transformer paper. In particular,
    given an input tensor of shape `(batch_size, sequence_length, d_model)`, we project
    it to create queries, keys, and values, and then perform causal multi-headed attention with
    those queries, keys, and values.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        positional_encoder: RotaryEmbedding
            The RoPE module to use.

    Returns:
        Tensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        positional_encoder: RotaryEmbedding,
    ):
        super().__init__()
        self.d_model = d_model

        self.d_k = self.d_model
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.d_k)
        self.k_proj = Linear(self.d_model, self.d_k)
        self.v_proj = Linear(self.d_model, self.d_v)

        self.output_proj = Linear(self.d_v, self.d_model)

        self.positional_encoder = positional_encoder  # RoPE

    def forward(self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None) -> Float[Tensor, " ... seq d_v"]:
        """
        Args:
            x: The input to perform multi-headed self-attention on.
            positional_ids: The positional indices along the sequence dimension of the input embeddings.

        Returns:
            Self-attention outputs.
        """
        *b, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # debug: print the shape of Q K V
        # ([8, 256, 64])
        # batch size, sequence length, dimension
        # print(f"the shape of Q: {Q.shape}")
        # print(f"the shape of K: {K.shape}")
        # print(f"the shape of v: {V.shape}")

        # Skip spliting into many heads
        # Q, K, V all have shape (batch_size, seq, d)
        # Take apart each head from the embedding dimension of Q, K, V to shape (..., num_heads, seq_len, d_k).
        # Q, K, V = (
        #     rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        #     for X in (Q, K, V)
        # )  # fmt: skip

        if token_positions is None:
            token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))
            
        # debug: print token positions
        # print(f"the length of b: {len(b)}")  # 1
        # print(f"the value of b: {b}")        # 8
        # print(f"token_positions: {token_positions}")  # from 0 to sequence length

        # Duplicate token positions for each head
        # token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        Q = self.positional_encoder(Q, token_positions)
        K = self.positional_encoder(K, token_positions)

        # Construct causal mask
        seq = torch.arange(sequence_length, device=x.device)
        # qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
        # kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
        # fix the number of head dimension
        qi = einx.rearrange('query -> b... query 1', seq, b=[1] * len(b))
        kj = einx.rearrange('key   -> b... 1     key', seq, b=[1] * len(b))
        causal_mask = qi >= kj  # (query, key)
        
        # debug: print the mask 
        # print(f"qi: \n{qi}")
        # print(f"kj: \n{kj}")
        # print(f"causal mask: \n{causal_mask}")

        # Shape: (..., num_heads, sequence_length, d_k)
        attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)

        # No reconstruct needed since there are aren't many heads
        # Concatenate the attention output from all heads.
        # (..., sequence_length, num_heads * d_v).
        # attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

        # Apply the output projection
        output = self.output_proj(attn_output)
        return output



if __name__ == '__main__':
    # hyperparameters
    batch_size = 8
    sequence_lengths = [256,1024,4096,8192,16384]
    d_models = [16,32,64,128]
    device = "cuda"
    
    # iterate thru all combinations
    for sequence_length in sequence_lengths:
        for d_model in d_models:
            print(f"\nCurrent sequence length: {sequence_length}, d_model: {d_model}")
            
            try:
                # input
                x = torch.randn(batch_size, sequence_length, d_model).to(device)
                
                # model
                rope = RotaryEmbedding(context_length=sequence_length, dim=d_model).to(device)
                attention = CausalNoHeadSelfAttention(d_model=d_model, positional_encoder=rope).to(device)
                
                # Use no_grad for inference to save memory
                with torch.no_grad():
                    # warmup runs
                    for _ in range(5):
                        _ = attention(x)
                        torch.cuda.synchronize()
                    
                    # Print memory usage after warmup (shows actual peak usage)
                    # Note: this does not work, since it is only a timestamp.
                    # But the memory reaches its peak during attention forwarding
                    # if device == "cuda":
                    #     allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    #     reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                    #     max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
                    #     print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Peak: {max_allocated:.2f} GB")
                    
                    # timing pass
                    time_list = []
                    for _ in range(100):
                        # start timer
                        start = timeit.default_timer()
                        
                        # forward pass
                        _ = attention(x)
                        torch.cuda.synchronize()
                        
                        # stop timer
                        end = timeit.default_timer()
                        
                        # record time 
                        time_list.append(end - start)
                        
                # get the statistics of time used: max, min, mean, std
                mean_val = statistics.mean(time_list)
                variance_val = statistics.stdev(time_list)
                max_val = max(time_list)
                min_val = min(time_list)
                print(f"最大值: {max_val:.6f} 秒, 最小值: {min_val:.6f} 秒, 平均值: {mean_val:.6f} 秒, 标准差: {variance_val:.6f}")
                
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️  CUDA OOM: Skipping (seq_len={sequence_length}, d_model={d_model})")
                # Clear cache to recover from OOM
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"❌ Error: {type(e).__name__}: {e}")
                continue
            finally:
                # the memory will be automatically released,
                # but we explicit clean up after each iteration
                if 'x' in locals():
                    del x
                if 'rope' in locals():
                    del rope
                if 'attention' in locals():
                    del attention
                    
                # Note: locals() returns a dictionary of all local variables in the current scope
                # This is to ensure that the varialbes we delete are actually in the scope
                torch.cuda.empty_cache()