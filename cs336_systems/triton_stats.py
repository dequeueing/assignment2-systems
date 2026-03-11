import triton
import triton.language as tl

import torch
import torch.nn as nn

from einops import rearrange # Required for the rearrange call


@triton.jit
def compute_mean_and_variance(
    x_ptr,  # Input pointers
    mean_ptr, var_ptr, # Output pointer
    x_stride_row, x_stride_dim,  # Strides tell us how to move one element in each axis of a tensor
    mean_stride, var_stride,  # Likely 1
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,  # Tile shapes must be known at compile time
):
    row_tile_idx = tl.program_id(0)
    
    # make block pointers
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    mean_block_ptr = tl.make_block_ptr(
        mean_ptr,
        shape=(ROWS, ),
        strides=(mean_stride,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, ),
        block_shape=(ROWS_TILE_SIZE, ),
        order=(0,),
    )
    var_block_ptr = tl.make_block_ptr(
        var_ptr,
        shape=(ROWS, ),
        strides=(var_stride,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, ),
        block_shape=(ROWS_TILE_SIZE, ),
        order=(0,),
    )
    
    # init a buffer to write to
    mean = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    var = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    
    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        # read from memorys
        rows = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        
        # add to partial sum
        mean += tl.sum(rows, axis=1)
        
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))  # Move by D_TILE_SIZE in the last dimension
        
    # diviede by dimension
    mean = mean / D
    tl.store(mean_block_ptr, mean, boundary_check=(0))
    
    # re-iterate to find variance
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        # read from memorys
        rows = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        rows = (rows - mean[:, None]) * (rows - mean[:, None])
        
        # add to partial sum
        var += tl.sum(rows, axis=1)
        
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))  # Move by D_TILE_SIZE in the last dimension
    var = var / D
    tl.store(var_block_ptr, var, boundary_check=(0))
        

@triton.jit
def welford_algorithm(
    x_ptr,  # Input pointers
    mean_ptr, var_ptr, # Output pointer
    x_stride_row, x_stride_dim,  # Strides tell us how to move one element in each axis of a tensor
    mean_stride, var_stride,  # Likely 1
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,  # Tile shapes must be known at compile time
):
    row_tile_idx = tl.program_id(0)
    
    # make block pointers
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    mean_block_ptr = tl.make_block_ptr(
        mean_ptr,
        shape=(ROWS, ),
        strides=(mean_stride,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, ),
        block_shape=(ROWS_TILE_SIZE, ),
        order=(0,),
    )
    var_block_ptr = tl.make_block_ptr(
        var_ptr,
    shape=(ROWS, ),
        strides=(var_stride,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, ),
        block_shape=(ROWS_TILE_SIZE, ),
        order=(0,),
    )
    
    # init a buffer to write to
    mean = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    var = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    num_element_processed = 0          # total number elements processed so far
    num_each_tile = D_TILE_SIZE     # number of elements processed in each iteration
    
    for _ in range(tl.cdiv(D, D_TILE_SIZE)):
        # 1. read from memorys and check num_each_tile
        rows = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        if num_each_tile + num_element_processed > D:  # Note: the last tile, number is different
            num_each_tile = D - num_element_processed
        
        # 2. calculate the mean and M_2 of the new tiles
        # Note: since we have loaded the memory, this process does not consume memory bandwidth
        
        # 2.1 calculate mean 
        sum_rows = tl.sum(rows, axis=-1)
        mean_rows = sum_rows / num_each_tile

        # 2.2 calculate M_2 
        # Note: padding = zero. Be careful with the (0 - mean) terms! Therefore we use another ways
        sum_squared_rows = tl.sum(rows * rows, axis=-1)
        m2_rows = sum_squared_rows - (num_each_tile * mean_rows * mean_rows)
        
        # 3. update existing values
        tmp = mean_rows - mean
        mean = mean + tmp * num_each_tile / (num_element_processed + num_each_tile)
        var = var + m2_rows + tmp * tmp * num_element_processed * num_each_tile / (num_element_processed + num_each_tile)
        num_element_processed += num_each_tile
        
        # 4. move to the next tile
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))  # Move by D_TILE_SIZE in the last dimension
        
    # diviede by dimension to get the real variance
    var /= D
    
    # store to memory
    tl.store(mean_block_ptr, mean, boundary_check=(0))
    tl.store(var_block_ptr, var, boundary_check=(0))
    
    
def naive_torch(x: torch.Tensor):
    # shape: (rows, D)
    mean = torch.mean(x, dim=-1)
    var = torch.var(x, dim=-1, unbiased=False)
    return mean, var
    
naive_torch_compiled = torch.compile(naive_torch)


    
if __name__ == '__main__':
    ROWS, D = 4096, 4096  
    
    # init tensors
    x      = torch.randn(ROWS, D, device="cuda")
    mean   = torch.empty(ROWS,    device="cuda")
    var    = torch.empty(ROWS,    device="cuda")
    
    # compute tile sizes 
    ROWS_TILE_SIZE = 32
    D_TILE_SIZE    = 1024  # = 8

    def cdiv(a, b): return (a + b - 1) // b
    grid = (cdiv(ROWS, ROWS_TILE_SIZE),)  

    # compute_mean_and_variance[grid](
    #     x,        
    #     mean, var,
    #     x.stride(0), x.stride(1),  
    #     mean.stride(0),
    #     var.stride(0),
    #     ROWS=ROWS, D=D,
    #     ROWS_TILE_SIZE=ROWS_TILE_SIZE,
    #     D_TILE_SIZE=D_TILE_SIZE,
    # )
    
    # benchmark Triton kernel
    welford_algorithm[grid](
        x,        
        mean, var,
        x.stride(0), x.stride(1),  
        mean.stride(0),
        var.stride(0),
        ROWS=ROWS, D=D,
        ROWS_TILE_SIZE=ROWS_TILE_SIZE,
        D_TILE_SIZE=D_TILE_SIZE,
    )
    
    # benchmark Pytorch kernel
    mean_real, var_real = naive_torch(x)   
     
    
    # check: correctness proved
    print(torch.allclose(mean_real, mean))
    print(torch.allclose(var_real, var))
    
    mean = torch.empty((ROWS,), device=x.device, dtype=torch.float32)
    var = torch.empty((ROWS,), device=x.device, dtype=torch.float32)
    
    def run_welford(_x=x, ROWS_TILE_SIZE=32, D_TILE_SIZE=1024):
        ROWS, D = _x.shape
                
        # 2. 定义 Grid：按行分块
        # 因为你的 kernel 逻辑里 row_tile_idx = tl.program_id(0)
        grid = (triton.cdiv(ROWS, ROWS_TILE_SIZE),)
        
        # 3. 启动 Kernel
        welford_algorithm[grid](
            x_ptr=_x,
            mean_ptr=mean,
            var_ptr=var,
            x_stride_row=_x.stride(0),
            x_stride_dim=_x.stride(1),
            mean_stride=mean.stride(0),
            var_stride=var.stride(0),
            ROWS=ROWS,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )
        return mean, var
    
    # benchmark triton implementation 
    ms_triton = triton.testing.do_bench(run_welford)
    gbps = (x.nelement() * x.element_size()) / (ms_triton * 1e-3) / 1e9
    print(f"Time: {ms_triton:.4f} ms")
    print(f"Bandwidth: {gbps:.2f} GB/s")
    
    def run_triton(_x=x):
        return naive_torch_compiled(_x)
    
    ms_torch = triton.testing.do_bench(run_triton)
    gbps = (x.nelement() * x.element_size()) / (ms_torch * 1e-3) / 1e9
    print(f"Time: {ms_torch:.4f} ms")
    print(f"Bandwidth: {gbps:.2f} GB/s")
    
    
    # benchmark pytorch implementation
    