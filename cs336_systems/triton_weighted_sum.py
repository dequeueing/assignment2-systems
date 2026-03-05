import triton
import triton.language as tl

import torch
import torch.nn as nn

from einops import rearrange # Required for the rearrange call


@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,  # Input pointers
    output_ptr,  # Output pointer
    x_stride_row, x_stride_dim,  # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim,  # Likely 1
    output_stride_row,  # Likely 1
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,  # Tile shapes must be known at compile time
):
    # Each instance will compute the weighted sum of a tile of rows of x.
    # `tl.program_id` gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)

    # Block pointers give us a way to select from an ND region of memory
    # and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the memory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to use load/store at a time
    # - The order of the dimensions in memory from major to minor
    # axes (= np.argsort(strides)) for optimizations, especially useful on H100

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # Initialize a buffer to write to
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load the current block pointer
        # Since ROWS_TILE_SIZE might not divide ROWS, and D_TILE_SIZE might not divide D,
        # we need boundary checks for both dimensions
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)

        # Compute the weighted sum of the row.
        output += tl.sum(row * weight[None, :], axis=1)

        # Move the pointers to the next tile.
        # These are (rows, columns) coordinate deltas
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))  # Move by D_TILE_SIZE in the last dimension
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))  # Move by D_TILE_SIZE

    # Write output to the output block pointer (a single scalar per row).
    # Since ROWS_TILE_SIZE might not divide ROWS, we need boundary checks
    tl.store(output_block_ptr, output, boundary_check=(0,))
    
    

# Helper for ceiling division
def cdiv(a, b):
    return (a + b - 1) // b




class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Cache variables for the backward pass
        D, output_dims = x.shape[-1], x.shape[:-1]
        
        # Reshape input tensor to 2D for the kernel
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        
        ctx.save_for_backward(x, weight)
        
        # Validation checks
        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"
        
        # Define tile sizes
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape
        
        # Initialize output tensor
        y = torch.empty(output_dims, device=x.device)
        n_rows = y.numel()
        
        # Launch Triton kernel
        weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, 
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        
        return y.view(input_shape[:-1])
    
    @staticmethod
    def backward(ctx, grad_out):
        # 1. 恢复缓存的张量和超参数
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE = ctx.ROWS_TILE_SIZE
        D_TILE_SIZE = ctx.D_TILE_SIZE
        n_rows, D = x.shape

        # 2. 分配输出梯度内存
        # partial_grad_weight 用于存储每个线程块计算的局部梯度累加值
        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), 
                                         device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        # 3. 启动内核
        # 网格大小（Launch Grid）设置为行分块的数量
        grid = (triton.cdiv(n_rows, ROWS_TILE_SIZE),)
        weighted_sum_backward[grid](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            n_rows, D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )

        # 4. 全局规约：对所有行块产生的局部梯度进行求和，得到最终的 grad_weight
        grad_weight = partial_grad_weight.sum(axis=0)
        
        return grad_x.view(ctx.input_shape), grad_weight

    
def use_triton_function():
    ROWS, D = 64, 128
    
    # init tensors
    x      = torch.randn(ROWS, D, device="cuda")
    weight = torch.randn(D,       device="cuda")
    output = torch.empty(ROWS,    device="cuda")
    
    # compute tile sizes 
    ROWS_TILE_SIZE = 16
    D_TILE_SIZE    = triton.next_power_of_2(D) // 16  # = 8
    
    
    def cdiv(a, b): return (a + b - 1) // b
    grid = (cdiv(ROWS, ROWS_TILE_SIZE),)  # = (4,)

    weighted_sum_fwd[grid](
        x, weight,        
        output,
        x.stride(0), x.stride(1),  
        weight.stride(0),
        output.stride(0),
        ROWS=ROWS, D=D,
        ROWS_TILE_SIZE=ROWS_TILE_SIZE,
        D_TILE_SIZE=D_TILE_SIZE,
    )
    
    
    expected = x @ weight
    print(torch.allclose(output, expected, atol=1e-4))
    
    

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,              # 输入张量指针
    grad_output_ptr,                # 输出梯度指针 (dL/dy)
    grad_x_ptr,                     # 待填充的 X 梯度指针
    partial_grad_weight_ptr,        # 待填充的 w 局部梯度缓冲区
    stride_xr, stride_xd,           # X 的步长
    stride_wd,                      # w 的步长
    stride_gr,                      # grad_output 的步长
    stride_gxr, stride_gxd,         # grad_x 的步长
    stride_gwb, stride_gwd,         # partial_grad_weight 的步长
    NUM_ROWS, D,                    # 矩阵形状
    ROWS_TILE_SIZE: tl.constexpr,   # 行分块大小
    D_TILE_SIZE: tl.constexpr,      # 维度分块大小
):
    # 获取当前程序实例 ID 和总实例数
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    # 1. 设置块指针 (Block Pointers)
    grad_output_block_ptr = tl.make_block_ptr(
        base=grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        base=grad_x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        base=partial_grad_weight_ptr,
        shape=(n_row_tiles, D,),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    # 2. 迭代维度 D 进行计算
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 加载当前 Tile 的数据
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # 计算 grad_x (外积: dL/dy * w)
        # 使用广播机制自动匹配形状进行计算
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # 计算局部 grad_weight (行缩减: dL/dy * X)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))

        # 沿着维度 D 推进指针
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))


f_weightedsum = WeightedSumFunc.apply

class WeightedSumLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 定义权重，设为 Parameter 后 PyTorch 会自动记录其梯度 
        self.weight = nn.Parameter(torch.randn(d_model, device='cuda'))

    def forward(self, x):
        # 2. 在 forward 中直接使用该函数 
        # x 是输入矩阵，self.weight 是该层的权重向量
        return f_weightedsum(x, self.weight)

def use_torch_weighted_sum_module():
    d_model = 128
    layer = WeightedSumLayer(d_model).cuda()
    
    # 随机生成一个输入 Batch (例如 batch_size=32)
    input_data = torch.randn(32, d_model, device='cuda', requires_grad=True)

    # 像普通层一样调用
    output = layer(input_data)

    # 检查输出，你会发现它带有 <WeightedSumFuncBackward>
    print(output.grad_fn)

def apply_autograd():
    x      = torch.randn(2, 3, 128, device="cuda", requires_grad=True)
    weight = torch.randn(128,       device="cuda", requires_grad=True)

    # 调用
    y = WeightedSumFunc.apply(x, weight)
    # y.shape → (2, 3)

    # 正常反向传播
    loss = y.sum()
    loss.backward()

    print(x.grad.shape)      # (2, 3, 128)
    print(weight.grad.shape) # (128,)

if __name__ == '__main__':
    # use_triton_function()
    # apply_autograd()
    use_torch_weighted_sum_module()
    
    
