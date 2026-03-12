import os
import timeit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, backend="gloo"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False, op=dist.ReduceOp.SUM)
    print(f"rank {rank} data (after all-reduce): {data}")
    
# 伪代码思路
def benchmark_all_reduce(rank, world_size, size_mb, backend="gloo", num_iters=10):
    setup(rank, world_size, backend)
    
    # 1. 准备数据
    n_elements = (size_mb * 1024 * 1024) // 4  # float32 占 4 bytes
    data = torch.randn(n_elements)
    
    # 2. 预热
    for _ in range(5):
        dist.all_reduce(data)
    if backend == 'nccl':
        torch.cuda.synchronize()
    
    # 3. 正式计时
    start_time = timeit.default_timer()
    for _ in range(num_iters):
        dist.all_reduce(data)
    if backend == 'nccl':
        torch.cuda.synchronize()
    end_time = timeit.default_timer()
    
    avg_time = (end_time - start_time) / num_iters
    # 之后收集所有 rank 的 avg_time 做分析
    
    # print the average time 
    # print(f"Average runtime for rank {rank}: {avg_time}")
    
    # use dist.all_gather_object to gather data from workers
    gather_list = [None for _ in range(world_size)]
    local_data = {'rank': rank, "avg_time": avg_time}
    dist.all_gather_object(gather_list, local_data)
    
    # calculate the average time of all workers
    if rank == 0: 
        times = [item['avg_time'] for item in gather_list]
        print(f'Setup: world_size: {world_size}; size_mb: {size_mb}. Average time of all_reduce: {sum(times) / len(times)}')
    

if __name__ == "__main__":
    world_sizes = [2,4,6]
    size_mbs = [1, 10, 100, 1024]
    for world_size in world_sizes:
        for size_mb in size_mbs:
            mp.spawn(fn=benchmark_all_reduce, args=(world_size, size_mb), nprocs=world_size, join=True)

# result:
# /home/exouser/cs336/assignment2-systems/.venv/bin/python /home/exouser/cs336/assignment2-systems/cs336_systems/single_node.py
# Setup: world_size: 2; size_mb: 1. Average time of all_reduce: 0.0028260210514417846
# Setup: world_size: 2; size_mb: 10. Average time of all_reduce: 0.01110658730030991
# Setup: world_size: 2; size_mb: 100. Average time of all_reduce: 0.06552880689996528
# Setup: world_size: 2; size_mb: 1024. Average time of all_reduce: 0.8440175713010831
# Setup: world_size: 4; size_mb: 1. Average time of all_reduce: 0.0031647121752030214
# Setup: world_size: 4; size_mb: 10. Average time of all_reduce: 0.01246256982558407
# Setup: world_size: 4; size_mb: 100. Average time of all_reduce: 0.14036746909987413
# Setup: world_size: 4; size_mb: 1024. Average time of all_reduce: 1.2840342192997922
# Setup: world_size: 6; size_mb: 1. Average time of all_reduce: 0.00578858473309083
# Setup: world_size: 6; size_mb: 10. Average time of all_reduce: 0.025816386449635808
# Setup: world_size: 6; size_mb: 100. Average time of all_reduce: 0.2173778752500463
# Setup: world_size: 6; size_mb: 1024. Average time of all_reduce: 1.532787616866699


# analysis:
# more workers, higher communication overhead
# more number of tensors, higher runtime due to limited memory bandwidth