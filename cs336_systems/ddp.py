import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp


    
            
class ToyModel(torch.nn.Module):
    def __init__(self, in_feature=5, out_feature=10):
        super().__init__()
        self.linear1  = torch.nn.Linear(in_feature, 10)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, out_feature)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 CUDA 卷积等操作也是确定性的
    torch.backends.cudnn.deterministic = True

def worker_demo(rank, world_size, mig_uuids):
    # --- 关键修复 1: 解决 NCCL 在 MIG 上的冲突 ---
    os.environ["NCCL_P2P_DISABLE"] = "1"   # 禁用 P2P，因为 MIG 实例间不支持它
    os.environ["NCCL_IB_DISABLE"] = "1"   # 禁用 IB (InfiniBand)，在单卡 MIG 环境下通常不需要
    os.environ["NCCL_SHM_DISABLE"] = "0" # 强制开启共享内存
    os.environ["NCCL_NET_GDR_LEVEL"] = "0" # 禁用 GPU Direct
    
    # 手动设置分布式环境
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    # 关键：MIG 绑定
    os.environ["CUDA_VISIBLE_DEVICES"] = mig_uuids[rank]
    
    try:
        # 初始化进程组
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        
        # 锁定设备
        torch.cuda.set_device(0)

        # 业务逻辑
        data = torch.tensor([float(rank)]).cuda()
        print(f"[Rank {rank}] 启动成功！使用 MIG: {mig_uuids[rank][:15]}... 初始值: {data.item()}")
        
        # 执行同步
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        
        print(f"[Rank {rank}] All-Reduce 完成！结果: {data.item()}")

    except Exception as e:
        print(f"[Rank {rank}] 发生错误: {e}")
    finally:
        # 无论成功还是失败，都要清理进程组，否则会泄露资源并报错
        if dist.is_initialized():
            dist.destroy_process_group()
            
            
def worker_set(rank, world_size, mig_uuids):
    # set the random seed
    set_seed()
    
    # setup process groups
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    
    # bind GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = mig_uuids[rank]
    torch.cuda.set_device(0)

            
            
def ddp(rank, world_size, mig_uuids, x_global):
    # setup
    worker_set(rank, world_size, mig_uuids)

    # create a model for each worker 
    model = ToyModel().to('cuda')
    model.load_state_dict(torch.load("initial_weights.pt"))
    
    # sync all the models first 
    # Note: since we load the same weight, sync is not needed
    # for param in model.parameters():
    #     dist.broadcast(param.data, src = 0)
    
    # prepare inputs
    # x_global shape: batch_size, in_fea,
    # each worker should have range: batch_size / world_size
    batch_size_global = x_global.shape[0]
    batch_size_local = batch_size_global // world_size # FIXME: / will produce a float, use // to get an integer
    x_local = x_global[rank * batch_size_local: (rank + 1) * batch_size_local ,:].to('cuda')
    
    
    # init optimizer
    optimizer = optim.AdamW(model.parameters())
    
    # forward and backward
    y = model(x_local)
    loss = torch.sum(y)
    loss.backward()
    
    # now gradient is ready, all-reduce! 
    for param in model.parameters(): 
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
        
    # now all workers should have the same copy of gradient 
    optimizer.step()
    
    # save the model weights
    if rank == 0:
        torch.save(model.state_dict(), "ddp_weights.pt")
    

    
def train_model():
    # prepare input
    batch_size = 90
    in_feature=5
    out_feature=10
    x = torch.randn((batch_size, in_feature,))
    
    model = ToyModel()
    optimizer = optim.AdamW(model.parameters())
    
    # forward, backward and optimizer step()
    y = model(x)
    loss = torch.sum(y)
    loss.backward()
    optimizer.step()
    
    print("Finished!")
    
    
        

def run_mig():
    my_migs = [
        "MIG-3f18af02-5e92-5bf8-9b58-64a5f2ce138e",
        "MIG-08f9474c-c0e2-5f66-b24e-d3264326151f",
        "MIG-d632caf3-aa2c-5930-8cc8-50d288ada72d"
    ]
    
    world_size = len(my_migs)
    print(f"即将启动 {world_size} 个进程...")
    
    mp.spawn(
        worker_demo,
        args=(world_size, my_migs),
        nprocs=world_size,
        join=True
    )
    
    
def run_single_node(x):
    # init and load model parameters
    x = x.to('cuda')
    model = ToyModel()
    model.load_state_dict(torch.load("initial_weights.pt"))
    model.to('cuda')
    
    # forward, backward and optimizer step
    optimizer = optim.AdamW(model.parameters())
    y = model(x)
    loss = torch.sum(y)
    loss.backward()
    optimizer.step()
    
    # save the weight
    torch.save(model.state_dict(), "single_node_weights.pt")
    
    
    
    
def run_ddp(x_global):    
    my_migs = [
        "MIG-3f18af02-5e92-5bf8-9b58-64a5f2ce138e",
        "MIG-08f9474c-c0e2-5f66-b24e-d3264326151f",
        "MIG-d632caf3-aa2c-5930-8cc8-50d288ada72d"
    ]
    world_size = len(my_migs)
    
    
    mp.spawn(
        ddp,
        args=(world_size, my_migs, x_global),
        nprocs=world_size,
        join=True
    )
    
def run_comparison(path="initial_weights.pt"):
    set_seed()
    
    # init model and save the parameters
    model = ToyModel()
    initial_state = model.state_dict()
    torch.save(initial_state, path)
    
    # prepare global input 
    batch_size_global = 90
    in_fea = 5
    x_global = torch.randn((batch_size_global, in_fea))
    
    # run model training on a single worker first 
    run_single_node(x_global)
    run_ddp(x_global)
    
    # compare the weights
    weight1 = torch.load("single_node_weights.pt")
    weight2 = torch.load("ddp_weights.pt")
    
    for item in weight1:
        print(torch.allclose(weight1[item], weight2[item]))
    
    
    
    
if __name__ == "__main__":
    run_comparison()