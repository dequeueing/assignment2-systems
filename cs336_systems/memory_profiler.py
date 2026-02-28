import torch
import torch.nn as nn
import torch.nn.functional as F


from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy


def demo():

    # 1. 环境检查
    if not torch.cuda.is_available():
        print("⚠️ 警告：检测不到 GPU，内存分析器只能在 CUDA 环境下运行。")
        exit()

    # 2. 定义一个模型，让显存有点“动静”
    model = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048)
    ).cuda()

    input_data = torch.randn(512, 2048).cuda()

    # --- 核心分析部分 ---

    print("Step 1: 启动内存历史记录...")
    # max_entries=100000 是为了防止记录过多导致程序卡顿
    torch.cuda.memory._record_memory_history(max_entries=100000, stacks='all')

    try:
        print("Step 2: 执行运算（模拟训练循环）...")
        for i in range(3):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            optimizer.zero_grad()
            
            output = model(input_data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            print(f"  Iteration {i+1} done.")

        # Step 3: 导出快照文件
        print("Step 4: 导出快照文件 'memory_snapshot.pickle'...")
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        
    finally:
        # Step 5: 无论成功失败，最后都要停止记录，释放分析器占用的额外内存
        print("Step 6: 停止记录。")
        torch.cuda.memory._record_memory_history(enabled=None)

    print("\n🎉 完成！你现在当前目录下可以看到 'memory_snapshot.pickle' 了。")
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_configs = {
    'small': {'vocab_size': 10000, 'context_length': 256, 'd_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'vocab_size': 10000, 'context_length': 256,'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
    # 'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    # 'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    # '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32},
}

# hyperparameters for models
batch_size = 4
vocab_size = 10000
seq_len = 100

# hyperparameters for benchmarking 
warmup_iter = 5
benchmark_iter = 5

    
def benchmark_llm_full(config):
    # init model, loss and optimizer
    model = BasicsTransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=10000.0
    ).to(device)
    
    criterion = cross_entropy
    
    optimizer = AdamW(
        params=model.parameters(),
        lr=1e-3,              
        betas=(0.9, 0.999),   
        eps=1e-8,             
        weight_decay=0.01,    
    )
    
    # init data 
    data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # warmup
    for _ in range(warmup_iter):
        logits = model(data)
        if device.type == 'cuda':
            torch.cuda.synchronize() 
        # Note: warmup也需要backward pass
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
    # Memory profiling region
    torch.cuda.memory._record_memory_history(max_entries=100000, stacks='all')        
    try:
         
        # benchmark
        for _ in range(benchmark_iter):
            # zero up gradiant
            optimizer.zero_grad()
            
            # forward pass
            logits = model(data)
            loss = criterion(logits, targets)
            
            # debug: print the size of logits and targets
            # print(f"size of logits: {logits.size()}")
            # print(f"size of targets: {targets.size()}")
            
            # backward pass
            loss.backward()
            
            # update optimizer
            optimizer.step()
            
            # sync to ensure everything finishes
            torch.cuda.synchronize()
        
        torch.cuda.memory._dump_snapshot("llm_benchmarking_complete.pickle")
        print("file dumped")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
        
        
        
def benchmark_llm_forward_only(config):
    # init model, loss and optimizer
    model = BasicsTransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=10000.0
    ).to(device)
    
    criterion = cross_entropy
    
    optimizer = AdamW(
        params=model.parameters(),
        lr=1e-3,              
        betas=(0.9, 0.999),   
        eps=1e-8,             
        weight_decay=0.01,    
    )
    
    # init data 
    data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # warmup
    for _ in range(warmup_iter):
        logits = model(data)
        if device.type == 'cuda':
            torch.cuda.synchronize() 
        # Note: warmup也需要backward pass
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
    # Memory profiling region
    torch.cuda.memory._record_memory_history(max_entries=100000, stacks='all')        
    try:
         
        # benchmark
        for _ in range(benchmark_iter):            
            # forward pass
            logits = model(data)
            
            # sync to ensure everything finishes
            torch.cuda.synchronize()
        
        torch.cuda.memory._dump_snapshot("llm_benchmarking_forward_only.pickle")
        print("file dumped")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    
    
    
if __name__ == '__main__':
    # demo()
    # benchmark_llm(model_configs['medium'])
    # benchmark_llm_forward_only(model_configs['medium'])
    benchmark_llm_full(model_configs['medium'])