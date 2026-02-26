# from cs336_basics import *
from cs336_basics.model import BasicsTransformerLM
import torch
import torch.nn.functional as F
import timeit
import statistics

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

# 遍历所有模型配置
for model_name, config in model_configs.items():
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"配置: d_model={config['d_model']}, d_ff={config['d_ff']}, "
          f"num_layers={config['num_layers']}, num_heads={config['num_heads']}")
    print(f"{'='*60}\n")
    
    # init model
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=10000.0
    ).to(device)

    # randomly generate data
    # shape of data: [batch_size, sequence_length], type of data: int
    data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # warmup steps
    if warmup:
        print(f"warmup运行")
        for _ in range(warmup_iter):
            logits = model(data)
        if device.type == 'cuda':
            torch.cuda.synchronize() 
        # Note: warmup也需要backward pass
            if time_backward:
                loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
                loss.backward()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
        
    else:
        print(f"无warmup运行")

    # benchmark steps
    results_foward = []
    results_backward = []
    results_total = []
    for _ in range(benchmark_iter):
        time1 = 0
        time2 = 0
        if time_forward:
            start = timeit.default_timer()
            logits = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = timeit.default_timer()
            time1 = end - start
            results_foward.append(time1)
            # print(f"forward pass运行耗时: {end - start:.6f} 秒")
        
        if time_backward:
            start = timeit.default_timer()
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = timeit.default_timer()
            time2 = end - start
            results_backward.append(time2)
            # print(f"backward pass运行耗时: {end - start:.6f} 秒")

        if time_forward and time_backward:
            results_total.append(time1 + time2)
            # print(f"total pass运行耗时: {time1 + time2:.6f} 秒")
            
        # zero auto grad
        model.zero_grad()
    
    print(f"是否有warmup:{warmup}")
    if warmup:
        print(f"warmup iterations: {warmup_iter}")
    if time_forward:
        mean_val = statistics.mean(results_foward)
        variance_val = statistics.stdev(results_foward)
        print(f"\n{model_name} forward pass 模型结果:")
        print(f"平均值: {mean_val:.6f} 秒, 标准差: {variance_val:.6f}")
    if time_backward:
        mean_val = statistics.mean(results_backward)
        variance_val = statistics.stdev(results_backward)
        print(f"\n{model_name} backward pass 模型结果:")
        print(f"平均值: {mean_val:.6f} 秒, 标准差: {variance_val:.6f}")
    if time_backward and time_forward:
        mean_val = statistics.mean(results_total)
        variance_val = statistics.stdev(results_total)
        print(f"\n{model_name} forward and backward pass 模型结果:")
        print(f"平均值: {mean_val:.6f} 秒, 标准差: {variance_val:.6f}")
    
    # 清理模型以释放内存
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()