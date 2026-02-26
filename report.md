# Assignment 2: Systems - Report

## 1.1.3 End-to-End Benchmarking

### (b) Forward and Backward Pass Timings (with 5 warmup steps, 10 measurement steps)

| Model Size | Forward (mean ± std) | Backward (mean ± std) | Total (mean ± std) |
|------------|----------------------|-----------------------|--------------------|
| small      | 0.0340 ± 0.0016 s   | 0.0426 ± 0.0046 s    | 0.0766 ± 0.0048 s |
| medium     | 0.0630 ± 0.0018 s   | 0.1013 ± 0.0056 s    | 0.1643 ± 0.0072 s |

The forward pass is consistently faster than the backward pass (roughly 1.3–1.6x faster), which is expected since the backward pass must compute gradients through all layers and store intermediate results. The standard deviation is small relative to the mean (< 10%), indicating low variability and stable measurements with warmup.

### (c) Effect of Warmup Steps

warmup的作用如下：
1. CUDA Context Initialization（CUDA 上下文初始化）： 当你第一次在代码中调用 CUDA 相关的操作时，NVIDIA 驱动程序必须在 GPU 上创建一个 CUDA Context。这涉及到在 Host（CPU）和 Device（GPU）之间建立通信渠道、分配基础资源等，是一个开销极大的单次系统调用。
2. PyTorch Caching Allocator（PyTorch 显存分配策略）： 这是影响最大的因素之一。PyTorch 并不是每次遇到矩阵乘法都去找底层要显存，而是自己维护了一个“显存池（Caching Allocator）”。
  - 前几次运行：遇到新的 Tensor 和梯度计算，PyTorch 发现池子里没显存，只能被迫调用极慢的底层 API（如 cudaMalloc）向操作系统要显存。
  - Warmup 之后：池子建好了，显存被 PyTorch 内部缓存下来。之后的每一轮前向和反向传播，都只是在复用池子里的显存，速度极快。如果你不把显存分配的时间从 Benchmark 里剔除，你测出来的就不是纯粹的算力性能。
3. Kernel Loading & Auto-Tuning（内核加载与自动调优）：
  - 具体的计算指令（CUDA Kernels）需要被加载到 GPU 的指令缓存（Instruction Cache）中。
  - 此外，底层库（如 cuBLAS 或 cuDNN）非常聪明。在第一次遇到某种特定 shape 的矩阵乘法或卷积时，它们可能会在后台偷偷运行几个“微测试（Heuristics）”，来决定针对你当前的数据形状，用哪一种具体的硬件算法跑得最快。这个寻找最优解的过程只在最开始发生。
4. Hardware Power States (硬件电源状态 / P-States)： 当 GPU 处于空闲状态时，为了省电，它的时钟频率会降得很低。当你突然塞给它一个庞大的 Transformer 模型时，硬件需要几毫秒到几十毫秒的时间来提高电压、唤醒核心，并把主频拉升到最高的 Boost Clock。如果不做 Warmup，你实际上是在用 GPU 的“低频省电模式”测算力。

如果没有warmup，系统初始化的开销会被计入第一次的前向传播。导致表现变差。 

结果，模型越大，运行时间越长。多次benchmark结果表现稳定，说明warmup是有效的。但是初次之外没有更多的insight。