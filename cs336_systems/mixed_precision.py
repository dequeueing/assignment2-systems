import torch
import torch.nn as nn


def accuracy():
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s)

    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)



class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(x.dtype)
        x = self.ln(x)
        print(x.dtype)
        x = self.fc2(x)
        print(x.dtype)
        return x
    
    
if __name__ == '__main__':
    model = ToyModel(10,10).cuda()
    dtype: torch.dtype = torch.float16
    # dtype: torch.dtype = torch.bfloat16
    x: torch.Tensor = torch.rand(10, device="cuda")
    targets: torch.Tensor = torch.rand(10, device="cuda")
    
    optimizer = torch.optim.AdamW(model.parameter(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda")  # 只在 float16 需要
    criterion = nn.MSELoss()
        
    with torch.autocast(device_type="cuda",dtype=dtype):
        y = model(x)
        loss = criterion(y, targets)
        
    # backward
    optimizer.zero_grad()           # 清零上一步的梯度
    scaler.scale(loss).backward()   # loss * scale → backward
    scaler.step(optimizer)          # unscale → check inf/nan → optimizer.step
    scaler.update()                 # 动态调整 scale factor
    
    print(f"fc1 weight gradient dtype: {model.fc1.weight.grad.dtype}")