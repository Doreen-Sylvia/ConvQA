# test_pytorch.py
import torch
import torch.nn as nn

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")


# 测试简单的神经网络
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


model = SimpleModel()
x = torch.randn(2, 10)
output = model(x)
print(f"模型测试成功! 输入形状: {x.shape}, 输出形状: {output.shape}")