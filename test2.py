# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# 加载数据
data = pd.read_csv('C:\\Users\\hf\\Desktop\\python组免试题\\免试题2\\train.csv', header=None)  # 从CSV文件加载数据
X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)  # 提取特征并转换为PyTorch张量
y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)  # 提取标签并转换为PyTorch张量

# 定义模型
net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))  # 创建一个神经网络模型

# 定义损失函数和优化器
loss = nn.MSELoss()  # 均方误差损失函数，用于回归问题
optimizer = optim.Adam(net.parameters(), lr=0.03)  # Adam优化器，用于更新模型参数

# 训练模型
epochs = 7000  # 训练周期数
for epoch in range(epochs):
    optimizer.zero_grad()  # 清零梯度，防止梯度累积
    outputs = net(X)  # 使用模型进行前向传播
    l = loss(outputs, y)  # 计算预测值与实际值之间的损失
    l.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新模型参数，执行优化步骤

# 返回模型参数
print(net.state_dict())  # 打印训练后的模型参数
