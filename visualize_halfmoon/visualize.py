import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import make_moons
import torch
import torch.nn as nn
import torch.optim as optim

# 生成半月形数据集
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# 定义极简 MLP 模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x, _ = torch.sort(x,dim=-1)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleMLP()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# 创建网格用于绘制函数
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# 创建 matplotlib 图像对象
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='coolwarm', edgecolor='k')
contour = ax.contourf(xx, yy, np.zeros_like(xx), levels=50, cmap='RdBu', alpha=0.7)
ax.set_title('Training Visualization')

# 定义更新函数
def update(frame):
    global model, optimizer, criterion
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    # 生成输出值用于更新等高线图
    model.eval()
    with torch.no_grad():
        grid_outputs = model(grid)
        grid_outputs = torch.sigmoid(grid_outputs).reshape(xx.shape)

    # 更新等高线图
    ax.clear()
    ax.contourf(xx, yy, grid_outputs, levels=50, cmap='RdBu', alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='coolwarm', edgecolor='k')
    ax.set_title(f'Training Iteration: {frame + 1}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=1000, repeat=False)

# 显示动画
plt.show()
