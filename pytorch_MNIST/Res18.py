from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 当输入输出通道数不同或步幅不为1时进行下采样
        
    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = torch.relu(out)
        
        return out

# 定义ResNet18网络
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 输入为单通道灰度图
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 添加ResNet的4个阶段
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
train_data = datasets.MNIST(root="data", train=True, transform=transform, download=True)
test_data = datasets.MNIST(root="data", train=False, transform=transform, download=True)

# 定义数据加载器
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# 打印数据集信息
print("训练集大小：", len(train_data))
print("测试集大小：", len(test_data))
print("图像大小：", train_data[0][0].size())
print("标签：", train_data.classes)
print("类别数：", len(train_data.classes))
print("训练集批次数：", len(train_loader))
print("测试集批次数：", len(test_loader))

model = ResNet18()
model.to(device)
criterion = nn.CrossEntropyLoss()
epochs = 10
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss_batch_list = []
train_acc_batch_list = []
test_loss_list = []
test_loss_batch_list = []
test_acc_list = []
# 训练模型
for epoch in range(epochs):  
    model.train()
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        train_loss_batch_list.append(loss.item())
        train_acc_batch_list.append((torch.argmax(output, 1) == labels).sum().item() / len(labels))

        loss.backward()
        optimizer.step()
    
    

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            test_loss_batch_list.append(loss.item())

            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_loss_list.append(sum(test_loss_batch_list) / len(test_loss_batch_list))
        test_acc_list.append(correct / total)
        test_loss_batch_list = []

    print(f"Accuracy: {100 * correct / total}%")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
# train loss per batch
plt.plot(train_loss_batch_list, label="train")
# test loss transform to per epoch
plt.plot([i * len(train_loader) for i in range(1, len(test_loss_list) + 1)], test_loss_list, label="test")
plt.title("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_acc_batch_list, label="train")
plt.plot([i * len(train_loader) for i in range(1, len(test_acc_list) + 1)], test_acc_list, label="test")
plt.title("Accuracy")
plt.legend()
plt.show()
