from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),  # 调整尺寸
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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

model = VGG()
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
