import torch
import torch.nn as nn

# 定义LeNet网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 输入通道数改为3
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # 输入大小根据图像尺寸调整
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)  # 输出类别数改为5
        self.dropout = nn.Dropout(p=0.5)  # 添加Dropout层，丢弃率为0.5
        self.batch_norm1 = nn.BatchNorm2d(6)  # 添加批量归一化层
        self.batch_norm2 = nn.BatchNorm2d(16)  # 添加批量归一化层
        self.batch_norm3 = nn.BatchNorm1d(120)  # 添加批量归一化层
        self.batch_norm4 = nn.BatchNorm1d(84)  # 添加批量归一化层

    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.conv1(x)))  # 在卷积层之后添加批量归一化
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.batch_norm2(self.conv2(x)))  # 在卷积层之后添加批量归一化
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16 * 53 * 53)
        x = self.dropout(torch.relu(self.batch_norm3(self.fc1(x))))  # 在全连接层之后添加Dropout和批量归一化
        x = self.dropout(torch.relu(self.batch_norm4(self.fc2(x))))  # 在全连接层之后添加Dropout和批量归一化
        x = self.fc3(x)
        return x

Model = LeNet

if __name__ == '__main__':
    model = Model()
    print(model)
    input = torch.randn(2, 3, 224, 224)
    output = model(input)
    print(output.shape)