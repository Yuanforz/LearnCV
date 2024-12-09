import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
#from LeNet import Model
#from VGG import Model
#from Res18 import Model
from ViT import Model

# 定义数据变换
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # 随机水平翻转
    A.Rotate(limit=10, p=0.5),  # 随机旋转
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),  # 随机颜色抖动
    A.GaussianBlur(blur_limit=3, p=0.5),  # 高斯模糊
    A.RandomResizedCrop(224, 224, scale=(0.5, 1.0), ratio=(0.75, 1.333), p=0.5),  # 随机裁剪
    A.Resize(224, 224),  # 缩放到224x224
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])
test_transform = A.Compose([
    A.Resize(224, 224),  # 缩放到224x224
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_folder, label_file,readstart=0,readend=-1, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = []
        self.labels = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if  int(parts[0]) < readstart or (readend != -1 and int(parts[0]) >= readend):
                    continue
                self.image_files.append(parts[0])
                self.labels.append([int(label) for label in parts[1:]])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        img_path = img_path + '.png'  # 图像文件后缀为.png
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.zeros(5)  # 有5个类别
        for l in self.labels[idx]:
            label[l] = 1
        return image, label


# 指定前 n 张图片为训练集，其余为验证集
n = 6000  # 设定前 n 张图片为训练集
# 加载数据集
train_dataset = CustomDataset(image_folder='img', label_file='img/labels.txt', readstart=0, readend=n, transform=train_transform)
val_dataset = CustomDataset(image_folder='img', label_file='img/labels.txt', readstart=n, readend=-1, transform=test_transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 获取数据集信息
train_data = train_dataset
val_data = val_dataset

# 打印数据集信息
print("训练集大小：", len(train_data))
print("验证集大小：", len(val_data))
print("图像大小：", train_data[0][0].size())
print("标签：", train_data[0][1])
print("训练集批次数：", len(train_loader))
print("验证集批次数：", len(val_loader))


if __name__ == '__main__':
    device = torch.device("cuda")
    model = Model()
    #model.load_state_dict(torch.load('model.pth'))
    model = model.to(device)

    print(model)
    print(device)
    epochs = 30
    pos_weight = torch.tensor([1.0, 1.0, 1.5, 1.5, 2.0]).to(device)  # 设置正样本权重
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 使用二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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

            loss.backward()
            optimizer.step()

        # 验证模型
        model.eval()
        tp_list, fp_list, fn_list = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                test_loss_batch_list.append(loss.item())

                # 使用固定阈值0.5进行多目标分类
                preds = torch.sigmoid(output) > 0.5
                test_acc_list.append((preds == labels).float().mean().item())
                total_1 = ((preds == 1) | (labels == 1)).sum().item()
                tp_list.append(((preds == 1) & (labels == 1)).sum().item()/total_1)
                fp_list.append(((preds == 1) & (labels == 0)).sum().item()/total_1)
                fn_list.append(((preds == 0) & (labels == 1)).sum().item()/total_1)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {sum(train_loss_batch_list)/len(train_loss_batch_list)}, Test Loss: {sum(test_loss_batch_list)/len(test_loss_batch_list)}, Test Acc: {sum(test_acc_list)/len(test_acc_list)}")
        precision = sum(tp_list) / (sum(tp_list) + sum(fp_list) + 1e-6)
        recall = sum(tp_list) / (sum(tp_list) + sum(fn_list) + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

    print("训练完成！")

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')