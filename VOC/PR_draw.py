import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train import val_loader, Model

# 加载模型
device = torch.device("cpu")
model = Model().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))  # 替换为实际的模型路径
model.eval()

# 加载验证集
# val_dataset = CustomDataset(image_folder='img', label_file='img/labels.txt', transform=test_transform)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 收集所有预测和真实标签
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# 计算 Precision-Recall 曲线和平均精度
precision = dict()
recall = dict()
average_precision = dict()
n_classes = all_labels.shape[1]

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(all_labels[:, i], all_preds[:, i])
    average_precision[i] = average_precision_score(all_labels[:, i], all_preds[:, i])

# 绘制每个类别的 PR 曲线
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (AP = {average_precision[i]:0.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid()
plt.show()