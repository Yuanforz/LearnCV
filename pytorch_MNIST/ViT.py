import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.2):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.Q = nn.Linear(embed_size, embed_size)
        self.K = nn.Linear(embed_size, embed_size)
        self.V = nn.Linear(embed_size, embed_size)
        self.linear = nn.Linear(embed_size, embed_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, q, k, v):
        q = self.Q(q).reshape(q.size(0), q.size(1), self.num_heads, self.head_size)
        k = self.K(k).reshape(k.size(0), k.size(1), self.num_heads, self.head_size)
        v = self.V(v).reshape(v.size(0), v.size(1), self.num_heads, self.head_size)

        scores = torch.einsum("bqhe,bkhe->bhqk", q, k) #/ math.sqrt(self.head_size)
        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores = self.dropout_layer(scores)

        attention = torch.einsum("bhqk,bkhe->bqhe", scores, v).reshape(v.size(0), -1, self.embed_size)
        return self.linear(attention)
    
class MyTransformer(nn.Module):
    def __init__(self, embed_size, heads,hidden_size, dropout=0.0,activation=nn.GELU()):
        super(MyTransformer, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size, heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x= self.norm1(x)
        x = x + self.attention(x, x, x)
        x = x + self.mlp(self.norm2(x))
        return x

# 定义ViT模型
class ViT(nn.Module):
    def __init__(self, image_size=28, patch_size=7, num_classes=10, embed_size=128, num_heads=8, num_layers=1,dropout=0.2):
        super(ViT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_size ** 2, embed_size)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        #self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads,activation=nn.functional.gelu), num_layers)
        self.transformers = nn.ModuleList([])
        for _ in range(num_layers):
            self.transformers.append(MyTransformer(embed_size, num_heads,4*embed_size, dropout, nn.GELU()))
        self.norm = nn.LayerNorm(embed_size)
        self.fc = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.num_patches,self.patch_size**2)
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embedding
        x = self.dropout(x)
        #x = self.transformer(x)
        for transformer in self.transformers:
            x = transformer(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.fc(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
#model = ViT().to(device)
model = ViT()
#model.load_state_dict(torch.load('model.pth'))
model.to(device)
print(model)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss_batch_list = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss_batch_list.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')
    return sum(test_loss_batch_list) / len(test_loss_batch_list), correct / total
# 训练模型
def train(model, train_loader, criterion, optimizer, device, num_epochs=50, save_path='model.pth'):
    train_loss_batch_list = []
    train_acc_batch_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        running_loss = 0.0
        count = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}]')
        for images, labels in progress_bar:
            count += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_batch_list.append(loss.item())
            train_acc_batch_list.append((torch.argmax(outputs, 1) == labels).sum().item() / len(labels))
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_acc += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=running_loss / count, acc=train_acc / count / train_loader.batch_size)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
        a,b=test(model, test_loader, device)
        test_loss_list.append(a)
        test_acc_list.append(b)
    
    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')
    return train_loss_batch_list, train_acc_batch_list, test_loss_list, test_acc_list


# 运行训练和测试
train_loss_batch_list, train_acc_batch_list, test_loss_list, test_acc_list = train(model, train_loader, criterion, optimizer, device)
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

#test(model, test_loader, device)