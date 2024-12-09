import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=5, dim=768, depth=6, heads=12, mlp_dim=3072, dropout=0.1, emb_dropout=0.1):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # 3 channels for RGB

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True,activation='gelu')
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.num_patches, -1)  # Flatten patches
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embedding
        x = self.dropout(x)
        for transformer in self.transformers:
            x = transformer(x)
        x = x[:, 0]  # Use the class token
        x = self.norm(x)
        x = self.fc(x)
        #x = nn.functional.log_softmax(x, dim=1)
        return x

Model = ViT

if __name__ == '__main__':
    model = Model()
    print(model)
    input = torch.randn(2, 3, 224, 224)  # 2 samples, 3 channels, 224x224 image size
    output = model(input)
    print(output.shape)