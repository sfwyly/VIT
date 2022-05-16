
import torch
import torch.nn as nn


class Embeddings(nn.Module):

    def __init__(self, img_size=224, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = img_size
        patch_size = 16
        hidden_size = 768
        # 将图片分割成多少块 (224/16=14)
        n_patches = (img_size//patch_size)*(img_size//patch_size)
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size)
        # 设置可学习的位置拜编码信息 （1, 196+1, 786）
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, hidden_size))
        # 设置可学习的分类信息维度
        self.classifer_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        bs = x.shape[0]
        cls_token = self.classifer_token.expand(bs, -1, -1)  # bs 1 768
        x = self.patch_embeddings(x)  # bs 768 14 14
        x = x.flatten(2)  # bs 768 196
        x = x.transpose(-1, -2)
        x = torch.cat([cls_token, x], dim=1)  # bs 197 768
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


# inp = torch.randn((4, 3, 224, 224))
# print(Embeddings()(inp).shape)