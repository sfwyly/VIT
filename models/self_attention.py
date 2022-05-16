
"""
    @Description: self_attention
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
from models.embed import Embeddings
# bs = 4
# query_layer = torch.randn((bs, 32, 64))
# key_layer = torch.randn((bs, 32, 64))
# value_layer = torch.randn((bs, 32, 64))
# dims = 32*32
#
# # 1. q k
# attention_scores = torch.matmul(query_layer, key_layer.permute(0, 2, 1))
# # 2. 得分情况进行softmax
# attention_scores = attention_scores / np.sqrt(dims)
#
# attention_probs = torch.softmax(attention_scores, dim=-1)
#
# # 3. 将概率与value详细进行相乘
# context_layer = torch.matmul(attention_probs, value_layer)
#
# print(context_layer.shape)


class Attention(nn.Module):

    def __init__(self, vis=True):
        super(Attention, self).__init__()
        self.vis = vis
        hidden_size = 768
        self.num_attention_heads = 12
        self.attention_head_size = hidden_size // self.num_attention_heads  # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(768, 768)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(0.5)
        self.proj_dropout = nn.Dropout(0.5)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # bs 197
        return x.permute(0, 2, 1, 3)  # bs 12 197 64

    def forward(self, hidden_states):
        # hidden_states 为 (bs 197 768)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # bs 12 197 197
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        weights = attention_probs if self.vis else None

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


# if __name__ == '__main__':
#     inp = torch.randn((4, 197, 768))
#     out_attn, weights = Attention()(inp)
#     print(out_attn.shape, weights.shape)

class Mlp(nn.Module):

    def __init__(self):
        super(Mlp, self).__init__()
        hidden_size = 768
        mlp_dim = 3072
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = F.gelu
        self.dropout = nn.Dropout(0.5)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        hidden_size = 768
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.ffn = Mlp()
        self.attn = Attention()

    def forward(self, x):

        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):

    def __init__(self, vis=True):
        super(Encoder, self).__init__()
        hidden_size = 768
        self.vis=vis
        self.layers = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        num_layers = 3
        for _ in range(num_layers):
            layer = Block()
            self.layers.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layers:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, img_size, vis=True):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size)
        self.encoder = Encoder(vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights  # bs 197 768


class VisionTransformer(nn.Module):
    def __init__(self, img_size=256, num_classes=51, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        hidden_size = 768
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = None
        self.transformer = Transformer(img_size, vis)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = F.cross_entropy
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights
if __name__=='__main__':
    transformer = VisionTransformer(img_size=256)
    logits, _ = transformer(torch.randn(4,3,256,256))
    print(logits.shape)
