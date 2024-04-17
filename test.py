# 手写MultiHead_Attention
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHead_Attention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, numheads=8):
        super(MultiHead_Attention, self).__init__()
        # 定义类属性
        self.dim_in =  dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = numheads
        # 将输入空间映射到QKV空间
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)

    def forward(self, x):
        # x.shape = (b,n,dim_in) batchsize, number_of_word dim_of_word
        b, n, dim_in = x.shape
        assert dim_in == self.dim_in
        num_heads = self.num_heads
        dim_k = self.dim_k
        dim_v = self.dim_v
        # 计算每个head上的dim
        dim_per_head_k = dim_k // num_heads
        dim_per_head_v = dim_v // num_heads
        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)

        # reshape成带多头qkv
        q = Q.reshape(b,num_heads,n,dim_per_head_k)
        k = K.reshape(b,num_heads,n,dim_per_head_k)
        v = V.reshape(b,num_heads,n,dim_per_head_v)

        scale = math.sqrt(dim_k)
        scores_init = torch.matmul(q,k.transpose(-1,-2) / scale) # (b,num_heads, n,n)
        score_final = F.softmax(scores_init,dim=-1)

        attention = torch.matmul(score_final, v)

        output = attention.transpose(1,2).reshape(b,n,dim_k)

        return output



attention = MultiHead_Attention(3,64,64,8)

input = torch.randn(64,50,3)
output = attention(input)
print(output.shape)
