# coding:utf-8
# Donny You
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat


class SelfAttentionModule(nn.Module):

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super(SelfAttentionModule, self).__init__()
        self.kernel_size = self.pair(kernel_size)
        self.dilation = self.pair(dilation)
        self.padding = self.pair(padding)
        self.stride = self.pair(stride)

    @staticmethod
    def pair(x):
        if isinstance(x, (list, tuple)):
            return x
        return tuple(repeat(x, 2))

    def _out_size(self, size):
        return [(s + 2 * self.padding[i] -
                 self.dilation[i] * (self.kernel_size[i]-1) - 1) // self.stride[i] for i, s in enumerate(size)]

    def forward(self, x):
        b, c, h, w = x.size()
        unfold_x = F.unfold(x, kernel_size=self.kernel_size,
                            dilation=self.dilation, padding=self.padding, stride=self.stride)
        unfold_h, unfold_w = self._out_size([h, w])
        unfold_x = unfold_x.view(b, c, -1, unfold_h, unfold_w).permute(0, 2, 3, 4, 1).contiguous()
        center_index = unfold_x.size(1) // 2
        center_x = unfold_x[:, center_index:center_index+1].contiguous()
        sim_mat = torch.matmul(unfold_x, center_x)
        sim_mat = F.softmax(sim_mat, 1)
        out = sim_mat.unsqueeze(-1) * unfold_x
        return out.sum(2).contiguous().permute(0, 3, 1, 2)


if __name__ == "__main__":
    in_x = torch.randn((2, 1024, 1024, 2048)).cuda()
    self_attention = SelfAttentionModule(kernel_size=3, dilation=1, padding=1)
    out_x = self_attention(in_x)
    print(out_x.size())
