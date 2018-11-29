# coding:utf-8
# Donny You
import os
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
                 self.dilation[i] * (self.kernel_size[i]-1) - 1) // self.stride[i] + 1 for i, s in enumerate(size)]

    def forward(self, x):
        b, c, h, w = x.size()
        unfold_x = F.unfold(x, kernel_size=self.kernel_size,
                            dilation=self.dilation, padding=self.padding, stride=self.stride)
        unfold_h, unfold_w = self._out_size([h, w])
        unfold_x = unfold_x.view(b, c, -1, unfold_h, unfold_w).contiguous()
        print('unfold_x: {}'.format(unfold_x.size()))
        center_index = unfold_x.size(2) // 2
        center_x = unfold_x[:, :, center_index:center_index+1].contiguous()
        print('center_x: {}'.format(center_x.size()))
        sim_mat = (unfold_x * center_x).sum(1, keepdim=True)
        print('sim_mat: {}'.format(sim_mat.size()))
        sim_mat = F.softmax(sim_mat, 2)
        out = sim_mat * unfold_x
        return out.sum(2).contiguous()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    in_x = torch.randn((2, 24, 512, 1024)).cuda()
    self_attention = SelfAttentionModule(kernel_size=3, dilation=1, padding=1)
    import time
    for i in range(1000):
        start_time = time.time()
        out_x = self_attention(in_x)
        print(time.time() - start_time)

    print(out_x.size())
