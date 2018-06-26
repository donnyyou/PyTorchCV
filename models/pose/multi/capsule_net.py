#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)


import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from utils.tools.configer import Configer as Config


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes,
                       in_channels, out_channels,
                       kernel_size=None, stride=None,
                       num_iterations=None):

        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules,
                                                          num_route_nodes,
                                                          in_channels,
                                                          out_channels))
        else:
            self.capsules = nn.ModuleList([nn.Conv2d(in_channels, out_channels,
                                           kernel_size=kernel_size, stride=stride,
                                           padding=0) for _ in range(num_capsules)])

    def __squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def __softmax(self, x, dim=1):
        transposed_input = x.transpose(dim, len(x.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(x.size()) - 1)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = torch.matmul(x[None, :, :, None, :], self.route_weights[:, None, :, :, :])
            # priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = self.__softmax(logits, dim=2)
                outputs = self.__squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.__squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.pose_capsules = CapsuleLayer(num_capsules=Config.get('num_keypoints'),
                                          num_route_node=Config.get('num_keypoints'),
                                          in_channels=Config.get('capsule', 'l_vec'),
                                          out_channels=1)

    def forward(self, x):
        x = self.pose_capsules(x)

        return x


if __name__ == "__main__":
    pass
