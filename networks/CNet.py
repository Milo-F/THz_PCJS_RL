#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File:CNet.py
    @Author:Milo
    @Date:2022/05/30 11:28:19
    @Version:1.0
    @Description: 价值网络
'''



import torch.nn as nn
import torch.nn.functional as nn_f
import torch


# 价值网络
class CNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_1=10, hidden_2=1):
        super(CNet, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(hidden_2, hidden_2)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_2, out_features)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = nn_f.leaky_relu(self.fc1(x))
        x = nn_f.leaky_relu(self.fc2(x))
        x = nn_f.leaky_relu(self.fc3(x))
        x = self.out(x)
        return x
