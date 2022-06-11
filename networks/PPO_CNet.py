#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: PPO_CNet.py
    @Author: Milo
    @Date: 2022/06/09 20:13:41
    @Version: 1.0
    @Description: 
'''

import torch.nn as nn
import torch.nn.functional as nn_f
import torch


# 动作网络
class PPO_CNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_1=128, hidden_2=64):
        super(PPO_CNet, self).__init__()
        # self.bn = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, hidden_1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(hidden_2, hidden_2)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_2, out_features)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s):
        # s = torch.batch_norm(s)
        x = self.fc1(s)
        x = nn_f.leaky_relu(x)
        x = self.fc2(x)
        x = nn_f.leaky_relu(x)
        x = self.fc3(x)
        x = nn_f.leaky_relu(x)
        x = self.out(x)
        return x
