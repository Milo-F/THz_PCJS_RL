# -*- coding: utf-8 -*-
"""
Created on Sun May 29 15:13:57 2022
    ANet网络
@author: milo
"""

import torch.nn as nn
import torch.nn.functional as nn_f
import torch


# 动作网络
class ANet(nn.Module):
    def __init__(self, in_features, out_features, hidden_1=32, hidden_2=8):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(hidden_2, hidden_2)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_2, out_features)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s):
        x = self.fc1(s)
        x = nn_f.leaky_relu(x)
        x = self.fc2(x)
        x = nn_f.leaky_relu(x)
        x = self.fc3(x)
        x = nn_f.leaky_relu(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x
