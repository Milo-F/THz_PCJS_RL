#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: PPO_ANET.py
    @Author: Milo
    @Date: 2022/06/09 18:43:44
    @Version: 1.0
    @Description: PPO_ANET
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPO_ANET(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_1=128, hidden_2=64) -> None:
        super(PPO_ANET, self).__init__()
        # self.bn = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_dim, hidden_1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc2.weight.data.normal_(0, 0.1)
        self.f_mean_1 = nn.Linear(hidden_2, hidden_1)
        self.f_mean_2 = nn.Linear(hidden_1, out_dim)
        self.f_var_1 = nn.Linear(hidden_2, hidden_1)
        self.f_var_2 = nn.Linear(hidden_1, out_dim)

    def forward(self, s):
        x = self.fc1(s)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)

        x_mean = self.f_mean_1(x)
        x_mean = F.leaky_relu(x_mean)
        x_mean = self.f_mean_2(x_mean)
        x_mean = torch.sigmoid(x_mean)

        x_var = self.f_var_1(x)
        x_var = F.leaky_relu(x_var)
        x_var = self.f_var_2(x_var)
        x_var = F.softplus(x_var)
        return x_mean, x_var
