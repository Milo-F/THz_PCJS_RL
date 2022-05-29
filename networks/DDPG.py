# -*- coding: utf-8 -*-
"""
Created on Sun May 29 15:13:57 2022

@author: milo
"""

import torch
from torch import nn
from torch.nn import functional as nn_f

# 动作网络
class ANet(nn.Module):
    def __init__(self, in_features, out_features, hidden_1 = 32, hidden_2 = 8):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_1,hidden_2)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(hidden_2, hidden_2)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_2,out_features)
        self.out.weight.data.normal_(0, 0.1)
        
    def forward(self, s):
        x = self.fc1(s)
        x = nn_f.leaky_relu(x)
        x = self.fc2(x)
        x = nn_f.leaky_relu(x)
        x = self.fc3(x)
        x = nn_f.leaky_relu(x)
        x = self.out(x)
        return x

# 价值网络
class CNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_1 = 10, hidden_2 = 1):
        super(CNet, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(hidden_2, hidden_2)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_2,out_features)
        self.out.weight.data.normal_(0, 0.1)
        
        
    def forward(self, s, a):
        x = torch.cat([s,a],1)
        x = nn_f.leaky_relu(self.fc1(x))
        x = nn_f.leaky_relu(self.fc2(x))
        x = nn_f.leaky_relu(self.fc3(x))
        x = self.out(x)
        return x

class DDPG:
    def __init__(
            self, 
            num_actions, 
            num_features, 
            lr = 0.005, 
            reward_decay = 0.9, 
            e_greedy = 0.9,
            replace_target_iter = 300,
            memery_size = 1000,
            batch_size = 32,
            tau = 0.0001,
            e_greedy_increment=None
            ):
        pass