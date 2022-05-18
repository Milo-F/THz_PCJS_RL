# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:38:55 2022
    基础网络
@author: milo
"""

import torch.nn as nn

class Network(nn.Module):
    
    # initial function    
    def __init__(self, in_dimension = 1, out_dimention = 3):
        super(Network, self).__init__()
        # layers defination
        self.layer_1 = nn.Linear(in_dimension, 32)
        self.layer_1.weight.data.normal_(0, 0.1)
        self.relu_1 = nn.LeakyReLU()
        self.layer_2 = nn.Linear(32, 32)
        self.layer_2.weight.data.normal_(0, 0.1)
        self.relu_2 = nn.LeakyReLU()
        self.layer_3 = nn.Linear(32, 32)
        self.layer_3.weight.data.normal_(0, 0.1)
        self.relu_3 = nn.LeakyReLU()
        self.out_layer = nn.Linear(32, out_dimention)
        self.out_layer.weight.data.normal_(0, 0.1)
        
    # forward spread function    
    def forward(self, s):
        x = self.layer_1(s)
        x = self.relu_1(x)
        x = self.layer_2(s)
        x = self.relu_2(x)
        x = self.layer_3(s)
        x = self.relu_3(x)
        x = self.out_layer(x)
        return x
    