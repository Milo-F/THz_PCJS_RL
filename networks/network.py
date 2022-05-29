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
        
        self.layer_1 = nn.Linear(in_dimension, 64)
        nn.init.normal_(self.layer_1.weight, 0, 1)
        self.relu_1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(64)
        
        self.layer_2 = nn.Linear(64, 128)
        nn.init.normal_(self.layer_2.weight, 0, 1)
        self.relu_2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(128)
        
        self.layer_3 = nn.Linear(128, 64)
        nn.init.normal_(self.layer_3.weight, 0, 1)
        self.relu_3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(64)
        
        self.layer_4 = nn.Linear(64, 32)
        nn.init.normal_(self.layer_4.weight, 0, 1)
        self.relu_4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm1d(32)
        
        self.out_layer = nn.Linear(32, out_dimention)
        nn.init.normal_(self.out_layer.weight, 0, 1)
        
    # forward spread function    
    def forward(self, s):
        x = self.layer_1(s)
        x = self.bn1(x)
        x = self.relu_1(x)  
        x = self.layer_2(x)
        x = self.bn2(x)
        x = self.relu_2(x)
        x = self.layer_3(x)
        x = self.bn3(x)
        x = self.relu_3(x)
        x = self.layer_4(x)
        x = self.bn4(x)
        x = self.relu_4(x)
        x = self.out_layer(x)
        return x
     