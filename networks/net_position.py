# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:37:22 2022
    定位网络
@author: milo
"""

import network
import config as cfg

class PositionNet (network.Network):
    
    # initial function
    def __init__(self):
        self.layers = super(PositionNet, self).__init__(in_dimension=cfg.N+3, out_dimention=4)
    # forward spread function
    def forward(self, s):
        return self.layers.forward(s)
