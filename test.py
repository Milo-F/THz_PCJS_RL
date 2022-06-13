#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: test_1.py
    @Author: Milo
    @Date: 2022/06/11 11:19:10
    @Version: 1.0
    @Description: 
'''

import math
from multiprocessing import Event
import torch
import time
import Tools
from enviroment.DirectionVec import DirectionVec as DV
from enviroment.Env import Env

theta = math.pi/5
phi = math.pi/3
a = DV(theta, phi, 0).a
b = DV(theta, phi, 0).a
print(a.H*b)

p = [0.4,0.6]
position = [200, 320, -10]
sigma = 1e-9
env = Env(position, sigma)
_, _, rate = env.step(p)
print(rate)
# # today = time.date.ctime()
# tag = str(time.strftime("%m%d%H%M"))
# print(tag)


# c = 0.3,0.4
# b = []
# b = [a, c]
# print(b)
# tensor_a = torch.tensor(a, dtype=torch.float32)
# tensor_b = torch.tensor(b, dtype=torch.float32)
# print(torch.diag_embed(tensor_b))
# print(tensor_b)