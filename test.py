#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File:test.py
    @Author:Milo
    @Date:2022/05/30 11:23:28
    @Version:1.0
    @Description: 局部测试
'''

import Crb
import math
import Signal
import Config as cfg

s = Signal.Signal().S_p
p = [200, math.pi/5, math.pi/4]
alpha = cfg.C/(4*math.pi*p[0]*cfg.F_C)
crb = Crb.Crb(p, 80, alpha, s, 6e-9).crb
print(crb)
print(math.sqrt(crb[0,0])*cfg.C, math.sqrt(crb[1,1])*p[0], math.sqrt(crb[2,2])*p[0])



