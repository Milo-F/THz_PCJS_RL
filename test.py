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

S = Signal.Signal().S_p

c = Crb.Crb([500, math.pi/3, math.pi/5], 10, 1e-7+1j*1e-7, S, 1e-10)

print(c.crb)



