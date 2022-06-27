#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: DirectVector2D.py
    @Author: Milo
    @Date: 2022/06/23 15:27:19
    @Version: 1.0
    @Description: 均匀线阵的方向向量
'''

from enviroment import Config as cfg
import cmath
import numpy as np

class DirectVector2D():
    
    # theta为角度， n为子载波索引
    def __init__(self, theta, n) -> None:
        # 索引
        self.idx = [x for x in range(-int((cfg.M2D-1)/2), int((cfg.M2D-1)/2)+1)]
        # 指数公共部分
        pi = cmath.pi
        self.com_part = 1j*2*pi*(cfg.F_C + n/cfg.N/cfg.T_S)*cfg.D_0/cfg.C
        self.var = self.com_part * cmath.cos(theta)
        self.var_div = -self.com_part * cmath.sin(theta)
        
        self.a, self.b = self._get_a()
        pass

    def _get_a(self):
        a = []
        b = []
        for i_idx in self.idx:
            a_tmp = (1/cmath.sqrt(cfg.M2D)) * cmath.exp(self.var * i_idx)
            b_tmp = self.var_div * i_idx * a_tmp
            a.append(a_tmp)
            b.append(b_tmp)
        a_vec = np.mat(a).T
        b_vec = np.mat(b).T
        return a_vec, b_vec
    
    
