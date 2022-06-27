#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: DelayFactor.py
    @Author: Milo
    @Date: 2022/06/24 11:16:28
    @Version: 1.0
    @Description: 延迟因子
'''

from enviroment import Config as cfg
import cmath
import numpy as np

class DelayFactor():
    
    def __init__(self, tau) -> None:
        self.tau = tau
        self.idx = [x for x in range(cfg.N)]
        self.com_part = -1j*2*cmath.pi/(cfg.N*cfg.T_S)
        self.D, self.D_ = self._get_d()
        pass

    def _get_d(self):
        d = []
        d_ = []
        for i in self.idx:
            d_tmp = cmath.exp(self.com_part*i*self.tau)
            d.append(d_tmp)
            d_.append(self.com_part*i*d_tmp)
        D = self._vec2diag(d)
        D_ = self._vec2diag(d_)
        return D, D_
        
        
    def _vec2diag(self, vec):
        n = len(vec)
        diag = np.zeros([n,n], dtype=np.complex64)
        for i in range(n):
            diag[i,i] = vec[i]
        return diag