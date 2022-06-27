#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Signal.py
    @Author: Milo
    @Date: 2022/06/23 14:59:54
    @Version: 1.0
    @Description: 2D环境的定位信号
'''

import numpy as np
from enviroment import Config as cfg

class Signal():
    
    def __init__(self, sd = 0) -> None:
        np.random.seed(sd)
        self.s = self._get_signal()
        pass

    # 每个子载波的发送信号对发送天线归一化
    def _get_signal(self):
        M = cfg.M2D # 天线数
        N = cfg.N # 子载波数
        s = np.zeros([M,N], dtype=np.complex64)
        for n_idx in range(N):
            for m_idx in range(M):
                s_nm = np.random.randn() + 1j*np.random.randn()
                s[m_idx,n_idx]=s_nm
            s[:,n_idx] = s[:,n_idx]/np.linalg.norm(s[:,n_idx])            
        return s
