#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Signal.py
    @Author: Milo
    @Date: 2022/05/31 10:38:32
    @Version: 1.0
    @Description: 发送信号类
'''

import numpy as np
import config as cfg

class Signal():
    def __init__(self) -> None:
        np.random.seed(0)
        self.s_p = self.position_signal()
        self.S_p = self.position_signal_matrix()
        pass
    
    # 单一信号
    def position_signal(self):
        s = []
        for idx in range(cfg.M_X*cfg.M_Z):  
            # s.append(1)         
            s.append(np.random.randint(-2,3))
        s = np.mat(s).T
        # s = s/np.linalg.norm(s)
        return s
    
    # 多载波搭载不同定位信号
    def position_signal_matrix(self):
        S = np.zeros([cfg.M, cfg.N]) # 全部信号
        s_n = [] # 第n个子载波的信号
        for i_idx in range(cfg.N):
            s_n.clear()
            for j_idx in range(cfg.M):
                s_n.append(np.random.randint(1,5))
            S[:,i_idx]=s_n
        S = np.mat(S)
        return S