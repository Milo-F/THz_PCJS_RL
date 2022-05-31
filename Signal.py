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
        self.s_p = self.position_signal()
        pass
    
    def position_signal(self):
        s = []
        np.random.seed(0)
        for idx in range(cfg.M_X*cfg.M_Z):  
            # s.append(1)         
            s.append(np.random.randint(-2,3))
        s = np.mat(s).T
        # s = s/np.linalg.norm(s)
        return s