# -*- coding: utf-8 -*-
"""
Created on Sun May 29 15:13:57 2022
    经验池
@author: milo
"""

import numpy as np


class Mem():

    def __init__(self, mem_deepth=1000, mem_width=1) -> None:
        self.mem_cnt = 0
        self.mem_deepth = mem_deepth
        self.mem = np.zeros(mem_deepth, mem_width)
        pass

    # 经验存储函数
    def store_trans(self, s, a, r, s_) -> None:
        trans = np.hstack((s, a, r, s_))
        idx = self.mem_cnt % self.mem_deepth
        self.mem[idx, :] = trans
        self.mem_cnt += 1

    # 经验采样函数
    def sample(self, n):
        assert self.mem_cnt >= self.mem_deepth, "Memory has not been fulfilled"
        idxs = np.random.choice(self.mem_deepth, size=n)
        return self.mem[idxs, :]
