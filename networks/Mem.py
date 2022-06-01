#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Mem.py
    @Author: Milo
    @Date: 2022/05/30 11:27:08
    @Version: 1.0
    @Description: 经验池
'''


import numpy as np
import torch


class Mem():

    def __init__(self, mem_deepth=1000, mem_width=1) -> None:
        self.mem_cnt = 0
        self.mem_deepth = mem_deepth
        self.mem = np.zeros([mem_deepth, mem_width])
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
    
    # 提取
    def extract(self, batch_mem, state_dim, action_dim):
        s = torch.FloatTensor(batch_mem[:,:state_dim])
        a = torch.FloatTensor(batch_mem[:, state_dim:state_dim + action_dim])
        r = torch.FloatTensor(batch_mem[:, -state_dim-1: -state_dim])
        s_ = torch.FloatTensor(batch_mem[:, -state_dim:])
        return s, a, r, s_
