#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: main.py
    @Author: Milo
    @Date: 2022/06/01 15:28:00
    @Version: 1.0
    @Description: 顶层运行文件
'''

from asyncio import constants
from math import ceil
import matplotlib.pyplot as plt
from ComChannelNoRobust import ComChannelNoRobust
from Constraints import Constraints
from networks import DDPG
import Config as cfg
import RLEnv
import numpy as np
import torch
import Crb
import math

def print_log(str):
    s_tmp = (40-ceil(len(str)/2))
    e_tmp = s_tmp
    if (len(str)%2):
        e_tmp = e_tmp +1
    print("="*s_tmp + str + "="*e_tmp)

def main():
    # 设置随机种子
    np.random.seed(0)
    torch.manual_seed(0)
    #
    print_log("START")
    # 初始化DDPG网络
    print_log("INITIAL DDPG")
    # DDPG网络配置
    state_dim = cfg.N*2
    action_dim = 2
    batch_size= 32
    epoch_total = 3000
    reward_decay = 0
    lr = 0.002
    mem_deepth = 1000
    ddpg = DDPG.DDPG(state_dim, action_dim, batch_size, reward_decay, mem_deepth, lr)    
    # 初始化环境
    print_log("INITIAL ENVIROMENT")
    position = [100, 120, -10] # 位置
    sigma = 1e-8
    env = RLEnv.RLEnv(position, sigma)
    # 初始化功率分配
    var = 1
    rate_list = []
    p_list = [10, 2]
    s, rate, reward, crb_p = env.step(p_list)
    crb = Crb.Crb(position, p_list[0], 1e-8, env.S, sigma)
    print(crb.crb_diag_sqrt)
    print(rate)
    print(crb.crb)
    p_she = [3e-8, math.pi/5, math.pi/4]
    ch_1 = ComChannelNoRobust(0, p_she, [0,0,0], p_list[1], 1).channel
    ch_2 = ComChannelNoRobust(0, p_she, crb.crb_diag_sqrt[0:3], p_list[1], 1)
    print(abs(ch_1), abs(ch_2.channel))
    print(abs(ch_2.beamforming.H*ch_2.direct_a)/9)
    


if __name__ == "__main__":
    print('abcdefcdghcd'.split('cd',0))
    main()
    
    

