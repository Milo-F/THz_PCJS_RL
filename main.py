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
from Constraints import Constraints
from networks import DDPG
import Config as cfg
import RLEnv
import numpy as np
import torch

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
    snr = 15 # 信噪比
    env = RLEnv.RLEnv(position, snr)
    # 初始化功率分配
    var = 1
    rate_list = []
    p_list = [0.7, 0.2]
    s, rate, reward = env.step(p_list)
    for epoch in range(epoch_total):
        # 选择动作
        s_tensor = torch.Tensor(s)
        action = ddpg.choose_action(s_tensor)
        action = np.clip(np.random.normal(action, var), 0.0001, 0.9999)
        p_list[0] = action[0]*Constraints().beta_p
        p_list[1] = action[1]*Constraints().beta_c
        # p_list[1] = Constraints().p_total-p_list[0]
        # p_list[1] = np.clip(np.random.normal(p_list[1], var), 0.0001, Constraints().beta_c)
        s_, rate, reward = env.step(p_list)
        # 保存经验到经验池
        ddpg.mem.store_trans(s, action, reward, s_)
        if ddpg.mem.mem_cnt > mem_deepth:
            var = var*0.995
            ddpg.learn()
        # 更新状态
        s = s_
        
        rate_list.append(rate)
        print(p_list, rate)
    
    pt = ddpg.cost_c
    x = [x for x in range(len(pt))]   
    plt.plot(x, pt)
    plt.show()
    


if __name__ == "__main__":
    main()
