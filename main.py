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
    batch_size= 256
    epoch_total = 30000
    reward_decay = 0.9
    lr = 0.0005
    mem_deepth = 10000
    ddpg = DDPG.DDPG(state_dim, action_dim, batch_size, reward_decay, mem_deepth, lr)    
    # 初始化环境
    print_log("INITIAL ENVIROMENT")
    position = [100, 120, -10] # 位置
    sigma = 1e-8 # 噪声
    env = RLEnv.RLEnv(position, sigma)
    # 初始化功率分配
    var = 0.4
    rate_list = []
    reward_list = []
    reward_plot = []
    p_list = [2, 8]
    s, rate, reward, crb = env.step(p_list)
    print(rate)
    for epoch in range(epoch_total):
        # 选择动作
        s_tensor = torch.Tensor(s)
        # if ddpg.mem.mem_cnt > 3*mem_deepth:
        action = ddpg.choose_action(s_tensor)
        action = np.clip(np.random.normal(action, var), 0.0001, 1)
        # else:
        #     action = np.random.random(2)
        p_list[0] = action[0]*Constraints().beta_p
        p_list[1] = action[1]*Constraints().beta_c
        # p_list[1] = Constraints().p_total-p_list[0]
        # p_list[1] = np.clip(np.random.normal(p_list[1], var), 0.0001, Constraints().beta_c)
        s_, rate, reward, crb = env.step(p_list)
        # 保存经验到经验池
        ddpg.mem.store_trans(s, action, reward/15, s_)
        if ddpg.mem.mem_cnt > mem_deepth:
            var = var*0.999
            ddpg.learn()
        # 更新状态
        s = s_
        
        rate_list.append(rate)
        reward_list.append(reward)
        epoch_len = 100
        if epoch%epoch_len == 0:
            print("%-50s" % (str(p_list)), end=' ')
            print(sum(reward_list)/epoch_len)
            reward_plot.append(sum(reward_list)/epoch_len)
            reward_list.clear()
        # print(p_list, rate)
    pt = ddpg.cost_c
    x = [x for x in range(len(pt))]   
    plt.plot(x, pt)
    plt.savefig("./ddpg_cnet_loss.jpg")
    plt.show()
    


if __name__ == "__main__":
    main()
