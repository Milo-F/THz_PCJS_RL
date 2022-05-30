#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: DDPG.py
    @Author: Milo
    @Date: 2022/05/30 11:27:41
    @Version: 1.0
    @Description: DDPG网络
'''


from networks import ANet
from networks import CNet
import numpy as np
import torch.nn as nn
import torch


# DDPG网络定义
class DDPG():

    # 构造函数
    def __init__(self,
                 state_dim=1,  # 状态信息维度
                 action_dim=1,  # 动作信息维度
                 lr=0.002  # 学习率
                 ) -> None:
        # 学习率
        self.a_lr = lr
        self.c_lr = lr
        # 构造动作评估网络和动作目标网络
        self.a_net_eval = ANet(state_dim, action_dim)
        self.a_net_target = ANet(state_dim, action_dim)
        # 构造价值评估网络和价值动作网络
        self.c_net_eval = CNet(state_dim, action_dim)
        self.c_net_target = CNet(state_dim, 1)
        # 损失函数
        self.loss_fun = nn.MSELoss()
        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.a_net_eval.parameters(), lr=self.a_lr, weight_decay=0.00001)
        self.critic_optimizer = torch.optim.Adam(
            self.c_net_eval.parameters(), lr=self.c_lr, weight_decay=0.00001)

    # 动作选择函数
    def choose_action(self, ob):

        pass

    # 学习
    def learn(self):
        pass
