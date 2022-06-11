#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: DDPG.py
    @Author: Milo
    @Date: 2022/05/30 11:27:41
    @Version: 1.0
    @Description: DDPG网络
'''


from networks.ANet import ANet
from networks.CNet import CNet
import numpy as np
import torch
from networks.Mem import Mem


# DDPG网络定义
class DDPG():

    # 构造函数
    def __init__(self,
                 state_dim=1,  # 状态信息维度
                 action_dim=1,  # 动作信息维度
                 batch_size = 64, # 批大小
                 reward_decay = 0,   # 奖励折扣
                 mem_deepth = 1000,
                 lr=0.002  # 学习率
                 ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_decay = reward_decay
        # 经验池
        self.mem = Mem(mem_deepth=mem_deepth, mem_width=self.state_dim*2+self.action_dim+1)
        # 学习率
        self.a_lr = lr
        self.c_lr = lr
        # 构造动作评估网络和动作目标网络
        self.a_net_eval = ANet(self.state_dim, self.action_dim)
        self.a_net_target = ANet(self.state_dim, self.action_dim)
        # 构造价值评估网络和价值动作网络
        self.c_net_eval = CNet(self.state_dim + self.action_dim, 1)
        self.c_net_target = CNet(self.state_dim + self.action_dim, 1)
        # 损失函数
        self.loss_fun = torch.nn.MSELoss()
        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.a_net_eval.parameters(), lr=self.a_lr, weight_decay=0.00001)
        self.critic_optimizer = torch.optim.Adam(
            self.c_net_eval.parameters(), lr=self.c_lr, weight_decay=0.00001)
        # 学习迭代
        self.batch_size = batch_size
        # loss保存
        self.cost_c = []
        self.cost_a = []
        # 学习计数器
        self.learn_step = 0
        
        
    # 动作选择函数
    def choose_action(self, ob):
        ob = ob * 10 ** 8
        ob = torch.Tensor(ob[np.newaxis, :])
        action = self.a_net_eval(ob).data.numpy()

        action = np.squeeze(action)
        return action

    # 学习(前提为经验池存满)
    def learn(self):
        # 从评估网络更新目标网络
        update_factor = 0.1  # 更新系数
        for x in self.a_net_target.state_dict().keys():
            eval('self.a_net_target.' + x + '.data.mul_((1-update_factor))')
            eval('self.a_net_target.' + x +
                 '.data.add_(update_factor*self.a_net_eval.' + x + '.data)')
        for x in self.c_net_target.state_dict().keys():
            eval('self.c_net_target.' + x + '.data.mul_((1-update_factor))')
            eval('self.c_net_target.' + x +
                 '.data.add_(update_factor*self.c_net_eval.' + x + '.data)')
        # 从经验池采样
        batch_mem = self.mem.sample(self.batch_size)
        # 提取
        batch_s, batch_a, batch_r, batch_s_ = self.mem.extract(batch_mem, self.state_dim, self.action_dim)
        # 正向传播    
        a = self.a_net_eval(batch_s)
        q = self.c_net_eval(batch_s, a)        
        # 反向传播更新参数
        # print("q:",q)
        action_loss = - torch.mean(q)
        self.actor_optimizer.zero_grad()
        action_loss.backward()
        self.actor_optimizer.step()        
        # 目标网络
        a_target = self.a_net_target(batch_s_)
        q_tmp = self.c_net_target(batch_s_, a_target)
        
        q_target = batch_r + self.reward_decay * q_tmp
        # print(q_target, q_tmp)
        q_eval = self.c_net_eval(batch_s, batch_a)
        # print(q_target, q_eval)        
        # 价值网络反向传播与更新
        td_error = self.loss_fun(q_target, q_eval)
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()        
        # 保存loss
        self.cost_a.append(action_loss.item())
        self.cost_c.append(td_error.item())        
        self.learn_step += 1
        
        
        
        
        