#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: PPO.py
    @Author: Milo
    @Date: 2022/06/09 16:55:34
    @Version: 1.0
    @Description: PPO网络
'''

import torch
from networks import PPO_CNet
from networks import PPO_ANET


class PPO():
    def __init__(
        self,
        state_dim, # 状态维度
        action_dim, # 动作维度
        a_lr, # 动作网络学习率
        c_lr, # 评估网络学习率
        eps, # 
        epsilon, # PPO2重要性裁剪系数
        actor_update_step, 
        critic_update_step 
        ) -> None:
        # 超参数
        self.eps = eps
        self.epsilon = epsilon
        self.actor_update_step = actor_update_step
        self.critic_update_step = critic_update_step
        # 构建critic网络
        self.c_net = PPO_CNet.PPO_CNet(state_dim, 1 ,hidden_1=64, hidden_2=32)
        self.c_optmizer = torch.optim.Adam(self.c_net.parameters(), lr=c_lr, weight_decay=0.00001)
        
        # 构建actor网络
        self.a_net = PPO_ANET.PPO_ANET(state_dim, action_dim, hidden_1=128, hidden_2=64)
        self.a_optmizer = torch.optim.Adam(self.a_net.parameters(), lr=a_lr, weight_decay=0.00001)
        
        # 构建actor_old网络
        self.a_net_old = PPO_ANET.PPO_ANET(state_dim, action_dim, hidden_1=128, hidden_2=64)
        self.a_net_old.load_state_dict(self.a_net.state_dict()) # 新动作网络参数复制到旧动作网络
        pass
    
    # 训练actor网络
    def a_train(self, state, action, advantage):        
        mu, std = self.a_net(state)
        conv = torch.diag_embed(std)
        pi = torch.distributions.MultivariateNormal(mu, conv)
        
        mu_old, std_old = self.a_net_old(state)
        conv_old = torch.diag_embed(std_old)
        pi_old = torch.distributions.MultivariateNormal(mu_old, conv_old)
        
        # 重要性比例
        log_ratio = pi.log_prob(action) - pi_old.log_prob(action)
        ratio = torch.exp(log_ratio)
        surr_1 = ratio * advantage
        surr_2 = torch.clamp(surr_1, 1-self.epsilon, 1+self.epsilon) * advantage
        
        a_loss = -torch.mean(torch.minimum(surr_1, surr_2))
        self.a_optmizer.zero_grad()
        a_loss.backward()
        self.a_optmizer.step()
            
    
    # 训练critic网络
    def c_train(self, R, state):
        v = self.c_net(state)
        adv = R-v
        c_loss = torch.mean(torch.square(adv))
        self.c_optmizer.zero_grad()
        c_loss.backward()
        self.c_optmizer.step()
    
    # 计算  
    def cal_adv(self, R, state):
        with torch.no_grad():
            advantage = R - self.c_net(state)
            return advantage
    
    # 更新旧的策略
    def update_old_actor(self):
        self.a_net_old.load_state_dict(self.a_net.state_dict())
    
    # 更新
    def update(self, state, action, R):
        self.update_old_actor()
        advantage = self.cal_adv(R, state)
        for _ in range(self.actor_update_step):
            self.a_train(state, action, advantage)
        
        for _ in range(self.critic_update_step):
            self.c_train(R, state)
        
    
    # 动作选择
    def choose_action(self, state):
        with torch.no_grad():
            mu, std = self.a_net(state)
            conv = torch.diag_embed(std)
            pi = torch.distributions.MultivariateNormal(mu, conv)
            action = torch.clamp(pi.sample(), 0, 1)
            return action
    
    def get_v(self, state):
        with torch.no_grad():
            v = self.c_net(state)
            return v