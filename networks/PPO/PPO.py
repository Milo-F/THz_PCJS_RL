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
from networks.PPO import PPO_CNet
from networks.PPO import PPO_ANET


class PPO():
    def __init__(
        self,
        state_dim,  # 状态维度
        action_dim,  # 动作维度
        a_lr,  # 动作网络学习率
        c_lr,  # 评估网络学习率
        eps,
        epsilon,  # PPO2重要性裁剪系数
        actor_update_step,
        critic_update_step
    ) -> None:
        # 超参数
        self.eps = eps
        self.epsilon = epsilon
        self.actor_update_step = actor_update_step
        self.critic_update_step = critic_update_step
        # 构建critic网络
        self.c_net = PPO_CNet.PPO_CNet(state_dim, 1, hidden_1=64, hidden_2=32)
        self.c_optmizer = torch.optim.Adam(
            self.c_net.parameters(), lr=c_lr, weight_decay=0.00001)

        # 构建actor网络
        self.a_net = PPO_ANET.PPO_ANET(
            state_dim, action_dim, hidden_1=128, hidden_2=64)
        self.a_optmizer = torch.optim.Adam(
            self.a_net.parameters(), lr=a_lr, weight_decay=0.00001)

        # 构建actor_old网络
        self.a_net_old = PPO_ANET.PPO_ANET(
            state_dim, action_dim, hidden_1=128, hidden_2=64)
        self.a_net_old.load_state_dict(
            self.a_net.state_dict())  # 新动作网络参数复制到旧动作网络
        pass

    def a_train(self, state, action, advantage):
        '''训练actor网络'''
        mu, std = self.a_net(state)
        conv = torch.diag_embed(std)
        pi = torch.distributions.MultivariateNormal(mu, conv)

        mu_old, std_old = self.a_net_old(state)
        conv_old = torch.diag_embed(std_old)
        pi_old = torch.distributions.MultivariateNormal(mu_old, conv_old)

        # 重要性比例
        ratio = pi.log_prob(action).exp() / (pi_old.log_prob(action).exp()+self.eps)
        # ratio = torch.exp(log_ratio)
        surr_1 = ratio * advantage
        surr_2 = torch.clamp(surr_1, 1-self.epsilon,
                             1+self.epsilon) * advantage

        a_loss = -torch.mean(torch.minimum(surr_1, surr_2))
        self.a_optmizer.zero_grad()
        a_loss.backward()
        self.a_optmizer.step()

    def c_train(self, R, state):
        '''训练critic网络'''
        v = self.c_net(state)
        adv = R-v
        c_loss = torch.mean(torch.square(adv))
        self.c_optmizer.zero_grad()
        c_loss.backward()
        self.c_optmizer.step()

    def cal_adv(self, R, state):
        '''计算advantage'''
        with torch.no_grad():
            advantage = R - self.c_net(state)
            return advantage

    def update_old_actor(self):
        '''更新旧的策略'''
        self.a_net_old.load_state_dict(self.a_net.state_dict())

    def update(self, state, action, R):
        '''更新PPO网络'''
        self.update_old_actor()
        advantage = self.cal_adv(R, state)
        for _ in range(self.actor_update_step):
            self.a_train(state, action, advantage)

        for _ in range(self.critic_update_step):
            self.c_train(R, state)

    def choose_action(self, state):
        '''动作选择'''
        with torch.no_grad():
            mu, std = self.a_net(state)
            conv = torch.diag_embed(std)
            pi = torch.distributions.MultivariateNormal(mu, conv)
            action = torch.clamp(pi.sample(), 0.0001, 1)
            return action

    def get_v(self, state):
        '''获取状态的价值'''
        with torch.no_grad():
            v = self.c_net(state)
            return v
