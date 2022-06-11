#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Env.py
    @Author: Milo
    @Date: 2022/05/30 16:35:48
    @Version: 1.0
    @Description: 强化学习的交互环境，包含通信速率，定位CRB等优化问题的目标与约束等
'''

from asyncio import constants
import Crb  # 克拉美劳界
import Signal  # 发送信号
import Config as cfg  # 配置固定参数
import ComChannelNoRobust as cchnr  # 通信信道
import Position  # 定位估计位置
import Constraints as cons  # 约束的边界值
import DirectionVec as DV  # 方向向量
import numpy.linalg as lg
import cmath
import math
import numpy as np


class RLEnv():
    state_dim = 32
    action_dim = 2
    # 构造函数
    def __init__(
        self,
        position,  # 位置
        sigma, # 背景噪声
        # snr_db=10  # 信噪比
    ) -> None:
        # 真实位置（直角坐标）
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        # 真实位置（球坐标）
        self.d, self.theta, self.phi = Position.car2sphe(
            self.x, self.y, self.z)
        # 路损
        self.alpha_loss = cfg.C/(4*cmath.pi*self.d*cfg.F_C)
        self.alpha_moss = math.exp(-0.5*cfg.KAPPA*self.d)
        self.alpha_random = np.random.randn()*cfg.ALPHA_SIGMA + 1j*np.random.randn()*cfg.ALPHA_SIGMA
        self.alpha = self.alpha_loss*self.alpha_moss + self.alpha_random
        # 信噪比
        self.sigma = sigma
        # self.snr_db = snr_db
        # self.snr = 10**(self.snr_db/10)
        # 定位信号矩阵
        self.S = Signal.Signal().S_p
        pass

    # 计算通信速率
    def solve_rate(self, p_sphe_hat, Delta_p_sphe, p_c, sigma):
        ch = [] # 列表用于存放通信信道
        for n in range(1, cfg.N+1):
            ch.append(cchnr.ComChannelNoRobust(n, p_sphe_hat, Delta_p_sphe, p_c, self.alpha).channel) # 产生不同子载波的通信信道
        ch = np.mat(ch)
        snr_c = lg.norm(ch)**2/(cfg.N*sigma**2) # 通信信噪比
        r_c = math.log2(1+snr_c)
        return ch, r_c

    # 计算奖励
    def solve_reward(self, crb_p, p_list, rate):
        p_p = p_list[0]
        p_c = p_list[1]
        reward = 0
        if np.sqrt(crb_p[0,0]) <= cons.RHO and p_list[0] + p_list[1] <= cons.P_TOTAL:
            reward = rate
        else:
            # print("voilate")
            # reward = -rate*100
            reward = 0
        return reward
    
    # 按信噪比求解噪声
    # def get_sigma(self, p_p):
    #     # 构建定位信道
    #     dv = DV.DirectionVec(self.theta, self.phi, 0)
    #     a = dv.a
    #     W = Position.vec2diag(a)
    #     X = (W*self.S).H
    #     # 根据信噪比产生定位噪声和通信噪声
    #     sigma = math.sqrt((p_p*abs(self.alpha)**2*lg.norm(X*a)**2)/(cfg.N*self.snr))
    #     return sigma
    
    # 环境交互函数，输入当前状态以及选择的功率分配，得
    # 到该功率分配下的下一个时刻的信道状态以及选择该功
    # 率分配下下一个时刻的通信速率（奖励）
    def step(self, p_list):
        p_p = p_list[0] # i帧得到的定位功率
        p_c = p_list[1] # i帧得到的通信功率
        p_sphe = [self.d, self.theta, self.phi] # 真实位置
        # 根据给定的信噪比求i+1帧的噪声标准差sigma
        sigma = self.sigma
        # 求选择当前功率分配下的CRB
        crb = Crb.Crb(p_sphe, p_p, self.alpha, self.S, sigma).crb  # 解算CRB（通过真实位置解算的）
        # crb = Crb.Crb(p_p, alpha, self.S, sigma) # 作用范围内的采样平均CRB
        crb_p = crb[0:3, 0:3] # 取得位置有关的误差
        # p_sphe_hat = Position.get_position_hat(crb_p, p_sphe) # 根据crb获得下一帧估计位置
        p_sphe_hat = p_sphe
        # print(p_sphe_hat)
        Delta_p_sphe = [math.sqrt(crb_p[0,0]), math.sqrt(crb_p[1,1]), math.sqrt(crb_p[2,2])] # 位置相关的误差
        s_, rate = self.solve_rate(p_sphe_hat, Delta_p_sphe, p_c, sigma) # i+1帧的观测CIS以及通信速率
        reward = self.solve_reward(crb_p, p_list, rate)
        s_r = np.real(s_)
        s_i = np.imag(s_)
        s = np.hstack([s_r, s_i])
        s = np.array(s)
        s = np.squeeze(s)
        return s, rate, reward, crb_p

    def reset(self):
        s, ra, re, _ = self.step([5, 5])
        return s