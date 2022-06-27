#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Env.py
    @Author: Milo
    @Date: 2022/06/19 15:38:47
    @Version: 1.0
    @Description: 环境
'''

import numpy as np
import math
import cmath

from enviroment import Config as cfg 
from enviroment.env2d.Signal import Signal
from enviroment import Constraints as cons
from enviroment.env2d.DirectVector2D import DirectVector2D as DV2D
from enviroment.env2d.Crb2D import Crb2D
from enviroment.env2d.DelayFactor import DelayFactor as DF

class Env2D():
    state_dim = 2
    action_dim = 2
    
    rate_eff = 0
    p_error = 0
    
    def __init__(
        self,
        position:list, # [tau, theta]
        sigma
        ) -> None:      
        self.theta = position[1]
        self.d = position[0]
        self.tau = self.d/cfg.C
        self.position = [self.tau, self.theta]
        # 噪声
        self.sigma = sigma
        # 路损
        alpha_loss = cfg.C/(4*math.pi*self.d*cfg.F_C)
        alpha_moss = math.exp(-0.5*cfg.KAPPA*self.d)
        alpha_random = np.random.randn()*cfg.ALPHA_SIGMA + 1j * np.random.randn()*cfg.ALPHA_SIGMA
        self.alpha = alpha_loss*alpha_moss+alpha_random
        # 信号 M*N
        self.s = Signal(0).s
        self.x = np.zeros(np.shape(self.s),dtype=np.complex64)
        # 初始全向波束赋形
        self.w = np.ones([cfg.M2D,1], dtype=np.complex64)/cmath.sqrt(cfg.M2D)
        # self.w = DV2D(self.theta, 0).a
        
    def _get_snr_p(self, p_p, a):
        X = np.mat(self.x).T
        snr_p = (p_p*abs(self.alpha)**2*np.linalg.norm(X*a)**2)/self.sigma**2
        return snr_p
    
    def _get_snr_c(self, p_c, a):
        snr_c = (p_c*abs(self.alpha)**2*np.linalg.norm(self.w.H*a)**2)/self.sigma**2
        return snr_c
    
    def _get_rate(self, snr_c, p_sum):
        rate_eff = math.log2(1+snr_c)
        return rate_eff
    
    def _get_reward(self, rate_eff, p_sum, p_error):
        reward = 0
        if p_sum <= cons.P_TOTAL and p_error <= cons.RHO:
            reward = rate_eff
        else:
            reward = 0
        return reward
    
    def reset(self):
        p_p = 0.3
        p_c = 0.3
        state_init, _ = self.step([p_p, p_c])
        return state_init
    
    def step(self, action):
        # 获得定位功率和通信功率
        p_p = action[0]*cons.BETA_P
        p_c = action[1]*cons.BETA_C
        # 获得波束赋形后的定位信号x
        for i_idx in range(cfg.N):
            for j_idx in range(cfg.M2D):
                self.x[j_idx, i_idx] = self.w[j_idx] * self.s[j_idx, i_idx]
        # 获得方向向量a
        dv2d = DV2D(self.theta, 0)
        a = dv2d.a
        a_ = dv2d.b
        # 获得延迟因子
        df = DF(self.tau)
        D = df.D
        D_ = df.D_
        # 计算CRB
        crb2d = Crb2D(a,a_, D, D_, p_p, self.alpha, self.x, self.sigma)
        crb = crb2d.crb
        crb_diag = crb2d.crb_diag_sqrt
        # 获得定位信噪比
        snr_p = self._get_snr_p(p_p, a)
        # 获得估计位置
        position_hat = np.random.multivariate_normal(self.position, crb[0:2,0:2])
        # 更新波束赋形
        self.w = DV2D(position_hat[1], 0).a
        # 获得通信信噪比
        snr_c = self._get_snr_c(p_c, a)
        # 更新下一个状态
        s_ = [snr_p, snr_c]
        # 获得速率能效
        self.rate_eff = self._get_rate(snr_c, p_p+p_c)
        # 获得定位误差
        self.p_error = math.sqrt((crb_diag[0]*cfg.C)**2 + (crb_diag[1]*self.d)**2)
        # 获得奖励
        reward = self._get_reward(self.rate_eff, p_p+p_c, self.p_error)
        return s_, reward

