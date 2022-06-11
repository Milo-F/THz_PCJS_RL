#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Noise.py
    @Author: Milo
    @Date: 2022/05/30 20:15:33
    @Version: 1.0
    @Description: 采用非稳健波束赋形的通信信道
'''

import numpy as np
import cmath
from enviroment import Config as cfg
from enviroment import DirectionVec as DV



# 非稳健波束成型的通信信道
class ComChannelNoRobust():
    # 构造函数
    def __init__(self, n, p_sphe_hat, Delta_p_sphe, p_c, alpha) -> None:
        self.n = n  # 第n个子载波
        self.alpha = alpha
        # 球坐标估计位置，根据估计位置和误差产生信道
        self.tau_hat = p_sphe_hat[0]
        self.theta_hat = p_sphe_hat[1]
        self.phi_hat = p_sphe_hat[2]
        # 真实位置
        self.tau = self.tau_hat + Delta_p_sphe[0]
        self.theta = self.theta_hat + Delta_p_sphe[1]
        self.phi = self.phi_hat + Delta_p_sphe[2]
        # 方向矢量(根据估计位置以及其误差得到)
        self.direct_a = DV.DirectionVec(self.theta, self.phi, self.n).a
        # 波束赋形(根据估计位置得到)
        self.beamforming = DV.DirectionVec(
            self.theta_hat, self.phi_hat, self.n).a
        # 时延因子
        self.delay = self.delay_factor()
        # 通信功率
        self.p_c = p_c
        self.channel = cmath.sqrt(
            self.p_c)*self.alpha*(self.direct_a.H*self.beamforming).item()
        pass

    # def get_noise(sigma):
    #     nu_r = np.random.randn()*sigma/2
    #     nu_i = np.random.randn()*sigma/2
    #     nu = nu_r + 1j*nu_i
    #     return nu

    def delay_factor(self):
        # tau = tau_hat + Delta_tau
        delay = cmath.exp(-1j*2*cmath.pi*(self.n*self.tau)/(cfg.N*cfg.T_S))
        return delay
