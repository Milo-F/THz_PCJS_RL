#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Env.py
    @Author: Milo
    @Date: 2022/05/30 16:35:48
    @Version: 1.0
    @Description: 强化学习的交互环境，包含通信速率，定位CRB等优化问题的目标与约束等
'''

import Crb  # 克拉美劳界
import Signal  # 发送信号
import Config as cfg  # 配置固定参数
import ComChannelNoRobust as cchnr  # 通信信道
import Position  # 定位估计位置
import Constraints as cons # 约束的边界值
import DirectionVec as DV # 方向向量
import numpy.linalg as lg
import cmath
import math
import numpy as np


class RLEnv():
    # 构造函数
    def __init__(
        self,
        position, #位置
        snr_db = 10 # 信噪比
        ) -> None:
        # 真实位置（直角坐标）
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        # 真实位置（球坐标）
        self.d, self.theta, self.phi = Position.car2sphe(self.x, self.y, self.z)
        # # 路损
        # self.alpha_loss = cfg.C/(4*cmath.pi*self.d*cfg.F_C)
        # self.alpha_moss = math.exp(-0.5*cfg.KAPPA*self.d)
        # self.alpha_random = np.random.randn()*cfg.ALPHA_SIGMA + 1j*np.random.randn()*cfg.ALPHA_SIGMA
        # self.alpha = self.alpha_loss*self.alpha_moss + self.alpha_random
        # # 信噪比
        # self.snr_db = snr_db
        # self.snr = 10**(self.snr_db/10)
        # # 定位信号矩阵
        # self.S = Signal.Signal().S_p
        # # 信道参数
        # dv = DV.DirectionVec(self.theta, self.phi, 0)
        # a = dv.a
        # W = Crb.Crb().vec2diag(a)
        # X = (W*self.S).H
        # 根据信噪比产生定位噪声和通信噪声
        # self.sigma_p = (power_p)/(cfg.N*self.snr)
        pass

    # 计算通信速率
    def solve_rate(self):
        pass
    
    # 计算奖励
    def solve_reward(self):
        pass
        
    
    # 环境交互函数
    def step(self):
        pass
