#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Env.py
    @Author: Milo
    @Date: 2022/06/11 19:37:29
    @Version: 1.0
    @Description: 匹配滤波波束赋形的强化学习环境
'''


from enviroment.Crb import Crb  # 克拉美劳界
from enviroment.Signal import Signal  # 发送信号
from enviroment import Config as cfg  # 配置固定参数
from enviroment.ComChannelNoRobust import ComChannelNoRobust as ccnr  # 通信信道
from enviroment import Position  # 定位估计位置
from enviroment import Constraints as cons  # 约束的边界值
from enviroment.DirectionVec import DirectionVec as DV  # 方向向量

import math
import numpy as np


class Env():

    # 状态动作维度
    state_dim = 2
    action_dim = 2

    def __init__(
        self,
        position: list,  # 直角坐标位置
        sigma  # 环境的背景噪声
    ) -> None:
        # 真实位置（直角坐标）
        x = position[0]
        y = position[1]
        z = position[2]
        # 真实位置（tau，theta，phi）
        d, theta, phi = Position.car2sphe(
            x, y, z)
        self.d = d
        tau = d/cfg.C
        self.p_real = [tau, theta, phi]
        # 路损
        alpha_loss = cfg.C/(4*math.pi*d*cfg.F_C)
        alpha_moss = math.exp(-0.5*cfg.KAPPA*d)
        alpha_random = np.random.randn()*cfg.ALPHA_SIGMA + 1j * \
            np.random.randn()*cfg.ALPHA_SIGMA
        self.alpha = alpha_loss*alpha_moss+alpha_random
        # 噪声标准差
        self.sigma = sigma
        # 定位信号
        self.S_p = Signal().S_p
        pass

    def _get_com_channel(self, p_hat, p_Delta, com_p):
        '''获得通信信道'''
        ch = []
        for n in range(1, cfg.N+1):
            ch.append(ccnr(n, p_hat, p_Delta, com_p, self.alpha).channel)
        ch=np.mat(ch)
        return ch

    def _get_snr_p(self, power):
        '''计算定位信噪比，参数：定位功率'''
        return (power*abs(self.alpha)**2)/(self.sigma**2)
        

    def _get_snr_c(self, p_hat, p_Delta, com_p):
        '''计算通信信噪比，参数：估计位置，估计误差，通信功率'''
        ch = self._get_com_channel(p_hat, p_Delta, com_p)
        snr_c = (np.linalg.norm(ch)**2)/(cfg.N*self.sigma**2)
        return snr_c

    def _get_rate(self, p_hat, p_Delta, com_p):
        '''计算通信速率，参数：估计位置， 估计误差， 通信功率'''
        snr_c = self._get_snr_c(p_hat, p_Delta, com_p)
        rate = math.log2(1+snr_c)
        return rate

    def _get_reward(self, p_cons: float, p_list, rate):
        '''计算奖励，参数：定位约束, 功率分配， 速率'''
        if sum(p_list) > cons.P_TOTAL:
            # 违反约束
            reward = 0
        else:
            reward = rate/10
        return reward

    def step(self, action):
        '''环境互动函数，参数：动作'''

        # 从动作获取分配的功率
        pos_p = action[0] * cons.BETA_P
        com_p = action[1] * cons.BETA_C

        # 获得该位置的CRB
        c = Crb(self.p_real, pos_p, self.alpha, self.S_p, self.sigma)
        crb = c.crb
        crb_p = crb[0:3, 0:3]

        # 获得该动作下的通信速率
        # 取得估计位置
        p_hat = Position.get_position_hat(crb_p, self.p_real)
        # 取得估计误差
        p_Delta = c.crb_diag_sqrt[0:3]
        # 计算速率
        rate = self._get_rate(p_hat, p_Delta, com_p)

        # 获得该动作的奖励
        # 获得定位误差用于约束奖励
        p_cons = math.sqrt((p_Delta[0]*cfg.C)**2+(p_Delta[1]
                                        * self.d)**2+(p_Delta[2]*self.d)**2)
        # 获得奖励
        reward = self._get_reward(p_cons, [pos_p, com_p], rate)

        # 根据动作获得新的状态
        snr_p = self._get_snr_p(pos_p)
        snr_c = self._get_snr_c(p_hat, p_Delta, com_p)
        state_nxt = [snr_p, snr_c]

        # 返回新状态、动作奖励、通信速率
        return state_nxt, reward, rate

    def reset(self):
        '''初始化环境'''
        # 初始化平均分配定位与通信功率
        p_init = [0.5, 0.5]
        
        # 获得初始状态
        state_init, __, __ = self.step(p_init)
        return state_init
