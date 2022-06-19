# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:27:10 2022
    固定参数配置
@author: milo
"""

C = 3e8     # 光速
F_C = 1e12   # 载波频率500GHz
G_BS = 1       # 基站天线增益
G_RA = 1       # 用户天线增益
BW = 1e9  # 带宽
T_S = 1/(BW*64)   # 码元周期
N = 1024      # 子载波数量
G = 1 # 定位次数
M_X = 31
M_Z = 31
M = M_X * M_Z  # 天线数量
KAPPA = 0.01    # 大气吸收系数
ALPHA_SIGMA = 1e-9 # 路损随机项

D_0 = (1/2)*(C/F_C)   # 基站天线间距
LAMBDA_C = C/F_C           # 载波波长
