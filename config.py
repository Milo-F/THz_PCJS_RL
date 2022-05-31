# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:27:10 2022
    固定参数配置
@author: milo
"""

C = 3e8     # 光速
F_C = 500e9   # 载波频率500GHz
G_BS = 1       # 基站天线增益
G_RA = 1       # 用户天线增益
BW = 1e9
T_S = 1/(BW*2)   # 码元周期
N = 16      # 子载波数量
M_X = 3
M_Z = 3
M = M_X * M_Z  # 天线数量
KAPPA = 0.01    # 大气吸收系数

D_0 = (1/2)*(C/F_C)   # 基站天线间距
LAMBDA_C = C/F_C           # 载波波长
