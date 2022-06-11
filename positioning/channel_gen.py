# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:15:08 2022

@author: milo
"""

import Config as cfg
import cmath
import numpy as np
import random as rd

rd.seed(0)
np.random.seed(0)

# 生成不同子载波的响应向量
def response_matrix(position):
    # tau = position[0]
    theta = position[1]
    phi = position[2]
    # 初始化列表
    A_list = [] # A_list[0]表示F_C+1/(N*T_S)的响应矩阵
    a_x = []
    a_z = []
    # 不同子载波有不同的响应矩阵
    for i_idx in range(1, cfg.N+1):
        # 清空
        a_x.clear()
        a_z.clear()
        # 生成a_x
        for x in range(int(-(cfg.M_X-1)/2), int((cfg.M_X-1)/2)+1):
            tmp = (1/cmath.sqrt(cfg.M_X))*cmath.exp(1j*2*cmath.pi*x*(cfg.F_C+i_idx/(cfg.N*cfg.T_S))*(cfg.D_0*cmath.cos(theta)/cfg.C))
            a_x.append(tmp)
        a_x_mat = np.mat(a_x).T
        # 生成a_z
        for x in range(int(-(cfg.M_Z-1)/2), int((cfg.M_Z-1)/2)+1):
            tmp = (1/cmath.sqrt(cfg.M_Z))*cmath.exp(1j*2*cmath.pi*x*(cfg.F_C+i_idx/(cfg.N*cfg.T_S))*(cfg.D_0*cmath.sin(phi)/cfg.C))
            a_z.append(tmp)
        a_z_mat = np.mat(a_z).T
        # 生成响应矩阵
        A = a_x_mat*a_z_mat.T
        a = A.flatten().T
        # 添加当前子载波的响应矩阵
        A_list.append(a)
    return A_list

# 生成不同子载波的波束赋形向量
def beamforming_matrix(position):
    # 定位的波束赋形方向与真实方向存在偏差
    theta = position[1]+rd.random()*(cmath.pi/12)
    phi = position[2]+rd.random()*(cmath.pi/12)
    # 定义列表
    W_list = [] # W_list[0]表示F_C+1/(N*T_S)的波束赋形矩阵
    w_x = []
    w_z = []
    # 不同子载波有不同的波束赋形矩阵
    for i_idx in range(1, cfg.N+1):
        # 清空
        w_x.clear()
        w_z.clear()
        # 生成a_x
        for x in range(int(-(cfg.M_X-1)/2), int((cfg.M_X-1)/2)+1):
            tmp = (1/cmath.sqrt(cfg.M_X))*cmath.exp(1j*2*cmath.pi*x*(cfg.F_C+i_idx/(cfg.N*cfg.T_S))*(cfg.D_0*cmath.cos(theta)/cfg.C))
            w_x.append(tmp)
        w_x_mat = np.mat(w_x).T
        # 生成a_z
        for x in range(int(-(cfg.M_Z-1)/2), int((cfg.M_Z-1)/2)+1):
            tmp = (1/cmath.sqrt(cfg.M_Z))*cmath.exp(1j*2*cmath.pi*x*(cfg.F_C+i_idx/(cfg.N*cfg.T_S))*(cfg.D_0*cmath.sin(phi)/cfg.C))
            w_z.append(tmp)
        w_z_mat = np.mat(w_z).T
        # 生成响应矩阵
        W = w_x_mat*w_z_mat.T
        w = W.flatten().T
        # 添加当前子载波的响应矩阵
        W_list.append(w)
    return W_list, theta, phi   
 
# 生成不同子载波的延迟因子
def delay_factor(position):
    tau = position[0]*1e-6
    delay_list = []
    for i_idx in range(1, cfg.N+1):
        tmp = cmath.exp(-1j*2*cmath.pi*(cfg.F_C+i_idx/(cfg.N*cfg.T_S))*tau)
        delay_list.append(tmp)
    return delay_list

# 产生随机的衰落
def alpha_gen(position):
    tau = position[0]*1e-6
    # 路径损耗
    alpha_loss = 1/(4*cmath.pi*cfg.F_C*tau)
    alpha_mos = cmath.exp(-(1/2)*cfg.KAPPA*cfg.C*tau)
    alpha_rd = rd.gauss(0, alpha_loss/5)+1j*rd.gauss(0, alpha_loss/5) # alpha中的随机衰落，高斯分布
    alpha = alpha_loss*alpha_mos+alpha_rd
    return alpha