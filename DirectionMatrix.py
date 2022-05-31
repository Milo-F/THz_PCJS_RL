#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Channel.py
    @Author: Milo
    @Date: 2022/05/30 19:54:20
    @Version: 1.0
    @Description: 信道相关的向量等的计算
'''

import cmath
import numpy as np
import config as cfg

# 计算a_x
def solve_a_x(theta, n):
    vartheta = 1j*2*cmath.pi*(cfg.F_C+n/(cfg.N*cfg.T_S))*cfg.D_0*cmath.cos(theta)/cfg.C
    x = [x for x in range(-int((cfg.M_X-1)/2), int((cfg.M_X-1)/2)+1)]
    # x = [x for x in range(0, cfg.M_X)]
    x = np.array(x)
    tmp = x*vartheta
    a_x = []
    for x in tmp:   
        a_x.append(1/cmath.sqrt(cfg.M_X)*cmath.exp(x))
    a_x = np.mat(a_x).T
    return a_x

# 计算a_z
def solve_a_z(phi, n):
    varphi = 1j*2*cmath.pi*(cfg.F_C+n/(cfg.N*cfg.T_S))*cfg.D_0*cmath.sin(phi)/cfg.C
    x = [x for x in range(-int((cfg.M_Z-1)/2), int((cfg.M_Z-1)/2)+1)]
    # x = [x for x in range(0, cfg.M_Z)]
    x = np.array(x)
    tmp = x*varphi
    a_z = []
    for x in tmp:   
        a_z.append(1/cmath.sqrt(cfg.M_X)*cmath.exp(x))
    a_z = np.mat(a_z).T
    return a_z

# 计算向量化之后的a
def get_a(theta, phi, n):
    a_x = solve_a_x(theta, n)
    a_z = solve_a_z(phi, n)
    A = a_x*a_z.T
    a = A.T.flatten().T
    return a

# 计算a_x对theta的导数
def solve_a_x_div(theta, n):
    vartheta = 1j*2*cmath.pi*(cfg.F_C+n/(cfg.N*cfg.T_S))*cfg.D_0*cmath.cos(theta)/cfg.C
    vartheta_ = -1j*2*cmath.pi*(cfg.F_C+n/(cfg.N*cfg.T_S))*cfg.D_0*cmath.sin(theta)/cfg.C
    x = [x for x in range(-int((cfg.M_X-1)/2), int((cfg.M_X-1)/2)+1)]
    # x = [x for x in range(0, cfg.M_X)]
    x = np.array(x)
    tmp = x*vartheta
    a_x_ = []
    for x in tmp:   
        a_x_.append(vartheta_*1/cmath.sqrt(cfg.M_X)*cmath.exp(x))
    a_x_ = np.mat(a_x_).T
    return a_x_

# 计算a_z对phi的导数
def solve_a_z_div(phi, n):
    varphi = 1j*2*cmath.pi*(cfg.F_C+n/(cfg.N*cfg.T_S))*cfg.D_0*cmath.sin(phi)/cfg.C
    varphi_ = 1j*2*cmath.pi*(cfg.F_C+n/(cfg.N*cfg.T_S))*cfg.D_0*cmath.cos(phi)/cfg.C
    x = [x for x in range(-int((cfg.M_Z-1)/2), int((cfg.M_Z-1)/2)+1)]
    # x = [x for x in range(0, cfg.M_Z)]
    x = np.array(x)
    tmp = x*varphi
    a_z_ = []
    for x in tmp:   
        a_z_.append(varphi_*1/cmath.sqrt(cfg.M_X)*cmath.exp(x))
    a_z_ = np.mat(a_z_).T
    return a_z_

# 得到a对theat的求导
def get_a_div_theta(theta, phi, n):
    a_x_ = solve_a_x_div(theta, n)
    a_z = solve_a_z(phi, n)
    A = a_x_*a_z.T
    a_div_theta = A.T.flatten().T
    return a_div_theta

# 得到a对phi的求导
def get_a_div_phi(theta, phi, n):
    a_x = solve_a_x(theta, n)
    a_z_ = solve_a_z_div(phi, n)
    A = a_x*a_z_.T
    a_div_phi = A.T.flatten().T
    return a_div_phi