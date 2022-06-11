#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Position.py
    @Author: Milo
    @Date: 2022/05/30 17:03:48
    @Version: 1.0
    @Description: 和用户位置相关的函数实现
'''

from enviroment import Config as cfg
import numpy as np
import math


# 根据输入的CRB协方差矩阵生成随机的估计位置
def get_position_hat(crb_p, mean):
    # crb_p[0,0] = (math.sqrt(crb_p[0,0])*cfg.C)**2
    x_hat, y_hat, z_hat = np.random.multivariate_normal(mean, crb_p).T
    return x_hat, y_hat, z_hat


# 笛卡尔坐标转化球坐标
def car2sphe(x, y, z):
    d = math.sqrt(x**2+y**2+z**2)
    theta = math.asin(y/math.sqrt(x**2+y**2))
    phi = math.acos(z/d)
    return d, theta, phi


# 球坐标转化笛卡尔坐标
def sphe2car(d, theta, phi):
    x = d*math.sin(phi)*math.cos(theta)
    y = d*math.sin(phi)*math.sin(theta)
    z = d*math.cos(phi)
    return x, y, z

# 向量对角化
def vec2diag(vec):
    vec = np.array(vec)
    # print(vec)
    mat = np.zeros([len(vec), len(vec)])+1j*np.zeros([len(vec), len(vec)])
    for i_idx in range(len(vec)):
        mat[i_idx,i_idx]=vec[i_idx]
    mat = np.mat(mat)
    return mat