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


def get_position_hat(crb_p, mean):
    '''
        根据输入的CRB协方差矩阵生成随机的估计位置
        参数：协方差矩阵，均值
    '''
    tau_hat, theta_hat, phi_hat = np.random.multivariate_normal(mean, crb_p)
    return tau_hat, theta_hat, phi_hat


def car2sphe(x, y, z):
    '''笛卡尔坐标转化球坐标'''
    d = math.sqrt(x**2+y**2+z**2)
    theta = math.asin(y/math.sqrt(x**2+y**2))
    phi = math.acos(z/d)
    return d, theta, phi


def sphe2car(tau, theta, phi):
    '''球坐标转化笛卡尔坐标'''
    d = tau*cfg.C
    x = d*math.sin(phi)*math.cos(theta)
    y = d*math.sin(phi)*math.sin(theta)
    z = d*math.cos(phi)
    return x, y, z


def vec2diag(vec):
    '''向量对角化'''
    vec = np.array(vec)
    # print(vec)
    mat = np.zeros([len(vec), len(vec)])+1j*np.zeros([len(vec), len(vec)])
    for i_idx in range(len(vec)):
        mat[i_idx, i_idx] = vec[i_idx]
    mat = np.mat(mat)
    return mat
