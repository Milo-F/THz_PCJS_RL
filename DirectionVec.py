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
from select import select
import numpy as np
import Config as cfg


class DirectionVec():
    def __init__(self, theta, phi, n) -> None:
        # 索引
        self.idx_x = [x for x in range(0, cfg.M_X)]
        self.idx_z = [x for x in range(0, cfg.M_Z)]
        # 指数公共部分
        self.com_part = 1j*2*cmath.pi * \
            (cfg.F_C+n/(cfg.N*cfg.T_S))*cfg.D_0/cfg.C
        # 指数
        self.vartheta = self.com_part * cmath.cos(theta)
        self.varphi = self.com_part * cmath.sin(phi)
        # 导数乘积项
        self.vartheta_div = -self.com_part * cmath.sin(theta)
        # self.vartheta_div_phi = self.com_part * cmath.cos(theta) * cmath.cos(phi)
        self.varphi_div = self.com_part * cmath.cos(phi)

        # 结果
        self.a = self.get_a()
        self.a_div_theta = self.get_a_div_theta()
        self.a_div_phi = self.get_a_div_phi()
        pass

    # 计算a_x
    def solve_a_x(self):
        a_x = []
        for i_idx in range(cfg.M_X):
            a_x.append(cmath.exp(self.idx_x[i_idx]*self.vartheta))
        a_x = np.mat(a_x).T
        return a_x

    # 计算a_z
    def solve_a_z(self):
        a_z = []
        for i_idx in range(cfg.M_Z):
            a_z.append(cmath.exp(self.idx_z[i_idx]*self.varphi))
        a_z = np.mat(a_z).T
        return a_z

    # 计算向量化之后的a
    def get_a(self):
        a_x = self.solve_a_x()
        a_z = self.solve_a_z()
        A = a_x*a_z.T
        a = A.T.flatten().T
        return a

    # 计算a_x对theta的导数
    def solve_a_x_div(self):
        a_x_ = []
        for i_idx in range(cfg.M_X):
            a_x_.append(self.idx_x[i_idx]*self.vartheta_div *
                        cmath.exp(self.idx_z[i_idx]*self.vartheta))
        a_x_ = np.mat(a_x_).T
        return a_x_

    # 计算a_x对phi的导数
    def solve_a_x_div_phi(self):
        a_x_ = []
        for i_idx in range(cfg.M_X):
            a_x_.append(self.idx_x[i_idx]*self.vartheta_div_phi *
                        cmath.exp(self.idx_z[i_idx]*self.vartheta))
        a_x_ = np.mat(a_x_).T
        return a_x_

    # 计算a_z对phi的导数

    def solve_a_z_div(self):
        a_z_ = []
        for i_idx in range(cfg.M_Z):
            a_z_.append(self.idx_z[i_idx]*self.varphi_div *
                        cmath.exp(self.idx_z[i_idx]*self.varphi))
        a_z_ = np.mat(a_z_).T
        return a_z_

    # 得到a对theat的求导
    def get_a_div_theta(self):
        a_x_ = self.solve_a_x_div()
        a_z = self.solve_a_z()
        A = a_x_*a_z.T
        a_div_theta = A.T.flatten().T
        return a_div_theta

    # 得到a对phi的求导
    def get_a_div_phi(self):
        a_z_ = self.solve_a_z_div()
        # a_x_ = self.solve_a_x_div_phi()
        # A = a_x_*a_z_.T
        a_x = self.solve_a_x()
        A = a_x*a_z_.T
        a_div_phi = A.T.flatten().T
        return a_div_phi
