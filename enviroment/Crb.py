#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: crb.py
    @Author: Milo
    @Date: 2022/05/31 09:43:46
    @Version: 1.0
    @Description: 计算CRB
'''
from enviroment import Config as cfg
import numpy as np
import numpy.linalg as lg
import cmath
import math
from enviroment import DirectionVec as DV

class Crb():
    def __init__(
        self,
        p_sphe_hat, # 估计位置
        p_p, # 分配的功率
        alpha, # 信道衰减系数
        S, # 定位信号矩阵
        sigma # 噪声标准差
        ) -> None:
        # 估计位置
        self.tau_hat = p_sphe_hat[0]/cfg.C
        self.theta_hat = p_sphe_hat[1]
        self.phi_hat = p_sphe_hat[2]
        # 信道参数
        self.p_p = p_p
        self.alpha = alpha
        # 发送信号
        self.S = S
        # 噪声
        self.sigma = sigma
        # crb
        self.crb, self.crb_diag_sqrt = self.solve_crb()
        pass
    
    
    # 获得时延向量
    def solve_delay_vec(self):
        delay_vec = []
        delay_idx = [x for x in range(1,cfg.N+1)]
        for idx in delay_idx:
            delay_vec.append(cmath.exp(-1j*2*cmath.pi*(cfg.F_C+idx/(cfg.N*cfg.T_S))*self.tau_hat))
            # delay_vec.append(cmath.exp(-1j*2*cmath.pi*(0+idx/(cfg.N*cfg.T_S))*self.tau_hat))
        delay_vec = np.mat(delay_vec).T
        return delay_vec
    
    
    # 获得时延求导
    def solve_delay_div_vec(self):
        delay_div_vec = []
        delay_idx = [x for x in range(1,cfg.N+1)]
        for idx in delay_idx:
            vartau = -1j*2*cmath.pi*(cfg.F_C + idx/(cfg.N*cfg.T_S))*self.tau_hat
            delay_div_vec.append(vartau*cmath.exp(-1j*2*cmath.pi*(cfg.F_C + idx/(cfg.N*cfg.T_S))*self.tau_hat))
            # vartau = -1j*2*cmath.pi*(0 + idx/(cfg.N*cfg.T_S))*self.tau_hat
            # delay_div_vec.append(vartau*cmath.exp(-1j*2*cmath.pi*(0 + idx/(cfg.N*cfg.T_S))*self.tau_hat))
        delay_div_vec = np.mat(delay_div_vec).T
        return delay_div_vec
    
    # 向量对角化
    def vec2diag(self, vec):
        vec = np.array(vec)
        # print(vec)
        mat = np.zeros([len(vec), len(vec)])+1j*np.zeros([len(vec), len(vec)])
        for i_idx in range(len(vec)):
            mat[i_idx,i_idx]=vec[i_idx]
        mat = np.mat(mat)
        return mat
    
    # 计算crb
    def solve_crb(self):
        # 时延项
        d = self.solve_delay_vec()
        d_ = self.solve_delay_div_vec()
        # 方向向量与波束赋形
        dv = DV.DirectionVec(self.theta_hat, self.phi_hat, 0)
        a = dv.a
        # print(a)
        W = self.vec2diag(a)
        a_theta = dv.a_div_theta
        a_phi = dv.a_div_phi
        # 波束赋形之后的信号
        X = (W*self.S).H
        # FIM
        j_11 = (2*self.p_p*abs(self.alpha)**2/self.sigma**2)*lg.norm(np.multiply(d_, X*a))**2
        j_22 = (2*self.p_p*abs(self.alpha)**2/self.sigma**2)*lg.norm(np.multiply(d, X*a_theta))**2
        j_33 = (2*self.p_p*abs(self.alpha)**2/self.sigma**2)*lg.norm(np.multiply(d, X*a_phi))**2
        j_44 = (2*self.p_p/self.sigma**2)*lg.norm(np.multiply(d, X*a))**2
        j_55 = j_44
        j_12 = (2*self.p_p*abs(self.alpha)**2/self.sigma**2)*np.real((np.multiply(d_, X*a)).H*(np.multiply(d, X*a_theta)))
        j_13 = (2*self.p_p*abs(self.alpha)**2/self.sigma**2)*np.real((np.multiply(d_, X*a)).H*(np.multiply(d, X*a_phi)))
        j_14 = (2*self.p_p/self.sigma**2)*np.real((self.alpha*np.multiply(d_, X*a)).H*(np.multiply(d, X*a)))
        j_15 = (2*self.p_p/self.sigma**2)*np.real((1j*self.alpha*np.multiply(d_, X*a)).H*(np.multiply(d, X*a)))
        j_23 = (2*self.p_p*abs(self.alpha)**2/self.sigma**2)*np.real((np.multiply(d, X*a_theta)).H*(np.multiply(d, X*a_phi)))
        j_24 = (2*self.p_p/self.sigma**2)*np.real(self.alpha*(np.multiply(d, X*a)).H*(np.multiply(d, X*a_theta)))
        j_25 = (2*self.p_p/self.sigma**2)*np.real(1j*self.alpha*(np.multiply(d, X*a)).H*(np.multiply(d, X*a_theta)))
        j_34 = (2*self.p_p/self.sigma**2)*np.real(self.alpha*(np.multiply(d, X*a)).H*(np.multiply(d, X*a_phi)))
        j_35 = (2*self.p_p/self.sigma**2)*np.real(1j*self.alpha*(np.multiply(d, X*a)).H*(np.multiply(d, X*a_phi)))
        j_45 = 0
        
        J = [[float(j_11), float(j_12), float(j_13), float(j_14), float(j_15)],
            [float(j_12), float(j_22), float(j_23), float(j_24), float(j_25)],
            [float(j_13), float(j_23), float(j_33), float(j_34), float(j_35)],
            [float(j_14), float(j_24), float(j_34), float(j_44), float(j_45)],
            [float(j_15), float(j_25), float(j_35), float(j_45), float(j_55)]]
        C = np.mat(J).I
        c_diag_sqrt = [math.sqrt(C[0,0]), math.sqrt(C[1,1]), math.sqrt(C[2,2]), math.sqrt(C[3,3]), math.sqrt(C[4,4])]
        return C, c_diag_sqrt
        
        
        
        