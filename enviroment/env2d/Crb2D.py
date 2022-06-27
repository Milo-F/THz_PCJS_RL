#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Crb2D.py
    @Author: Milo
    @Date: 2022/06/23 15:22:47
    @Version: 1.0
    @Description: 二维CRB推导
'''

from enviroment import Config as cfg
from enviroment.env2d.DirectVector2D import DirectVector2D as DV2D
import numpy as np
from numpy.linalg import norm
import math

class Crb2D():
    
    def __init__(
        self,
        a, # 方向向量
        a_, # 方向向量导数
        D, # 延时因子对角阵
        D_, # 延时因子对角阵导数
        p_pos, # 定位功率
        alpha, # 
        x, # 波束赋形之后的信号
        sigma
        ) -> None:
        self.a = a
        self.a_ = a_
        self.D = np.mat(D)
        self.D_ = np.mat(D_)
        self.p_pos = p_pos
        self.alpha = alpha
        self.x = np.mat(x).T
        self.sigma = sigma
        # crb
        self.crb, self.crb_diag_sqrt = self._get_crb()
        pass
    
    def _get_crb(self):
        J = np.zeros([4,4])
        J[0,0] = (2*self.p_pos)/(self.sigma**2)*abs(self.alpha)**2*norm(self.D_*self.x*self.a)**2
        J[1,1] = (2*self.p_pos)/(self.sigma**2)*abs(self.alpha)**2*norm(self.D*self.x*self.a_)**2
        J[2,2] = (2*self.p_pos)/(self.sigma**2)*norm(self.D*self.x*self.a)**2
        J[3,3] = J[2,2]
        J[0,1] = (2*self.p_pos)/(self.sigma**2)*abs(self.alpha)**2*np.real(self.a.H*self.x.H*self.D_.H*self.D*self.x*self.a_)
        J[1,0] = J[0,1]
        J[0,2] = (2*self.p_pos)/(self.sigma**2)*np.real(self.alpha*self.a.H*self.x.H*self.D_.H*self.D*self.x*self.a)
        J[2,0] = J[0,2]
        J[0,3] = (2*self.p_pos)/(self.sigma**2)*np.real(1j*self.alpha*self.a.H*self.x.H*self.D_.H*self.D*self.x*self.a)
        J[3,0] = J[0,3]
        J[1,2] = (2*self.p_pos)/(self.sigma**2)*np.real(self.alpha * self.a.H*self.x.H*self.D.H*self.D*self.x*self.a_)
        J[2,1] = J[1,2]
        J[1,3] = (2*self.p_pos)/(self.sigma**2)*np.real(1j*self.alpha * self.a.H*self.x.H*self.D.H*self.D*self.x*self.a_)
        J[3,1] = J[1,3]
        J[2,3] = 0
        J[3,2] = 0
        crb = np.mat(J).I
        crb_diag_sqrt = [math.sqrt(crb[0,0]), math.sqrt(crb[1,1])]
        return crb, crb_diag_sqrt
        
        
        
    