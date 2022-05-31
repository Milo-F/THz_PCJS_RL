#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File:test.py
    @Author:Milo
    @Date:2022/05/30 11:23:28
    @Version:1.0
    @Description: 局部测试
'''

import cmath
import DirectionMatrix as dm
import config as cfg
import numpy as np
import numpy.linalg as lg
from Signal import Signal


def solve_delay_vec(tau):
    delay_vec = []
    delay_idx = [x for x in range(1,cfg.N+1)]
    for idx in delay_idx:
        delay_vec.append(cmath.exp(-1j*2*cmath.pi*idx*tau/(cfg.N*cfg.T_S)))
    delay_vec = np.mat(delay_vec).T
    return delay_vec

def solve_delay_div_vec(tau):
    delay_div_vec = []
    delay_idx = [x for x in range(1,cfg.N+1)]
    for idx in delay_idx:
        vartau = -1j*2*cmath.pi*idx*tau/(cfg.N*cfg.T_S)
        delay_div_vec.append(vartau*cmath.exp(-1j*2*cmath.pi*idx*tau/(cfg.N*cfg.T_S)))
    delay_div_vec = np.mat(delay_div_vec).T
    return delay_div_vec


P_p = 10
theta = cmath.pi/3
phi = cmath.pi/4
sigma = 1e-8
alpha = 1e-7+1j*1e-7
tau = 1.33e-7
s = Signal()
d = solve_delay_vec(tau)
d_ = solve_delay_div_vec(tau)
a = dm.get_a(theta, phi, 0)
a_theta = dm.get_a_div_theta(theta, phi, 0)
a_phi = dm.get_a_div_phi(theta, phi, 0)
w = dm.get_a(theta, phi, 0)
x = np.multiply(w, s.s_p)
b_phi = dm.get_a_div_phi(theta, phi, 0)
# diag
j_11 = (2*P_p*abs(alpha)**2/sigma**2)*lg.norm(d_*(a.H*x))**2
j_22 = (2*P_p*abs(alpha)**2/sigma**2)*lg.norm(d*(a_theta.H*x))**2
j_33 = (2*P_p*abs(alpha)**2/sigma**2)*lg.norm(d*(a_phi.H*x))**2
j_44 = (2*P_p/sigma**2)*lg.norm(d*(a.H*x))**2
j_55 = j_44
print(j_11, j_22, j_33, j_44, j_55)
# no_diag
j_12 = (2*P_p*abs(alpha)**2/sigma**2)*np.real((d_*(a.H*x)).H*(d*(a_theta.H*x)))
j_13 = (2*P_p*abs(alpha)**2/sigma**2)*np.real((d_*(a.H*x)).H*(d*(a_phi.H*x)))
j_14 = (2*P_p/sigma**2)*np.real(alpha*(d_*(a.H*x)).H*(d*(a.H*x)))
j_15 = (2*P_p/sigma**2)*np.real(1j*alpha*(d_*(a.H*x)).H*(d*(a.H*x)))
j_23 = (2*P_p*abs(alpha)**2/sigma**2)*np.real((d*(a_theta.H*x)).H*(d*(a_phi.H*x)))
j_24 = (2*P_p/sigma**2)*np.real(alpha*(d*(a.H*x)).H*(d*(a_theta.H*x)))
j_25 = (2*P_p/sigma**2)*np.real(1j*alpha*(d*(a.H*x)).H*(d*(a_theta.H*x)))
j_34 = (2*P_p/sigma**2)*np.real(alpha*(d*(a.H*x)).H*(d*(a_phi.H*x)))
j_35 = (2*P_p/sigma**2)*np.real(1j*alpha*(d*(a.H*x)).H*(d*(a_phi.H*x)))
j_45 = 0
print(j_12, j_13, j_14, j_15, j_23, j_24, j_25, j_34, j_35)

J = [[float(j_11), float(j_12), float(j_13), float(j_14), float(j_15)],
     [float(j_12), float(j_22), float(j_23), float(j_24), float(j_25)],
     [float(j_13), float(j_23), float(j_33), float(j_34), float(j_35)],
     [float(j_14), float(j_24), float(j_34), float(j_44), float(j_45)],
     [float(j_15), float(j_25), float(j_35), float(j_45), float(j_55)]]
J = np.mat(J).I
# P = [[float(j_11), float(j_12), float(j_13)],
#      [float(j_12), float(j_22), float(j_23)],
#      [float(j_13), float(j_23), float(j_33)]]
# P = np.mat(P).I
print(J)
# print(P)


