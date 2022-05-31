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
import DirectionVec as DV
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

def vec2diag(vec):
    vec = np.array(vec)
    # print(vec)
    mat = np.zeros([len(vec), len(vec)])+1j*np.zeros([len(vec), len(vec)])
    for i_idx in range(len(vec)):
        mat[i_idx,i_idx]=vec[i_idx]
    mat = np.mat(mat)
    return mat

P_p = 10
theta = cmath.pi/3
phi = cmath.pi/4
sigma = 1e-9
alpha = 1e-7+1j*1e-7
tau = 1.33e-6
s = Signal()

d = solve_delay_vec(tau)
d_ = solve_delay_div_vec(tau)

dv = DV.DirectionVec(theta, phi, 0)
a = dv.a
w = dv.a
a_theta = dv.a_div_theta
a_phi = dv.a_div_phi
x = np.multiply(w, s.s_p)

S = s.S_p
W = vec2diag(w)
X = (W*S).H
j_11 = (2*P_p*abs(alpha)**2/sigma**2)*lg.norm(np.multiply(d_, X*a))**2
j_22 = (2*P_p*abs(alpha)**2/sigma**2)*lg.norm(np.multiply(d, X*a_theta))**2
j_33 = (2*P_p*abs(alpha)**2/sigma**2)*lg.norm(np.multiply(d, X*a_phi))**2
j_44 = (2*P_p/sigma**2)*lg.norm(np.multiply(d, X*a))**2
j_55 = j_44
# print(j_11, j_22, j_33, j_44, j_55)
# no_diag
j_12 = (2*P_p*abs(alpha)**2/sigma**2)*np.real((np.multiply(d_, X*a)).H*(np.multiply(d, X*a_theta)))
j_13 = (2*P_p*abs(alpha)**2/sigma**2)*np.real((np.multiply(d_, X*a)).H*(np.multiply(d, X*a_phi)))
j_14 = (2*P_p/sigma**2)*np.real((alpha*np.multiply(d_, X*a)).H*(np.multiply(d, X*a)))
j_15 = (2*P_p/sigma**2)*np.real((1j*alpha*np.multiply(d_, X*a)).H*(np.multiply(d, X*a)))
# j_12 = 0
# j_13 = 0
# j_14 = 0
# j_15 = 0
j_23 = (2*P_p*abs(alpha)**2/sigma**2)*np.real((np.multiply(d, X*a_theta)).H*(np.multiply(d, X*a_phi)))
j_24 = (2*P_p/sigma**2)*np.real(alpha*(np.multiply(d, X*a)).H*(np.multiply(d, X*a_theta)))
j_25 = (2*P_p/sigma**2)*np.real(1j*alpha*(np.multiply(d, X*a)).H*(np.multiply(d, X*a_theta)))
j_34 = (2*P_p/sigma**2)*np.real(alpha*(np.multiply(d, X*a)).H*(np.multiply(d, X*a_phi)))
j_35 = (2*P_p/sigma**2)*np.real(1j*alpha*(np.multiply(d, X*a)).H*(np.multiply(d, X*a_phi)))
j_45 = 0
# print(j_12, j_13, j_14, j_15, j_23, j_24, j_25, j_34, j_35)

J = [[float(j_11), float(j_12), float(j_13), float(j_14), float(j_15)],
     [float(j_12), float(j_22), float(j_23), float(j_24), float(j_25)],
     [float(j_13), float(j_23), float(j_33), float(j_34), float(j_35)],
     [float(j_14), float(j_24), float(j_34), float(j_44), float(j_45)],
     [float(j_15), float(j_25), float(j_35), float(j_45), float(j_55)]]
# J = [[float(j_11), float(j_14), float(j_15)],
#      [float(j_14), float(j_44), float(j_45)],
#      [float(j_15), float(j_45), float(j_55)]]
# J = [[float(j_22), float(j_23)],
#      [float(j_23), float(j_33)]] 

# print(np.mat(J))
J = np.mat(J).I
print(J)



