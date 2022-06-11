# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:52:58 2022
    产生定位网络训练、验证、测试数据
@author: milo
"""
# 
from enviroment import Config as cfg
import random as rd
import numpy  as np
import os
import math
import channel_gen as ch_g

from numpy import linalg as ll

# 随机种子设置
rd.seed(0)
np.random.seed(0)

# 产生定位信号
def generate_signal():
    s_real = [x for x in range(0, cfg.N)]
    s_imag = [x for x in range(0, cfg.N)]
    rd.shuffle(s_real)
    rd.shuffle(s_imag)
    s_real = np.array(s_real)
    s_imag = np.array(s_imag)
    s = s_real + 1j*s_imag
    s = np.array(s)
    s = s/ll.norm(s)
    return s

# 产生信道（包括方向向量与波束成形矩阵）
def generate_channel(position, P_c):
    # 产生alpha
    alpha = ch_g.alpha_gen(position)
    # 包含所有子载波的响应矩阵与波束成形
    A_list = ch_g.response_matrix(position)
    W_list, theta_w, phi_w = ch_g.beamforming_matrix(position)
    delay_list = ch_g.delay_factor(position)
    channel = []
    for i_idx in range(0, cfg.N):
        tmp = complex(math.sqrt(P_c)*alpha*delay_list[i_idx]*(A_list[i_idx].T.conjugate()*W_list[i_idx]))
        channel.append(tmp)
    channel = np.array(channel)
    return channel, alpha, theta_w, phi_w

# 产生噪声
def generate_noise(sigma):
    noise = np.random.randn(cfg.N)*sigma/2+1j*np.random.randn(cfg.N)*sigma/2
    return noise

# 产生随机位置
def generate_position():
    tau = (100+rd.random()*800)/cfg.C*1e6
    theta = rd.random()*np.pi-np.pi/2
    phi = rd.random()*np.pi
    return [tau, theta, phi]

# 建立保存数据的文件夹（按照信噪比分类）
def mkdir_snr(snr):
    path = "position_set"+os.sep+"snr({})".format(snr)
    # 判断路径是否存在
    if os.path.exists(path):
        print("="*10+"folder exist"+"="*10)
    else:
        os.makedirs(path)
        os.makedirs(path+os.sep+"train")
        os.makedirs(path+os.sep+"valid")
        os.makedirs(path+os.sep+"test")
        print("="*10+"new folder"+"="*10)

# 接收信号
def received_signal(channel, signal, noise):
    y_p = np.multiply(channel, signal)+noise
    return y_p.tolist()

# 脚本主体
s_p = generate_signal() # 发送信号
# snr_p = [x for x in range(0, 25, 5)] # 信噪比
snr_p = [0]
data_list = [] # 缓存数据的列表
lable_list = [] # 缓存标签的列表


# 按照不同的信噪比产生不同的训练数据
for i_idx in range(0, len(snr_p)): 
    
    mkdir_snr(snr_p[i_idx]) # 按照信噪比建立数据文件夹
    
    # 保存数据的根目录
    dir_path = "position_set"+os.sep+"snr({})".format(snr_p[i_idx])+os.sep 
    
    for j_idx in range(0, 10):
        
        # 清空字典
        data_list.clear()
        lable_list.clear()
        
        # 1000组数据为一组作为一个文件保存
        for k_idx in range(0, 1000):
            
            P_c = rd.random() # 定位功率
            
            position = generate_position() # 每条数据随机一个位置
            channel, alpha, theta_w, phi_w = generate_channel(position, P_c) # 按照随机位置产生信道
            # sigma = math.sqrt((ll.norm(channel)**2)/(10**(snr_p[i_idx]/10))/cfg.N)
            sigma = 0
            noise = generate_noise(sigma) # 产生随机噪声
            
            # 将随机位置与接收信号保存到字典
            y_p = received_signal(channel, s_p, noise) # 接收定位信号
            y_p.append(theta_w) # 定位波束赋形角度
            y_p.append(phi_w) # 
            y_p.append(P_c) # 定位功率
            y_p.append(sigma) # 噪声均方差
            
            data_list.append(y_p)
            
            position.append(alpha.real)
            position.append(alpha.imag)
            lable_list.append(position)
            
            if k_idx%100 == 0 :
                print("="*5+"generating data {}".format(k_idx+j_idx*1000)+", with SNR = {}".format(snr_p[i_idx]))
            
        # 保存训练集、验证集以及测试集
        if j_idx <= 7 : # 8000训练集
            print("saving train datas and lables number {}".format(j_idx))
            np.save(dir_path+"train"+os.sep+"data_{}".format(j_idx)+".npy", data_list)
            np.save(dir_path+"train"+os.sep+"lable_{}".format(j_idx)+".npy", lable_list)
        elif j_idx == 8: # 1000验证集
            print("saving valid datas and lables number {}".format(j_idx))
            np.save(dir_path+"valid"+os.sep+"data.npy", data_list)
            np.save(dir_path+"valid"+os.sep+"lable.npy", lable_list)
        else: # 1000测试集
            print("saving test datas and lables number {}".format(j_idx))
            np.save(dir_path+"test"+os.sep+"data.npy", data_list)
            np.save(dir_path+"test"+os.sep+"lable.npy", lable_list)
                
            