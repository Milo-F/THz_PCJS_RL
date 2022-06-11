#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Tools.py
    @Author: Milo
    @Date: 2022/06/11 21:01:11
    @Version: 1.0
    @Description: 保存结果的工具函数
'''

from math import ceil
import os, time
from matplotlib import pyplot as plt
from scipy.io import savemat

'''保存作图用到的数据'''
def save_fig_data(data:list, file_name:str):
    savemat("fig_datas"+os.sep+time.strftime("%m%d")+os.sep+time.strftime("%H%M")+"_{}.mat".format(file_name), mdict={file_name:data})

'''打印信息'''
def print_log(str: str):
    s_tmp = (40-ceil(len(str)/2))
    e_tmp = s_tmp
    if (len(str) % 2):
        e_tmp = e_tmp + 1
    print("="*s_tmp + str + "="*e_tmp)

'''创建今天的日志文件夹'''
def create_folder():
    # 创建图像文件夹和日志文件夹
    if not os.path.exists("fig_datas"):
        os.makedirs("fig_datas")
    if not os.path.exists("figures"):
        os.makedirs("figures")
    folder_name = time.strftime("%m%d")
    if not os.path.exists("fig_datas" + os.sep + folder_name):
        os.makedirs("fig_datas" + os.sep + folder_name)
    if not os.path.exists("figures" + os.sep + folder_name):
        os.makedirs("figures" + os.sep + folder_name)

'''画图函数'''
def plot_fig(y: list, x_str: str, y_str: str, fig_name: str):
    x = [x for x in range(len(y))]
    plt.plot(x, y)
    plt.xlabel(x_str)
    plt.ylabel(y_str)
    # 把图像保存到今天的文件夹中
    plt.savefig("figures"+os.sep+time.strftime("%m%d")+os.sep +
                "{}_{}.svg".format(time.strftime("%H%M"), fig_name), dpi = 600, format = "svg")
    plt.close()
    save_fig_data(y, fig_name)
    