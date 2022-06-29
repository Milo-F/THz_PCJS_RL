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


def save_fig_data(data:list, file_name:str):
    '''保存作图用到的数据'''
    savemat("fig_datas"+os.sep+time.strftime("%m%d")+os.sep+time.strftime("%H%M")+"_{}.mat".format(file_name), mdict={file_name:data})


def print_log(str: str):
    '''打印信息'''
    s_tmp = (40-ceil(len(str)/2))
    e_tmp = s_tmp
    if (len(str) % 2):
        e_tmp = e_tmp + 1
    print("="*s_tmp + str + "="*e_tmp)


def create_folder():
    '''创建今天的日志文件夹'''
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


def plot_fig(y: list, x_str: str, y_str: str, fig_name: str):
    '''画图函数'''
    x = [x for x in range(len(y))]
    plt.plot(x, y)
    plt.xlabel(x_str)
    plt.ylabel(y_str)
    # 把图像保存到今天的文件夹中
    plt.savefig("figures"+os.sep+time.strftime("%m%d")+os.sep +
                "{}_{}.svg".format(time.strftime("%H%M"), fig_name), dpi = 600, format = "svg")
    plt.close()
    save_fig_data(y, fig_name)
    
def progress(percent):
    '''打印进度条的函数'''
    if percent > 1:
        percent = 1
    res = int(10 * percent) * "#"
    print('\r[%-10s] %d%%' % (res, int(100 * percent)), end='\t')
    
def save_log(hyparams:dict, dg:str):
    '''保存运行日志的函数'''
    with open("figures"+os.sep+time.strftime("%m%d")+os.sep+"{}_配置.log".format(time.strftime("%H%M")), "w", encoding="utf-8") as f:
        for param_key in hyparams.keys():
            f.writelines(param_key + ": " + str(hyparams[param_key]) + "\n")
        f.writelines("备注：" + dg)
        f.close()