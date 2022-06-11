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
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists("figures"):
        os.makedirs("figures")
    folder_name = time.strftime("%m%d")
    if not os.path.exists("logs" + os.sep + folder_name):
        os.makedirs("logs" + os.sep + folder_name)
    if not os.path.exists("figures" + os.sep + folder_name):
        os.makedirs("figures" + os.sep + folder_name)

'''画图函数'''
def plot_jpg(y: list, x_str: str, y_str: str, fig_name: str):
    x = [x for x in range(len(y))]
    plt.plot(x, y)
    plt.xlabel(x_str)
    plt.ylabel(y_str)
    # 把图像保存到今天的文件夹中
    plt.savefig("figures"+os.sep+time.strftime("%m%d")+os.sep +
                "{}_{}.svg".format(time.strftime("%H%M"), fig_name), dpi = 600, format = "svg")