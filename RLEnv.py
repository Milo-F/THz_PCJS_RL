#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Env.py
    @Author: Milo
    @Date: 2022/05/30 16:35:48
    @Version: 1.0
    @Description: 强化学习的交互环境，包含通信速率，定位CRB等优化问题的目标与约束等
'''

import Crb # 克拉美劳界
import Signal # 发送信号
import config as cfg # 配置固定参数
import ComChannelNoRobust as cchnr # 通信信道
import Position # 定位估计位置


class RLEnv():
    # 构造函数
    def __init__(self) -> None:
        
        pass
    
    
    # 计算通信速率
    def solve_rate():
        pass
