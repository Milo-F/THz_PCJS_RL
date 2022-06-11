#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Arguments.py
    @Author: Milo
    @Date: 2022/06/09 16:34:05
    @Version: 1.0
    @Description: 超参数
'''

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # 参数
    parser.add_argument("--train", default=1, type=int, help="选择训练还是使用网络，训练为1，使用为0")
    parser.add_argument("--reward_decay", default=0.9, type=float, help="奖励折扣因子")
    parser.add_argument("--actor_lr", default=1e-6, type=float, help="actor网络学习率")
    parser.add_argument("--critic_lr", default=1e-5, type=float, help="critic网络学习率")
    parser.add_argument("--episode_num", default=500, type=int, help="训练迭代周期数量")
    parser.add_argument("--episode_len", default=512, type=int, help="每次迭代周期步数")
    parser.add_argument("--batch_size", default=64, type=int, help="批尺寸")
    parser.add_argument("--actor_update_step", default=10, type=int, help="actor网络更新需要步数")
    parser.add_argument("--critic_update_step", default=10, type=int, help="critic网络更新需要步数")
    parser.add_argument("--eps", default=1e-8, type=float, help="??")
    parser.add_argument("--epsilon", default=0.2, type=float, help="PPO2对重要性采样截断的超参数")
    config = parser.parse_args()
    return config
