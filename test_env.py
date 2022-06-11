#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: test_env.py
    @Author: Milo
    @Date: 2022/06/11 14:30:13
    @Version: 1.0
    @Description: 一个简单的环境用于测试网络是否收敛
'''

from random import random


class Test_Env():
    
    state_dim = 2
    action_dim = 2
    
    def __init__(self) -> None:
        self.a = 5
        self.b = 15
        pass
    
    def get_reward(self, action, x):
        if self.a*action[0] >= 10 and sum(action) < 10:
            reward = x
        else:
            reward = 0
        return reward/20
    
    def step(self, action):
        a = [0,0]
        a[0] = action[0]*6
        a[1] = action[1]*6
        x = self.b*a[1]
        reward = self.get_reward(a, x)
        state_nxt = [a[0]*self.a, a[1]*self.b]
        return state_nxt, reward, x   
    
    def reset(self):
        # 初始化环境
        state = [5*self.a, 5*self.b]
        return state