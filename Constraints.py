#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: Constraints.py
    @Author: Milo
    @Date: 2022/05/31 20:51:00
    @Version: 1.0
    @Description: 约束文件，存放约束及其边界值
'''
class Constraints():
    def __init__(self) -> None:
        self.p_total = 100 # 总功率约束
        self.beta_p = 60 # 独立的定位功率约束
        self.beta_c = 60 # 独立的通信功率约束
        self.rho = 10 # 定位crb精度约束
        pass
