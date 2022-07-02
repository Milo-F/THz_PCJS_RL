#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: test.py
    @Author: Milo
    @Date: 2022/06/19 12:18:57
    @Version: 1.0
    @Description: 
'''

import Tools

func = lambda x : x%2;
result = filter(func,[1,2,3,4,5])
print(list(result))
