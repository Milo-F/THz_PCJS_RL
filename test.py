#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: test.py
    @Author: Milo
    @Date: 2022/06/19 12:18:57
    @Version: 1.0
    @Description: 
'''

import numpy as np
from enviroment.env2d.Signal import Signal
from enviroment.env2d.DirectVector2D import DirectVector2D as dv2d
from enviroment.env2d.DelayFactor import DelayFactor as DF
from enviroment.env2d.Env2D import Env2D
import math
from enviroment import Config as cfg
import Tools
import time

for i in range(100):
    time.sleep(0.1)
    Tools.progress(i/100)

