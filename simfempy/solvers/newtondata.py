#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:38:16 2016

@author: becker
"""

import numpy as np

class StoppingData:
    def __init__(self, **kwargs):
        self.maxiter = kwargs.pop('maxiter',100)
        self.atol = kwargs.pop('atol',1e-14)
        self.rtol = kwargs.pop('rtol',1e-8)
        self.atoldx = kwargs.pop('atoldx',1e-14)
        self.rtoldx = kwargs.pop('rtoldx',1e-10)
        self.divx = kwargs.pop('divx',1e8)
        self.firststep = 1.0
        self.steptype = kwargs.pop('steptype','backtracking')
        if 'nbase' in kwargs: self.nbase = kwargs.pop('nbase')
        self.bt_maxiter = kwargs.pop('bt_maxiter',50)
        self.bt_omega = kwargs.pop('bt_omega',0.75)
        self.bt_c = kwargs.pop('bt_c',0.1)
        self.maxter_stepsize = 5

class IterationData:
    def __init__(self, resnorm, **kwargs):
        self.liniter, self.dxnorm, self.resnorm, self.step = [], [], [], []
        self.iter = 0
        self.resnorm.append(resnorm)
    def newstep(self, dx, liniter, resnorm, step):
        self.liniter.append(liniter)
        self.dxnorm.append(np.linalg.norm(dx))
        self.resnorm.append(resnorm)
        self.step.append(step)
        if len(self.dxnorm)>1:
            self.rhodx = self.dxnorm[-1]/self.dxnorm[-2]
        else:
            self.rhodx = 0
        self.rhor = self.resnorm[-1]/self.resnorm[-2]
        self.iter += 1
       