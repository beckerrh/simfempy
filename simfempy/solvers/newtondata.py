#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:38:16 2016

@author: becker
"""

class StoppingData:
    def __init__(self, **kwargs):
        self.maxiter = kwargs.pop('maxiter',100)
        self.atol = kwargs.pop('atol',1e-14)
        self.rtol = kwargs.pop('rtol',1e-8)
        self.atoldx = kwargs.pop('atoldx',1e-14)
        self.rtoldx = kwargs.pop('rtoldx',1e-10)
        self.divx = kwargs.pop('divx',1e8)
        self.firststep = 1.0

        self.bt_maxiter = kwargs.pop('bt_maxiter',50)
        self.bt_omega = kwargs.pop('bt_omega',0.75)
        self.bt_c = kwargs.pop('bt_c',0.1)
