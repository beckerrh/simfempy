#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:38:16 2016

@author: becker
"""

class StoppingData:
    def __init__(self):
        self.maxiter = 100
        self.atol = 1e-14
        self.rtol = 1e-10
        self.atoldx = 1e-14
        self.rtoldx = 1e-10
        self.divx = 1e8
        self.firststep = 1.0

        self.bt_maxiter = 50
        self.bt_omega = 0.5
        self.bt_c = 0.1
