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

        self.bt_maxiter = kwargs.pop('bt_maxiter',50)
        self.bt_omega = kwargs.pop('bt_omega',0.75)
        self.bt_c = kwargs.pop('bt_c',0.1)

class IterationData:
    def __init__(self, n, nsteps, **kwargs):
        self.iter, self.nsteps, self.nstepsused = 0, nsteps, 0
        self.dx = np.zeros(shape=(nsteps,n))
        self.liniter = np.zeros(shape=(nsteps))
        self.dxnorm = np.zeros(shape=(nsteps))
        self.ind = []
    def newstep(self, dx, liniter):
        self.last = self.iter%self.nsteps
        if self.nstepsused == self.nsteps:
           self.ind.pop(0)
        else:
            self.nstepsused += 1
        self.ind.append(self.last)

        self.liniter[self.last] = liniter
        self.dxnorm[self.last] = np.linalg.norm(dx)
        self.dx[self.last] = dx
        if len(self.ind)>1:
            self.rhodx = self.dxnorm[self.ind[-1]]/self.dxnorm[self.ind[-2]]
        else:
            self.rhodx = 0


        print(f"{self.iter=} {self.last=} {self.nstepsused=} {self.ind=}")
        print(f"{self.dxnorm[self.ind]}")

        self.iter += 1
       