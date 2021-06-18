# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np

#=================================================================#
class IterationCounter(object):
    """
    Simple class for information on iterative solver
    """
    def __init__(self, disp=20, name="", verbose=False):
        self.disp = disp
        self.name = name
        self.verbose = verbose
        self.niter = 0
    def __call__(self, rk=None):
        if self.disp and self.niter%self.disp==0:
            print(f"iter({self.name}) {self.niter:4d}\trk = {np.linalg.norm(rk)}")
        self.niter += 1
    def __del__(self):
        if self.verbose: print('niter ({}) {:4d}'.format(self.name, self.niter))
#=================================================================#
class IterationCounterWithRes(IterationCounter):
    """
    Simple class for information on iterative solver
    """
    def __init__(self, disp=20, name="", verbose=False, callback_type='x', b=None, A=None):
        super().__init__(disp, name, verbose)
        self.res = []
        self.callback_type = callback_type
        self.b, self.A = b, A
    def __call__(self, x):
        super().__call__(x)
        res = np.linalg.norm(self.b-self.A@x)
        if self.verbose: print(f"{res=}")
        self.res.append(res)
        # print(f"{self.niter =} \t{self.res=}")
 