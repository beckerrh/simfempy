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
    def __init__(self, disp=20, name=""):
        self.disp = disp
        self.name = name
        self.niter = 0
        self.history = []
    def __call__(self, val=None):
        res = np.linalg.norm(val)
        if self.disp and self.niter%self.disp==0:
            print(f"{self.name} {self.niter:4d}\t{res}")
        self.niter += 1
        self.history.append(res)
    # def __del__(self):
    #     if self.verbose: print('niter ({}) {:4d}'.format(self.name, self.niter))
#=================================================================#
class IterationCounterWithRes(IterationCounter):
    """
    Simple class for information on iterative solver
    """
    def __init__(self, disp=20, name="", callback_type='x', b=None, A=None):
        super().__init__(disp, name)
        self.callback_type = callback_type
        self.b, self.A = b, A
    def __call__(self, x):
        if self.callback_type == "x":
            super().__call__(self.b-self.A@x)
    def __call__(self, x, Fx):
        if self.callback_type == "x,Fx":
            super().__call__(Fx)
