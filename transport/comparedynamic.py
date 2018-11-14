# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from tools.comparerrors import CompareErrors
from transportcg1supgdynamic import TransportCg1SupgDynamic
from transportcg1patches import TransportCg1Patches

# ------------------------------------- #

if __name__ == '__main__':

    finaltime = 2*np.pi
    zero = lambda  x,y: 0.0
    def cone(x,y):
        c = 1.0 - np.sqrt( (x-0.5)**2 + (y-0.25)**2 )/0.15
        if c >=0.0: return c
        return 0.0
    def cylinder(x,y):
        d =  np.sqrt( (x-0.5)**2 + (y-0.25)**2 )
        if d <=0.15: return 1.0
        return 0.0
    def smooth(x,y):
        d = np.sqrt( (x-0.5)**2 + (y-0.25)**2 )/0.15
        if d <= 1.0:
            return 1.0 - 3*d**2 + 2*d**3
        return 0.0
    betarot = lambda x, y: (0.5-y, x-0.5)

    test = cone
    test = smooth
    methods = {}
    # methods['supg'] = TransportCg1SupgDynamic(upwind="supg", problem="RotatingCone", beta=betarot, dirichlet=zero, initialcondition=test,
    #                                           solexact=test, dontcomputeBcells=True, errorformula = 'cell', finaltime=finaltime)
    # methods['expl'] = TransportCg1Patches(upwind="two", timescheme="explicit", xi = "xi", problem="RotatingCone", beta=betarot, dirichlet=zero, initialcondition=test,
    #                                           solexact=test, dontcomputeBcells=True, errorformula = 'cell', finaltime=finaltime)
    # methods['impl'] = TransportCg1Patches(upwind="two", timescheme="implicit", xi = "xiter", problem="RotatingCone", beta=betarot, dirichlet=zero, initialcondition=test,
    #                                           solexact=test, dontcomputeBcells=True, errorformula = 'cell', finaltime=finaltime)
    methods['supg'] = TransportCg1SupgDynamic(upwind="supg", problem="RotatingCone", beta=betarot, dirichlet=zero, initialcondition=test,
                                              solexact=test, dontcomputeBcells=True, errorformula = 'cell', finaltime=finaltime)
    methods['supgsl'] = TransportCg1SupgDynamic(upwind="sl", problem="RotatingCone", beta=betarot, dirichlet=zero, initialcondition=test,
                                              solexact=test, dontcomputeBcells=True, errorformula = 'cell', finaltime=finaltime, xi = "xi")
    compareerrors = CompareErrors(methods, latex=True, vtk=False)
    niter = 2
    h = [0.1*np.power(0.5,i)  for i in range(niter)]
    compareerrors.compare(geomname="unitsquare01", h=h, solve="dynamic")