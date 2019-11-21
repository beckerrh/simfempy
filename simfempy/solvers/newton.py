#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:38:16 2016

@author: becker
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
if __name__ == '__main__':
    import stoppingdata
else:
    from . import stoppingdata

#----------------------------------------------------------------------
def backtracking(f, x0, dx, resfirst, sdata, firststep=1.0, verbose=False):
    maxiter, omega, c = sdata.bt_maxiter, sdata.bt_omega, sdata.bt_c
    step = firststep
    x = x0 - step*dx
    res = f(x)
    resnorm = linalg.norm(res)
    it = 0
    if verbose:
        print("{} {:>3} {:^10} {:^10}  {:^9}".format("bt", "it", "resnorm", "resfirst", "step"))
        print("{} {:3} {:10.3e} {:10.3e}  {:9.2e}".format("bt", it, resnorm, resfirst, step))
    while resnorm > (1-c*step)*resfirst and it<maxiter:
        it += 1
        step *= omega
        x = x0 - step * dx
        res = f(x)
        resnorm = linalg.norm(res)
        if verbose:
            print("{} {:3} {:10.3e}  {:9.2e}".format("bt", it, resnorm, step))
    return x, res, resnorm, step, it

#----------------------------------------------------------------------
def newton(x0, f, computedx=None, sdata=None, verbose=False, jac=None):
    """
    Aims to solve f(x) = 0
    computedx: gets dx from f'(x) dx =  -f(x)
    """
    if sdata is None: sdata = stoppingdata.StoppingData()
    atol, rtol, atoldx, rtoldx = sdata.atol, sdata.rtol, sdata.atoldx, sdata.rtoldx
    maxiter, divx, firststep = sdata.maxiter, sdata.divx, sdata.firststep
    x = np.asarray(x0)
    assert x.ndim == 1
    n = x.shape[0]
    if not computedx:  assert jac
    xnorm = linalg.norm(x)
    dxnorm = xnorm
    res = f(x)
    resnorm = linalg.norm(res)
    tol = max(atol, rtol*resnorm)
    toldx = max(atoldx, rtoldx*xnorm)
    it = 0
    if verbose:
        print("{} {:>3} {:^10} {:^10} {:^10} {:^9}".format("newton", "it", "|x|", "|dx|", '|r|', 'step'))
        print("{} {:3} {:10.3e} {:^10} {:10.3e} {:^9}".format("newton", it, xnorm, 3*'-', resnorm, 3*'-'))
    while( (resnorm>tol or dxnorm>toldx) and it < maxiter):
        it += 1
        if not computedx:
            J = jac(x)
            dx = linalg.solve(J, res)
        else:
            dx = computedx(res, x)
        dxnorm = linalg.norm(dx)
        x, res, resnorm, step, itbt = backtracking(f, x, dx, resnorm, sdata, firststep=firststep)
        xnorm = linalg.norm(x)
        if verbose:
            print("{} {:3} {:10.3e} {:10.3e} {:10.3e} {:9.2e}".format("newton", it, xnorm, dxnorm, resnorm, step))
        if xnorm >= divx:
            return (x, maxiter)
    return (x,it)


# ------------------------------------------------------ #

if __name__ == '__main__':
    f = lambda x: 10.0 * np.sin(2.0 * x) + 4.0 - x * x
    df = lambda x: 20.0 * np.cos(2.0 * x) - 2.0 * x
    f = lambda x: x**2 -11
    df = lambda x: 2.0 * x
    def computedx(r, x):
        return r/df(x)
    x0 = [3.]
    info = newton(x0, f, jac=df, verbose=True)
    info2 = newton(x0, f, computedx=computedx, verbose=True)
    print(('info=', info))
    assert info==info2
    x = np.linspace(-1., 4.0)
    plt.plot(x, f(x), [x[0], x[-1]], [0,0], '--r')
    plt.show()
