#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:38:16 2016

@author: becker
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


#----------------------------------------------------------------------
def backtracking(computeResidual, x0, dx, resfirst, nit=50, omega=0.5, c=0.1, firststep=1.0, verbose=False):
    step = firststep
    x = x0 + step*dx
    res = computeResidual(x)
    resnorm = linalg.norm(res)
    it = 0
    if verbose:
        print("{} {:>3} {:^10} {:^10}  {:^9}".format("bt", "it", "resnorm", "resfirst", "step"))
        print("{} {:3} {:10.3e} {:10.3e}  {:9.2e}".format("bt", it, resnorm, resfirst, step))
    while resnorm > (1-c*step)*resfirst and it<nit:
        it += 1
        step *= omega
        x = x0 + step * dx
        res = computeResidual(x)
        resnorm = linalg.norm(res)
        if verbose:
            print("{} {:3} {:10.3e}  {:9.2e}".format("bt", it, resnorm, step))
    return x, res, resnorm, step, it

#----------------------------------------------------------------------
def newton(x, computeResidual, computeUpdate=None, nit=50, atol=1e-14, rtol=1e-10, atoldx=1e-14, rtoldx=1e-10, divx=1e8, firststep=1.0, verbose=False, jac=None):
    """
    Aims to solve F(x) = 0
    computeResidual: F(x)
    computeUpdate: F'(x) dx =  res
    """
    x = np.asarray(x)
    assert x.ndim == 1
    n = x.shape[0]
    if not computeUpdate:
        assert jac
    xnorm = linalg.norm(x)
    dxnorm = xnorm
    res = computeResidual(x)
    resnorm = linalg.norm(res)
    tol = max(atol, rtol*resnorm)
    toldx = max(atoldx, rtoldx*xnorm)
    it = 0
    if verbose:
        print("{} {:>3} {:^10} {:^10} {:^10} {:^9}".format("newton", "it", "|x|", "|dx|", '|r|', 'step'))
        print("{} {:3} {:10.3e} {:^10} {:10.3e} {:^9}".format("newton", it, xnorm, 3*'-', resnorm, 3*'-'))
    while( (resnorm>tol or dxnorm>toldx) and it < nit):
        it += 1
        if not computeUpdate:
            J = jac(x)
            dx = linalg.solve(J, res)
        else:
            dx = computeUpdate(res, x)
        dxnorm = linalg.norm(dx)
        x, res, resnorm, step, itbt = backtracking(computeResidual, x, dx, resnorm, firststep=firststep)
        xnorm = linalg.norm(x)
        if verbose:
            print("{} {:3} {:10.3e} {:10.3e} {:10.3e} {:9.2e}".format("newton", it, xnorm, dxnorm, resnorm, step))
        if xnorm >= divx:
            return (x, nit)
    return (x,it)


# ------------------------------------------------------ #

if __name__ == '__main__':
    f = lambda x: 10.0 * np.sin(2.0 * x) + 4.0 - x * x
    df = lambda x: 20.0 * np.cos(2.0 * x) - 2.0 * x
    # f = lambda x: x**2 -11
    # df = lambda x: 2.0 * x
    def residual(x):
        return -f(x)
    def update(r, x):
        return r/df(x)
    x = np.linspace(-1., 4.0)
    x0 = -2.
    x0 = [3.]
    # info = newton(x0, residual, jac=df, verbose=True)
    info = newton(x0, residual, computeUpdate=update, verbose=True)
    print(('info=', info))
    plt.plot(x, f(x), x, np.zeros_like(x))
    plt.show()
