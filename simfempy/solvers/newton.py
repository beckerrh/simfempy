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
    import newtondata
else:
    from . import newtondata

#----------------------------------------------------------------------
def backtracking(f, x0, dx, resfirst, sdata, verbose=False):
    maxiter, omega, c = sdata.bt_maxiter, sdata.bt_omega, sdata.bt_c
    step = 1
    x = x0 + step*dx
    res = f(x)
    resnorm = np.linalg.norm(res)
    it = 0
    if verbose:
        print("{} {:>3} {:^10} {:^10}  {:^9}".format("bt", "it", "resnorm", "resfirst", "step"))
        print("{} {:3} {:10.3e} {:10.3e}  {:9.2e}".format("bt", it, resnorm, resfirst, step))
    while resnorm > (1-c*step)*resfirst and it<maxiter:
        it += 1
        step *= omega
        x = x0 + step * dx
        res = f(x)
        resnorm = np.linalg.norm(res)
        if verbose:
            print("{} {:3} {:10.3e}  {:9.2e}".format("bt", it, resnorm, step))
    return x, res, resnorm, step, it

#----------------------------------------------------------------------
def newton(x0, f, computedx=None, sdata=None, verbose=False, jac=None, maxiter=None, resred=0.1):
    """
    Aims to solve f(x) = 0, starting at x0
    computedx: gets dx from f'(x) dx =  -f(x)
    if not given, jac is called and linalg.solve is used
    """
    if sdata is None:
        if maxiter is None: raise ValueError(f"if sdata is None please give 'maxiter'") 
        sdata = newtondata.StoppingData(maxiter=maxiter)
    atol, rtol, atoldx, rtoldx = sdata.atol, sdata.rtol, sdata.atoldx, sdata.rtoldx
    maxiter, divx = sdata.maxiter, sdata.divx
    x = np.asarray(x0)
    assert x.ndim == 1
    # n = x.shape[0]
    # print(f"{x0=}")
    if not computedx:  assert jac
    xnorm = np.linalg.norm(x)
    dxnorm = xnorm
    res = f(x)
    resnorm = np.linalg.norm(res)
    tol = max(atol, rtol*resnorm)
    print(f"{tol=}")
    toldx = max(atoldx, rtoldx*xnorm)
    it = 0
    rhor = 1
    if verbose:
        print("{} {:>3} {:^10} {:^10} {:^10} {:^9} {:^5} {:^5} {:^3}".format("newton", "it", "|x|", "|dx|", '|r|', 'step','rhor','rhodx','lin'))
        print("{} {:3} {:10.3e} {:^10} {:10.3e} {:^9} {:^5} {:^5} {:^3}".format("newton", it, xnorm, 3*'-', resnorm, 3*'-', 3*'-', 3*'-', 3*'-'))
    # while( (resnorm>tol or dxnorm>toldx) and it < maxiter):
    dx, step, resold = None, None, np.zeros_like(res)
    while(resnorm>tol  and it < maxiter):
        it += 1
        if not computedx:
            J = jac(x)
            dx, liniter = linalg.solve(J, -res), 1
        else:
            dx, liniter = computedx(-res, x, (it,rhor,dx, step, res-resold))
        dxnormold = dxnorm
        dxnorm = linalg.norm(dx)
        resnormold = resnorm
        resold[:] = res[:]
        x, res, resnorm, step, itbt = backtracking(f, x, dx, resnorm, sdata)
        rhor, rhodx = resnorm/resnormold, dxnorm/dxnormold
        xnorm = linalg.norm(x)
        if verbose:
            print(f"newton {it:3} {xnorm:10.3e} {dxnorm:10.3e} {resnorm:10.3e} {step:9.2e} {rhor:5.2f} {rhodx:5.2f} {liniter:3d}")
        if xnorm >= divx:
            return (x, maxiter)
    return (x,it)


# ------------------------------------------------------ #

if __name__ == '__main__':
    f = lambda x: 10.0 * np.sin(2.0 * x) + 4.0 - x * x
    df = lambda x: 20.0 * np.cos(2.0 * x) - 2.0 * x
    f = lambda x: x**2 -11
    df = lambda x: 2.0 * x
    def computedx(r, x, info):
        return r/df(x),1
    x0 = [3.]
    info = newton(x0, f, jac=df, verbose=True, maxiter=10)
    info2 = newton(x0, f, computedx=computedx, verbose=True, maxiter=10)
    print(('info=', info))
    assert info==info2
    x = np.linspace(-1., 4.0)
    plt.plot(x, f(x), [x[0], x[-1]], [0,0], '--r')
    plt.show()
