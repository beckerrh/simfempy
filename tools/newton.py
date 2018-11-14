#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:38:16 2016

@author: becker
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def newton(f, solve, x, rtol=1e-10, gtol=1e-14, maxiter=100, checkmaxiter=False, silent=False, simple=False):
    """
    def newton(f, rtol, n=100):
            f, df : function objects
    """
    it = 0
    residual = f(x)
    resn = linalg.norm(residual)
    if resn <= gtol:
        print 'newton(): first residual is zero'
        return x, resn, 0
    rtol *= resn
    if gtol>rtol:  rtol = gtol
    if not silent: print 'it=', it, 'res, dx=', resn, rtol
    nbaditeration = 0
    redrate = -1
    while resn > rtol:
        if it == maxiter:
            if not checkmaxiter: return  x, resn, it
            raise ValueError('no convergence: maxiter reached')
        resnold = resn
        dx = solve(x, residual, redrate, it)
        # print 'it=', it,  'res, dx=', resn, linalg.norm(dx)
        xold = x
        nn = 0
        omega = 1.0
        nnmax = 20
        if simple: nnmax=1
        while 1:
            if nn == nnmax:
                if simple: break
                raise ValueError('no relaxation: omega=%g' % (omega))
            x = xold - omega * dx
            residual = f(x)
            resn = linalg.norm(residual)
            redrate = resn/resnold
            if resn <= resnold:
                resnold = resn
                if nn >=5:
                    nbaditeration += 1
                elif nn <=1:
                    nbaditeration = 0
                if nbaditeration >= 3:
                    raise ValueError('too many bad iterations in a row: nbaditeration=%d' % (nbaditeration))
                break
            omega *= np.power(0.6, nn+1)
            nn += 1
            print 'nn=', nn, 'resn/resnold', resn / resnold, 'omega', omega
        it += 1
        if not silent: print 'it=', it, 'res, dx=', resn, linalg.norm(dx)
    return x, resn, it


# ------------------------------------------------------ #

if __name__ == '__main__':
    # f = lambda x : np.exp(2.0*x)+3.0*x-10.0 + 12.0*np.sin(3.0*x)
    # df = lambda x : 2.0*np.exp(2.0*x)+3.0+ 36.0*np.cos(3.0*x)
    # f = lambda x : (x-1)*2 + x**3
    # df = lambda x : 2.0*(x-1) + 3.0*x**2
    f = lambda x: 10.0 * np.sin(2.0 * x) + 4.0 - x * x
    df = lambda x: 20.0 * np.cos(2.0 * x) - 2.0 * x
    def solve(x, b):
        return b/df(x)
    x = np.linspace(0.0, 2.0)
    y = f(x)
    plt.plot(x, y, x, np.zeros_like(x))
    plt.show()
    x0 = -2.
    x0 = 3.
    info = newton(f, solve, x0)
    print('info=', info)
