# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import time
import numpy as np
import scipy.sparse.linalg as splinalg
from tools.newton import newton
import scipy.optimize as optimize

class Solver(object):
    def __init__(self):
        self.timer = {'rhs':0.0, 'matrix':0.0, 'solve':0.0}

    def solveLinear(self):
        t0 = time.clock()
        b = self.computeRhs()
        t1 = time.clock()
        A = self.matrix()
        t2 = time.clock()
        u = splinalg.spsolve(A, b)
        t3 = time.clock()
        self.timer['rhs'] = t1-t0
        self.timer['matrix'] = t2-t1
        self.timer['solve'] = t3-t2
        return self.postProcess(u, self.timer)

    def residual(self, u):
        self.du[:]=0.0
        return self.form(self.du, u)- self.b

    def solvefornewton(self, x, b, redrate, iter):
        self.A = self.matrix(x)
        return splinalg.spsolve(self.A, b)
        import pyamg
        x = pyamg.solve(self.A, b, verb=True)
        return x
        # ilu = splinalg.spilu(self.A + 0.01*sparse.eye(self.A.shape[0]), fill_factor=2)
        # M = splinalg.LinearOperator(shape=self.A.shape, matvec=ilu.solve)
        # def linsolve(u):
        #     # return self.Amg.solve(u)
        #     print '--solve--'
        #     return splinalg.spsolve(self.A, u)
        # # M = None
        # M = linalg.LinearOperator(shape=self.A.shape, matvec=linsolve)
        #
        # def jacobian(x, res):
        #     print '--jac--'
        #     self.A = self.matrix(x)
        #     return self.A
        #
        # options = {'disp': True, 'jac_options': {'inner_M': M}}
        # sol = optimize.root(self.residual, u, method='Krylov', options=options, callback=jacobian, tol=1e-12)
        # u = optimize.newton_krylov(self.residual, u, inner_M=M, verbose=1)
        # sol = optimize.root(self.residual, u, method='broyden2')
        # print 'nit=', sol.nit
        # u = sol.x
        # u = linalg.spsolve(A, b)

    def newtonresidual(self, u):
        self.du = self.residual(u)
        # self.A = self.matrix(u)
        return splinalg.spsolve(self.A, self.du)
    def solvefornewtonresidual(self, x, b, redrate, iter):
        x = b
        return x

    def solveNonlinear(self, u=None, rtol=1e-10, gtol=1e-16, maxiter=100, checkmaxiter=True):
        t0 = time.clock()
        self.b = self.computeRhs()
        if u is None:
            u = np.zeros_like(self.b)
        else:
            assert u.shape == self.b.shape
        self.du = np.zeros_like(self.b)
        t1 = time.clock()
        self.A = self.matrix(u)
        t2 = time.clock()

        method = 'broyden2'
        method = 'anderson'
        method = 'krylov'
        method = 'df-sane'
        # method = 'ownnewton'
        # method = 'ownnewtonls'
        if method == 'ownnewton':
            u,res,nit = newton(self.residual, self.solvefornewton, u, rtol=1e-10, gtol=1e-14, maxiter=200)
        elif method == 'ownnewtonls':
            u,res,nit = newton(self.newtonresidual, self.solvefornewtonresidual, u, rtol=1e-10, gtol=1e-14, maxiter=200)
        else:
            self.A = self.matrix(u)
            sol = optimize.root(self.newtonresidual, u, method=method)
            u = sol.x
            nit = sol.nit

        # try:
        #     u,res,nit = newton(self.residual, solve, u, rtol=1e-10, gtol=1e-14)
        # except:
        #     nit = -1
        # print 'nit=', nit
        t3 = time.clock()
        self.timer['rhs'] = t1-t0
        self.timer['matrix'] = t2-t1
        self.timer['solve'] = t3-t2
        return self.postProcess(u, self.timer, nit)


# ------------------------------------- #

if __name__ == '__main__':
    raise ValueError("unit test to be written")