# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import time
import numpy as np
import scipy.sparse.linalg as splinalg
import scipy.optimize as optimize

#ml = pyamg.ruge_stuben_solver(A)
#x = ml.solve(b, tol=1e-10)

#lu = umfpack.splu(A)
#x = umfpack.spsolve(A, b)

# https://github.com/bfroehle/pymumps
#from mumps import DMumpsContext

class NewtonSolver(object):
    def __init__(self):
        self.timer = {'rhs':0.0, 'matrix':0.0, 'solve':0.0, 'bdry':0.0, 'postp':0.0}
        self.runinfo = {'niter':0}
        self.linearsolvers=[]
        self.linearsolvers.append('scipy')
        try:
            import pyamg
            self.linearsolvers.append('pyamg')
        except: pass
        try:
            from scikits import umfpack
            self.linearsolvers.append('umfpack')
        except: pass

    def linearSolver(self, A, b, u, solver = 'pyamg'):
        if solver == 'scipy':
            return splinalg.spsolve(A, b)
        elif solver == 'pyamg':
            import pyamg
            res=[]
            u = pyamg.solve(A, b, x0=u, tol=1e-12, residuals=res, verb=False)
            print("pyamg {:3d} ({:7.1e})".format(len(res),res[-1]/res[0]))
            # msg=""
            # for i, r in enumerate(res):
            #     msg += "{1:8.2e}({0:3d})  ".format(i,r)
            # print(msg)
            return u
        elif solver == 'umfpack':
            from scikits import umfpack
            return umfpack.spsolve(A, b)
        else:
            raise ValueError("unknown solve '{}'".format(solver))

        # ml = pyamg.ruge_stuben_solver(A)
        # B = np.ones((A.shape[0], 1))
        # ml = pyamg.smoothed_aggregation_solver(A, B, max_coarse=10)
        # res = []
        # # u = ml.solve(b, tol=1e-10, residuals=res)
        # u = pyamg.solve(A, b, tol=1e-10, residuals=res, verb=False,accel='cg')
        # for i, r in enumerate(res):
        #     print("{:2d} {:8.2e}".format(i,r))
        # lu = umfpack.splu(A)
        # u = umfpack.spsolve(A, b)

    def solveLinear(self):
        t0 = time.time()
        b = self.computeRhs()
        u = np.zeros_like(b)
        t1 = time.time()
        A = self.matrix()
        t2 = time.time()
        A,b,u = self.boundary(A, b, u)
        t3 = time.time()
        u = self.linearSolver(A, b, u)
        t4 = time.time()
        pp = self.postProcess(u)
        t5 = time.time()
        self.timer['rhs'] = t1-t0
        self.timer['matrix'] = t2-t1
        self.timer['bdry'] = t3-t2
        self.timer['solve'] = t4-t3
        self.timer['postp'] = t5-t4
        return pp

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
        t0 = time.time()
        self.b = self.computeRhs()
        if u is None:
            u = np.zeros_like(self.b)
        else:
            assert u.shape == self.b.shape
        self.du = np.zeros_like(self.b)
        t1 = time.time()
        self.A = self.matrix(u)
        t2 = time.time()

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
        t3 = time.time()
        pp = self.postProcess(u)
        t4 = time.time()
        self.timer['rhs'] = t1-t0
        self.timer['matrix'] = t2-t1
        self.timer['solve'] = t3-t2
        self.timer['postproc'] = t4-t3
        return pp


# ------------------------------------- #

if __name__ == '__main__':
    raise ValueError("unit test to be written")