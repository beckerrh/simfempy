# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import time
import numpy as np
import scipy.sparse.linalg as splinalg
import scipy.optimize as optimize
import scipy.sparse as sparse

import fempy.tools.analyticalsolution

# https://github.com/bfroehle/pymumps
#from mumps import DMumpsContext

#=================================================================#
class IterationCounter(object):
    def __init__(self, disp=20, name="", verbose=1):
        self.disp = disp
        self.name = name
        self.verbose = verbose
        self.niter = 0
    def __call__(self, rk=None):
        # if self.disp and self.niter%self.disp==0:
        #     print('iter({}) {:4d}\trk = {}'.format(self.name, self.niter, str(rk)))
        self.niter += 1
    def __del__(self):
        if self.verbose: print('niter ({}) {:4d}'.format(self.name, self.niter))


#=================================================================#
class Solver(object):
    def __init__(self, **kwargs):
        self.timer = {'rhs':0.0, 'matrix':0.0, 'solve':0.0, 'bdry':0.0, 'postp':0.0}
        self.runinfo = {'niter':0}
        self.linearsolvers=[]
        self.linearsolvers.append('umf')
        self.linearsolvers.append('lgmres')
        # self.linearsolvers.append('bicgstab')
        try:
            import pyamg
            self.linearsolvers.append('pyamg')
        except: pass
        self.linearsolver = 'umf'

        self.ncomp = 1
        if 'ncomp' in kwargs: self.ncomp = kwargs.pop('ncomp')
        if 'problem' in kwargs:
            if 'solexact' in kwargs: raise ValueError("not both 'problem' and 'solexact' can be specified")
            if 'problemname' in kwargs: raise ValueError("not both 'problem' and 'problemname' can be specified")
            random=True
            if 'random' in kwargs: random = kwargs.pop('random')
            self.solexact = self.defineAnalyticalSolution(problem=kwargs.pop('problem'), random=random)
        elif 'solexact' in kwargs:
            self.solexact = kwargs.pop('solexact')
        else:
            self.solexact = None
            self.bdrycond = kwargs.pop('bdrycond')
        if self.solexact:
            self.bdrycond = self.setAnalyticalBoundaryCondition(bdrycond=kwargs.pop('bdrycond'))
        if 'problemname' in kwargs: self.problemname = kwargs.pop('problemname')
        else: self.problemname="none"
        if 'postproc' in kwargs:
            self.postproc = kwargs.pop('postproc')
        else:
            self.postproc = {}
        if 'rhs' in kwargs:
            rhs = kwargs.pop('rhs')
            assert rhs is not None
            self.rhs = np.vectorize(rhs)
        else:
            self.rhs = None


    def defineAnalyticalSolution(self, problem, random=True):
        self.problemname = problem
        problemsplit = problem.split('_')
        if problemsplit[0] != 'Analytic':
            raise ValueError("unknown problem {}".format(problem))
        if len(problemsplit) != 3:
            raise ValueError("need three parts {}".format(problem))
        function = problemsplit[1]
        dim = int(problemsplit[2][0])
        solexact = fempy.tools.analyticalsolution.analyticalSolution(function, dim, self.ncomp, random)
        return solexact

    def setAnalyticalBoundaryCondition(self, bdrycond):
        if isinstance(bdrycond, (list, tuple)):
            if len(bdrycond) != self.ncomp: raise ValueError("length of bdrycond ({}) has to equal ncomp({})".format(len(bdrycond),self.ncomp))
            for icomp,bcs in enumerate(bdrycond):
                for color, bc in bcs.type.items():
                    if bc == "Dirichlet":
                        bcs.fct[color] = self.solexact[icomp]
                    elif bc == "Neumann":
                        bcs.fct[color] = None
                    else:
                        raise ValueError("unknown boundary condition {} for color {}".format(bc,color))
        else:
            def solexactall(x, y, z):
                return [self.solexact[icomp](x, y, z) for icomp in range(self.ncomp)]
            for color, bc in bdrycond.type.items():
                if bc == "Dirichlet":
                    if self.ncomp == 1:
                        bdrycond.fct[color] = self.solexact
                    else:
                        bdrycond.fct[color] = solexactall
                elif bc == "Neumann":
                    bdrycond.fct[color] = None
                else:
                    raise ValueError("unknown boundary condition {} for color {}".format(bc,color))
        return bdrycond

    def linearSolver(self, A, b, u=None, solver = 'umf', verbose=1):
        if solver == 'umf':
            return splinalg.spsolve(A, b, permc_spec='COLAMD')
        # elif solver == 'scipy-umf_mmd':
        #     return splinalg.spsolve(A, b, permc_spec='MMD_ATA')
        elif solver in ['gmres','lgmres','bicgstab','cg']:
            # defaults: drop_tol=0.0001, fill_factor=10
            M2 = splinalg.spilu(A.tocsc(), drop_tol=0.1, fill_factor=3)
            M_x = lambda x: M2.solve(x)
            M = splinalg.LinearOperator(A.shape, M_x)
            counter = IterationCounter(name=solver, verbose=verbose)
            args=""
            cmd = "u = splinalg.{}(A, b, M=M, tol=1e-12, callback=counter {})".format(solver,args)
            exec(cmd)
            return u
        elif solver == 'pyamg':
            import pyamg
            config = pyamg.solver_configuration(A, verb=False)
            # ml = pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='energy')
            # ml = pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='jacobi')
            ml = pyamg.rootnode_solver(A, B=config['B'], smooth='energy')
            # print("ml", ml)
            res=[]
            # if u is not None: print("u norm", np.linalg.norm(u))
            u = ml.solve(b, x0=u, tol=1e-12, residuals=res, accel='gmres')
            if(verbose): print('niter ({}) {:4d} ({:7.1e})'.format(solver, len(res),res[-1]/res[0]))
            return u
        else:
            raise ValueError("unknown solve '{}'".format(solver))

    def solveLinear(self):
        t0 = time.time()
        b = self.computeRhs()
        u = np.zeros_like(b)
        t1 = time.time()
        A = self.matrix()
        t2 = time.time()
        A,b,u = self.boundary(A, b, u)
        t3 = time.time()
        u = self.linearSolver(A, b, u, solver=self.linearsolver)
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