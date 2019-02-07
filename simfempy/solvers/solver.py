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

import simfempy.tools.analyticalsolution
import simfempy.tools.timer
import simfempy.tools.iterationcounter
import simfempy.applications.problemdata

# https://github.com/bfroehle/pymumps
#from mumps import DMumpsContext

#=================================================================#
class Solver(object):
    def generatePoblemData(self, exactsolution, bdrycond, postproc=None, random=True):
        problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc)
        problemdata.ncomp = self.ncomp
        problemdata.solexact = self.defineAnalyticalSolution(exactsolution=exactsolution, random=random)
        problemdata.rhs = self.defineRhsAnalyticalSolution(problemdata.solexact)
        if isinstance(bdrycond, (list, tuple)):
            if len(bdrycond) != self.ncomp: raise ValueError("length of bdrycond ({}) has to equal ncomp({})".format(len(bdrycond),self.ncomp))
            for color in self.mesh.bdrylabels:
                for icomp,bcs in enumerate(problemdata.bdrycond):
                    if bcs.type[color] == "Dirichlet":
                        bcs.fct[color] = problemdata.solexact[icomp]
                    else:
                        bcs.fct[color] = eval("self.define{}AnalyticalSolution_{:d}(problemdata.solexact)".format(bcs.type[color],icomp))
        else:
            if self.ncomp>1:
                def _solexactdir(x, y, z):
                    return [problemdata.solexact[icomp](x, y, z) for icomp in range(self.ncomp)]
            else:
                def _solexactdir(x, y, z):
                    return problemdata.solexact(x, y, z)
            for color in self.mesh.bdrylabels:
                if problemdata.bdrycond.type[color] == "Dirichlet":
                    problemdata.bdrycond.fct[color] = _solexactdir
                else:
                    problemdata.bdrycond.fct[color] = eval("self.define{}AnalyticalSolution(problemdata.solexact)".format(bdrycond.type[color]))
        return problemdata

    def defineAnalyticalSolution(self, exactsolution, random=True):
        dim = self.mesh.dimension
        return simfempy.tools.analyticalsolution.analyticalSolution(exactsolution, dim, self.ncomp, random)

    def __init__(self, **kwargs):
        self.ncomp = 1
        if 'ncomp' in kwargs: self.ncomp = kwargs.pop('ncomp')
        if 'geometry' in kwargs:
            geometry = kwargs.pop('geometry')
            self.mesh = simfempy.meshes.simplexmesh.SimplexMesh(geometry=geometry, hmean=1)
            showmesh = True
            if 'showmesh' in kwargs: showmesh = kwargs.pop('showmesh')
            if showmesh:
                self.mesh.plotWithBoundaries()
            return
        self.problemdata = kwargs.pop('problemdata')
        self.ncomp = self.problemdata.ncomp

        # temporary
        self.bdrycond = self.problemdata.bdrycond
        self.postproc = self.problemdata.postproc
        self.rhs = self.problemdata.rhs
        # temporary

        self.timer = simfempy.tools.timer.Timer(verbose=0)
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

    def solveLinear(self):
        self.timer.add('init')
        A = self.matrix()
        self.timer.add('matrix')
        b,u = self.computeRhs()
        self.timer.add('rhs')
        # A,b,u = self.boundary(A, b, u)
        # self.timer.add('boundary')
        u = self.linearSolver(A, b, u, solver=self.linearsolver)
        self.timer.add('solve')
        point_data, cell_data, info = self.postProcess(u)
        self.timer.add('postp')
        info['timer'] = self.timer.data
        return point_data, cell_data, info

    def linearSolver(self, A, b, u=None, solver = 'umf', verbose=1):
        if not hasattr(self, 'info'): self.info={}
        if solver == 'umf':
            return splinalg.spsolve(A, b, permc_spec='COLAMD')
        # elif solver == 'scipy-umf_mmd':
        #     return splinalg.spsolve(A, b, permc_spec='MMD_ATA')
        elif solver in ['gmres','lgmres','bicgstab','cg']:
            # defaults: drop_tol=0.0001, fill_factor=10
            M2 = splinalg.spilu(A.tocsc(), drop_tol=0.1, fill_factor=3)
            M_x = lambda x: M2.solve(x)
            M = splinalg.LinearOperator(A.shape, M_x)
            counter = simfempy.tools.iterationcounter.IterationCounter(name=solver, verbose=verbose)
            self.info['runinfo'] = counter.niter
            args=""
            cmd = "u = splinalg.{}(A, b, M=M, tol=1e-14, callback=counter {})".format(solver,args)
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
            u = ml.solve(b, x0=u, tol=1e-14, residuals=res, accel='gmres')
            if(verbose): print('niter ({}) {:4d} ({:7.1e})'.format(solver, len(res),res[-1]/res[0]))
            self.info['runinfo'] = len(res)
            return u
        else:
            raise ValueError("unknown solve '{}'".format(solver))

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