# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as splinalg

import simfempy.tools.analyticalfunction
import simfempy.tools.timer
import simfempy.tools.iterationcounter
import simfempy.applications.problemdata
from simfempy.tools.analyticalfunction import AnalyticalFunction

#=================================================================#
class Application(object):
    def __init__(self, **kwargs):
        self.linearsolvers=['umf', 'lgmres', 'bicgstab']
        try:
            import pyamg
            self.linearsolvers.append('pyamg')
        except:
            import warnings
            warnings.warn("*** pyamg not found (umf used instead)***")
        self.verbose = kwargs.pop('verbose', 0)
        self.linearsolver = kwargs.pop('linearsolver', 'umf')
        self.timer = simfempy.tools.timer.Timer(verbose=0)
        if 'problemdata' in kwargs:
            # self.problemdata = copy.deepcopy(kwargs.pop('problemdata'))
            self.problemdata = kwargs.pop('problemdata')
            self.ncomp = self.problemdata.ncomp
        if 'exactsolution' in kwargs:
            self.exactsolution = kwargs.pop('exactsolution')
            self._generatePDforES = True
            self.random_exactsolution = kwargs.pop('random',False)
        else:
            self._generatePDforES = False
        if 'geom' in kwargs:
            self.geom = kwargs.pop('geom')
            print(f"{self.geom=}")
            mesh = self.geom.generate_mesh()
            self.mesh = simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)
        if 'mesh' in kwargs:
            self.mesh = kwargs.pop('mesh')
        self._setMeshCalled = False
    def setMesh(self, mesh):
        self.mesh = mesh
        if self.verbose: print(f"{self.mesh=}")
        self._setMeshCalled = True
        if hasattr(self,'_generatePDforES') and self._generatePDforES:
            self.generatePoblemDataForAnalyticalSolution()
            self._generatePDforES = False
    def setParameter(self, paramname, param):
        assert 0
    def dirichletfct(self):
        if self.ncomp > 1:
            def _solexactdir(x, y, z):
                return [self.problemdata.solexact[icomp](x, y, z) for icomp in range(self.ncomp)]
        else:
            def _solexactdir(x, y, z):
                return self.problemdata.solexact(x, y, z)
        return _solexactdir
    def generatePoblemDataForAnalyticalSolution(self):
        bdrycond = self.problemdata.bdrycond
        self.problemdata.solexact = self.defineAnalyticalSolution(exactsolution=self.exactsolution, random=self.random_exactsolution)
        print("self.problemdata.solexact", self.problemdata.solexact)
        solexact = self.problemdata.solexact
        self.problemdata.params.fct_glob['rhs'] = self.defineRhsAnalyticalSolution(solexact)
        for color in self.mesh.bdrylabels:
            if color in bdrycond.type and bdrycond.type[color] in ["Dirichlet","dirichlet"]:
                bdrycond.fct[color] = self.dirichletfct()
            else:
                if color in bdrycond.type:
                    cmd = "self.define{}AnalyticalSolution(self.problemdata,{})".format(bdrycond.type[color], color)
                    # print(f"cmd={cmd}")
                    bdrycond.fct[color] = eval(cmd)
                else:
                    bdrycond.fct[color] = self.defineBdryFctAnalyticalSolution(color, solexact)
    def defineAnalyticalSolution(self, exactsolution, random=True):
        dim = self.mesh.dimension
        # print(f"defineAnalyticalSolution: {dim=} {self.ncomp=}")
        return simfempy.tools.analyticalfunction.analyticalSolution(exactsolution, dim, self.ncomp, random)
    def compute_cell_vector_from_params(self, name, params):
        if name in params.fct_glob:
            xc, yc, zc = self.mesh.pointsc.T
            fct = np.vectorize(params.fct_glob[name])
            arr = fct(self.mesh.cell_labels, xc, yc, zc)
        elif name in params.scal_glob:
            arr = np.full(self.mesh.ncells, params.scal_glob[name])
        elif name in params.scal_cells:
            arr = np.empty(self.mesh.ncells)
            for color in params.scal_cells[name]:
                arr[self.mesh.cellsoflabel[color]] = params.scal_cells[name][color]
        else:
            msg = f"{name} should be given in 'fct_glob' or 'scal_glob' or 'scal_cells' (problemdata.params)"
            raise ValueError(msg)
        return arr
    def static(self, iter=100, dirname='Run', mode='linear'):
        if mode != 'linear': raise NotImplementedError(f"Can only solve linear problems")
        # print(f"### static")
        if not self._setMeshCalled: self.setMesh(self.mesh)
        # self.timer.reset_all()
        result = simfempy.applications.problemdata.Results()
        self.timer.add('init')
        A = self.computeMatrix()
        self.timer.add('matrix')
        b, u = self.computeRhs()
        if np.linalg.norm(b)<1e-10: raise ValueError(f"rhs is zero {np.linalg.norm(b)=} {np.linalg.norm(u)=}")
        print(f"{np.linalg.norm(b)=}")
        print(f"{np.linalg.norm(u)=}")
        self.timer.add('rhs')
        import warnings
        # warnings.filterwarnings("error")
        try:
            u, niter = self.linearSolver(A, b, u, solver=self.linearsolver)
        except Warning:
            raise ValueError(f"matrix is singular {A.shape=} {A.diagonal()=}")
        self.timer.add('solve')
        pp = self.postProcess(u)
        self.timer.add('postp')
        result.setData(pp, timer=self.timer, iter={'lin':niter})
        return result
    def initialCondition(self, interpolate=True):
        #TODO: higher order interpolation
        if not 'initial_condition' in self.problemdata.params.fct_glob:
            raise ValueError(f"missing 'initial_condition' in {self.problemdata.params.fct_glob=}")
        if not self._setMeshCalled: self.setMesh(self.mesh)
        ic = AnalyticalFunction(self.problemdata.params.fct_glob['initial_condition'])
        fp1 = self.fem.interpolate(ic)
        if interpolate:
            return fp1
        self.Mass = self.fem.computeMassMatrix()
        b = np.zeros(self.fem.nunknowns())
        self.fem.massDot(b, fp1)
        u, niter = self.linearSolver(self.Mass, b, u=fp1)
        return u
    def dynamic(self, u0, t_span, nframes, dt=None, mode='linear', callback=None, method='CN', verbose=1):
        # TODO: passing time
        """
        u_t + A u = f, u(t_0) = u_0
        M(u^{n+1}-u^n)/dt + a Au^{n+1} + (1-a) A u^n = f
        (M/dt+aA) u^{n+1} =  f + (M/dt -(1-a)A)u^n
                          =  f + 1/a (M/dt) u^n -  (1-a)/a (M/dt+aA)u^n
        (M/(a*dt)+A) u^{n+1} =  (1/a)*f -  (1-a)/a (M/(a*dt)+A)u^n
        :param u0: initial condition
        :param t_span: time interval bounds (tuple)
        :param nframes: number of frames to store
        :param dt: time-step (fixed for the moment!)
        :param mode: (only linear for the moment!)
        :param callback: if given function called for each frame with argumntes t, u
        :param method: CN or BE for Crank-Nicolson (a=1/2) or backward Euler (a=1)
        :return: results with data per frame
        """
        if mode != 'linear': raise NotImplementedError(f"Can only solve linear problems")
        if not dt or dt<=0: raise NotImplementedError(f"needs constant positive 'dt")
        if t_span[0]>=t_span[1]: raise ValueError(f"something wrong in {t_span=}")
        if method not in ['BE','CN']: raise ValueError(f"unknown method {method=}")
        if method == 'BE': a = 1
        else: a = 0.5
        import math
        nitertotal = math.ceil((t_span[1]-t_span[0])/dt)
        if nframes > nitertotal:
            raise ValueError(f"Maximum valiue for nframes is {nitertotal=}")
        niter = nitertotal//nframes
        result = simfempy.applications.problemdata.Results()
        self.timer.add('init')
        if not hasattr(self, 'Mass'):
            self.Mass = self.fem.computeMassMatrix()
        if not hasattr(self, 'A'):
            self.Aimp = self.computeMatrix(coeff=a, coeffmass=1/dt)
            self.ml = self.build_pyamg(self.Aimp)
        self.timer.add('matrix')
        u = u0
        self.time = t_span[0]
        # rhs=None
        rhs=np.empty_like(u, dtype=float)
        # will be create by computeRhs()
        niterslinsol = np.zeros(niter)
        expl = (a-1)/a
        for iframe in range(nframes):
            if verbose: print(f"*** {self.time=} {iframe=} {niter=} {nframes=}")
            for iter in range(niter):
                self.time += dt
                rhs.fill(0)
                rhs += 1/(a*dt)*self.Mass.dot(u)
                rhs += expl*self.Aimp.dot(u)
                print(f"@1@{np.min(u)=} {np.max(u)=} {np.min(rhs)=} {np.max(rhs)=}")
                # rhs,u = self.computeRhs(b=rhs, u=u, coeffmass=1/(a*dt))
                rhs,up = self.computeRhs(b=rhs, coeff=1)
                print(f"@2@{np.min(u)=} {np.max(u)=} {np.min(rhs)=} {np.max(rhs)=}")
                self.timer.add('rhs')
                # u, niterslinsol[iter] = self.linearSolver(self.ml, rhs, u=u, verbose=0)
                u, res = self.solve_pyamg(self.ml, rhs, u=u)
                u, niterslinsol[iter] = u, len(res)
                print(f"@3@{np.min(u)=} {np.max(u)=} {np.min(rhs)=} {np.max(rhs)=}")
                self.timer.add('solve')
            result.addData(self.postProcess(u), time=self.time, iter=niterslinsol.mean())
            if callback: callback(self.time, u)
        return result

    def linearSolver(self, A, b, u=None, solver = None, verbose=0):
        # print(f"### linearSolver {solver=}")
        if spsp.issparse(A):
            if len(b.shape)!=1 or len(A.shape)!=2 or b.shape[0] != A.shape[0]:
                raise ValueError(f"{A.shape=} {b.shape=}")
        if solver is None: solver = self.linearsolver
        if not hasattr(self, 'info'): self.info={}
        if solver not in self.linearsolvers: solver = "umf"
        if solver == 'umf':
            return splinalg.spsolve(A, b, permc_spec='COLAMD'), 1
        elif solver in ['gmres','lgmres','bicgstab','cg']:
            if solver == 'cg':
                def gaussSeidel(A):
                    dd = A.diagonal()
                    D = spsp.dia_matrix(A.shape)
                    D.setdiag(dd)
                    L = spsp.tril(A, -1)
                    U = spsp.triu(A, 1)
                    return splinalg.factorized(D + L)
                M2 = gaussSeidel(A)
            else:
                # defaults: drop_tol=0.0001, fill_factor=10
                M2 = splinalg.spilu(A.tocsc(), drop_tol=0.1, fill_factor=3)
            M_x = lambda x: M2.solve(x)
            M = splinalg.LinearOperator(A.shape, M_x)
            counter = simfempy.tools.iterationcounter.IterationCounter(name=solver, verbose=verbose)
            args=""
            cmd = "u = splinalg.{}(A, b, M=M, tol=1e-14, callback=counter {})".format(solver,args)
            exec(cmd)
            return u, counter.niter
        elif solver == 'pyamg':
            ml = self.build_pyamg(A)
            u, res = self.solve_pyamg(ml, u, b)
            if(verbose): print('niter ({}) {:4d} ({:7.1e})'.format(solver, len(res),res[-1]/res[0]))
            return u, len(res)
        else:
            raise NotImplementedError("unknown solve '{}'".format(solver))
    def solve_pyamg(self, ml, b, u):
        res = []
        maxiter = 100
        SA_solve_args = {'cycle': 'V', 'maxiter': maxiter, 'tol': 1e-12, 'accel': 'bicgstab'}
        u = ml.solve(b=b, x0=u, residuals=res, **SA_solve_args)
        if len(res) >= maxiter: raise ValueError(f"***no convergence {res=}")
        return u, res
    def build_pyamg(self,A):
        import pyamg
        return pyamg.smoothed_aggregation_solver(A)
        B = np.ones((A.shape[0], 1))


    # def solveNonlinearProblem(self, u=None, sdata=None, method="newton", checkmaxiter=True):
    #     if not hasattr(self,'mesh'): raise ValueError("*** no mesh given ***")
    #     self.timer.add('init')
    #     A = self.matrix()
    #     self.timer.add('matrix')
    #     self.b,u = self.computeRhs(u)
    #     self.du = np.empty_like(u)
    #     self.timer.add('rhs')
    #     if method == 'newton':
    #         from . import newton
    #         u, nit = newton.newton(x0=u, f=self.residualNewton, computedx=self.solveForNewton, sdata=sdata, verbose=True)
    #     elif method in ['broyden2','krylov', 'df-sane', 'anderson']:
    #         sol = optimize.root(self.residualNewton, u, method=method)
    #         u, nit = sol.x, sol.nit
    #     else:
    #         raise ValueError(f"unknown method {method}")
    #     point_data, cell_data, info = self.postProcess(u)
    #     self.timer.add('postp')
    #     info['timer'] = self.timer
    #     info['iter'] = {'lin':nit}
    #     return point_data, cell_data, info

    # def residualNewton(self, u):
    #     # print(f"self.b={self.b.shape}")
    #     self.A = self.matrix()
    #     return self.A.dot(u) - self.b
    #
    # def solveForNewton(self, r, x):
    #     # print(f"solveForNewton r={np.linalg.norm(r)}")
    #     du, niter = self.linearSolver(self.A, r, x, verbose=0)
    #     # print(f"solveForNewton du={np.linalg.norm(du)}")
    #     return du



# ------------------------------------- #

if __name__ == '__main__':
    raise ValueError("unit test to be written")