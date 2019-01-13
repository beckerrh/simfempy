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

#ml = pyamg.ruge_stuben_solver(A)
#x = ml.solve(b, tol=1e-10)

#lu = umfpack.splu(A)
#x = umfpack.spsolve(A, b)

# https://github.com/bfroehle/pymumps
#from mumps import DMumpsContext

#=================================================================#
def is_symmetric(m):
    """Check if a sparse matrix is symmetric
        https://mail.python.org/pipermail/scipy-dev/2014-October/020117.html
        Parameters
        ----------
        m : sparse matrix

        Returns
        -------
        check : bool
    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, sparse.coo_matrix):
        m = sparse.coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False, "no_diag_sum", triu_no_diag.sum() - tril_no_diag.sum()

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check

#=================================================================#
class IterationCounter(object):
    def __init__(self, disp=20, name=""):
        self.disp = disp
        self.name = name
        self.niter = 0
    def __call__(self, rk=None):
        # if self.disp and self.niter%self.disp==0:
        #     print('iter({}) {:4d}\trk = {}'.format(self.name, self.niter, str(rk)))
        self.niter += 1
    def __del__(self):
        print('niter ({}) {:4d}'.format(self.name, self.niter))


#=================================================================#
class NewtonSolver(object):
    def __init__(self, **kwargs):
        self.timer = {'rhs':0.0, 'matrix':0.0, 'solve':0.0, 'bdry':0.0, 'postp':0.0}
        self.runinfo = {'niter':0}
        if 'problemname' in kwargs:
            self.problemname = kwargs.pop('problemname')
        self.linearsolvers=[]
        # self.linearsolvers.append('scipy-umf_mmd')
        self.linearsolvers.append('umf')
        # self.linearsolvers.append('gmres')
        self.linearsolvers.append('lgmres')
        # self.linearsolvers.append('bicgstab')
        try:
            import pyamg
            self.linearsolvers.append('pyamg')
        except: pass
        self.linearsolver = 'umf'

    def linearSolver(self, A, b, u=None, solver = 'umf'):
        # print("A is symmetric ? ", is_symmetric(A))
        if solver == 'umf':
            return splinalg.spsolve(A, b, permc_spec='COLAMD')
        # elif solver == 'scipy-umf_mmd':
        #     return splinalg.spsolve(A, b, permc_spec='MMD_ATA')
        elif solver in ['gmres','lgmres','bicgstab','cg']:
            M2 = splinalg.spilu(A, drop_tol=0.2, fill_factor=2)
            M_x = lambda x: M2.solve(x)
            M = splinalg.LinearOperator(A.shape, M_x)
            counter = IterationCounter(name=solver)
            args=""
            if solver == 'lgmres': args = ', inner_m=20, outer_k=4'
            cmd = "u = splinalg.{}(A, b, M=M, callback=counter {})".format(solver,args)
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
            u = ml.solve(b, tol=1e-12, residuals=res, accel = 'gmres')
            print("pyamg {:3d} ({:7.1e})".format(len(res),res[-1]/res[0]))
            return u
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