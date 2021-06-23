import numpy as np
import pyamg
import scipy.sparse.linalg as splinalg
import scipy.sparse as sparse
from simfempy import tools
import time

scipysolvers=['gmres','lgmres','gcrotmk','bicgstab','cgs']

#-------------------------------------------------------------------#
def selectBestSolver(solvers, reduction, b, **kwargs):
    maxiter = kwargs.pop('maxiter', 100)
    verbose = kwargs.pop('verbose', 0)
    tol = kwargs.pop('tol') if not 'tol' in kwargs else 0.1*reduction
    analysis = {}
    for solvername, solver in solvers.items():
        t0 = time.time()
        res = solver.testsolve(b=b, maxiter=maxiter, tol=tol)
        t = time.time() - t0
        monotone = np.all(np.diff(res) < 0)
        if len(res)==1:
            if res[0] > 1e-6: raise ValueError(f"no convergence in {solvername=} {res=}")
            iterused = 1
        else:
            rho = np.power(res[-1]/res[0], 1/len(res))
            if not monotone:
                print(f"***VelcoitySolver {solvername} not monotone {rho=}")
                continue
            if rho > 0.8: 
                print(f"***VelcoitySolver {solvername} bad {rho=}")
                continue
            iterused = int(np.log(reduction)/np.log(rho))+1
        treq = t/len(res)*iterused
        analysis[solvername] = (iterused, treq)
    # print(f"{self.analysis=}")
    if verbose:
        for solvername, val in analysis.items():
            print(f"{solvername=} {val=}")
    ibest = np.argmin([v[1] for v in analysis.values()])
    solverbest = list(analysis.keys())[ibest]
    # print(f"{solverbest=}")
    # self.solver = self.solvers[solverbest]
    return solverbest, analysis[solverbest][0]


#=================================================================#
class ScipySpSolve():
    def __init__(self, **kwargs):
        self.matrix = kwargs.pop('matrix')
    def solve(self, b, maxiter, tol, x0=None):
        return splinalg.spsolve(self.matrix, b)
    def testsolve(self, b, maxiter, tol):
        splinalg.spsolve(self.matrix, b)
        return [0]

#=================================================================#
class ScipySolve():
    def __init__(self, **kwargs):
        self.method = kwargs.pop('method')
        if "matrix" in kwargs:
            self.matvec = kwargs.pop('matrix')
            if not "matvecprec" in kwargs:
                # spilu = splinalg.spilu(self.matvec.tocsc(), drop_tol=0.1, fill_factor=2)
                spilu = splinalg.spilu(self.matvec.tocsc(), drop_tol=0.01, fill_factor=1)
                self.M = splinalg.LinearOperator(self.matvec.shape, lambda x: spilu.solve(x))
        else:
            # self.matvec = kwargs.pop('matvec')
            if not 'n' in kwargs: raise ValueError(f"need 'n' if no matrix given")
            n = kwargs.get('n')
            self.matvec = splinalg.LinearOperator(shape=(n, n), matvec=kwargs.pop('matvec'))
        if "matvecprec" in kwargs:
            n = kwargs.get('n')
            self.M = splinalg.LinearOperator(shape=(n, n), matvec=kwargs.pop('matvecprec'))
        else:
            self.M = None
        self.atol = 1e-14
        disp = kwargs.pop('disp', 0)
        # print(f"**** {disp=}")
        self.args = {"A": self.matvec, "M":self.M, "atol":self.atol}
        self.solver = eval('splinalg.'+self.method)
        name = self.method
        if self.method=='gcrotmk':
            self.args['m'] = kwargs.pop('m', 5)
            self.args['truncate'] = kwargs.pop('truncate', 'smallest')
            self.solver = splinalg.gcrotmk
            name += '_' + str(self.args['m'])
        if 'counter' in kwargs:
            self.counter = tools.iterationcounter.IterationCounter(name=kwargs.pop('counter')+name, disp=disp)
            self.args['callback'] = self.counter
    def solve(self, b, maxiter, tol, x0=None):
        # print(f"**** {maxiter=}")
        if hasattr(self, 'counter'): self.counter.reset()
        self.args['b'] = b
        self.args['maxiter'] = maxiter
        self.args['x0'] = x0
        self.args['tol'] = tol
        u, info = self.solver(**self.args)
        # if info: raise ValueError(f"no convergence in {self.method=} {info=}")
        return u
    def testsolve(self, b, maxiter, tol):
        # print(f"{np.linalg.norm(b)=} {maxiter=} {tol=}")
        counter = tools.iterationcounter.IterationCounterWithRes(name=self.method, callback_type='x', disp=0, b=b, A=self.matvec)
        args = self.args.copy()
        args["callback"] = counter
        args['b'] = b
        args['maxiter'] = maxiter
        args['tol'] = tol
        u, info = self.solver(**args)
        # print(f"{counter.res=}")
        return counter.history

#=================================================================#
class Pyamg():
    def __init__(self, A, **kwargs):
        self.nsmooth = kwargs.pop('nsmooth', 2)
        self.smoother = kwargs.pop('smoother', 'schwarz')
        # self.smoother = kwargs.pop('smoother', 'strength_based_schwarz')
        # self.smoother = kwargs.pop('smoother', 'block_gauss_seidel')
        smooth = ('energy', {'krylov': 'fgmres'})
        smoother = (self.smoother, {'sweep': 'symmetric', 'iterations': self.nsmooth})
        pyamgargs = {'B': pyamg.solver_configuration(A, verb=False)['B'], 'smooth': smooth, 'presmoother':smoother, 'postsmoother':smoother}
        pyamgargs['symmetry'] = 'nonsymmetric'
        pyamgargs['coarse_solver'] = 'splu'
        self.solver = pyamg.smoothed_aggregation_solver(A, **pyamgargs)
    def testsolve(self, b, maxiter, tol):
        res = []
        self.solver.solve(b, maxiter=maxiter, tol=tol, residuals=res)
        return np.asarray(res)
    def solve(self, b, maxiter, tol):
        return self.solver.solve(b, maxiter=maxiter, tol=tol)
