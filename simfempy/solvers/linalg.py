import numpy as np
import scipy.sparse.linalg as splinalg
import scipy.sparse as sparse
from simfempy import tools
import time

scipysolvers=['scipy_gmres','scipy_lgmres','scipy_gcrotmk','scipy_bicgstab','scipy_cgs', 'scipy_cg']
pyamgsolvers=['pyamg_fgmres','pyamg_bicgstab', 'pyamg_cg']
# pyamg_gmres seems to not work correctly
strangesolvers=['gmres']
othersolvers=['idr']


#=================================================================#
def matrix2systemdiagonal(A, ncomp):
    """
    creates a blockmatrix with A on the diaganals
    hyothesis: vector is stored as v[::icomp] and NOT one after the other
    :param self:
    :param A:
    :param ncomp:
    :return:
    """
    A = A.tocoo()
    data, row, col, shape = A.data, A.row, A.col, A.shape
    n = shape[0]
    assert n == shape[1]
    data2 = np.repeat(data, ncomp)
    nr = row.shape[0]
    row2 = np.repeat(ncomp * row, ncomp) + np.tile(np.arange(ncomp), nr).ravel()
    col2 = np.repeat(ncomp * col, ncomp) + np.tile(np.arange(ncomp), nr).ravel()
    return sparse.coo_matrix((data2, (row2, col2)), shape=(ncomp * n, ncomp * n)).tocsr()
def diagmatrix2systemdiagmatrix(A, ncomp):
    """
    creates a blockmatrix with A on the diaganals
    hyothesis: vector is stored as v[::icomp] and NOT one after the other
    :param self:
    :param A:
    :param ncomp:
    :return:
    """
    data = A.data
    n = A.shape[0]
    data = np.repeat(data, ncomp)
    return sparse.diags(data, offsets=(0), shape=(ncomp*n, ncomp*n))

#=================================================================#
class DiagonalScaleSolver():
    def __repr__(self):
        return f"{self.__class__.__name__}"
    def __init__(self, coeff):
        n = len(coeff)
        self.BP = sparse.diags(coeff, offsets=(0), shape=(n,n))
    def solve(self, b):
        return self.BP.dot(b)

#=================================================================#
class SaddlePointSystem():
    """
    A -B.T
    B  0
     or
    A -B.T 0
    B  0   M^T
    0  M   0
    """
    def __repr__(self):
        string =  f" {self.singleA=} {self.A.shape=} {self.B.shape=}"
        if hasattr(self, 'M'): string += f"{self.M.shape=}"
        return string
    def __init__(self, A, B, M=None, singleA=True, ncomp=None):
        if singleA and not ncomp:
            raise ValueError(f"{singleA=} needs 'ncomp'")
        if singleA:
            self.na = ncomp*A.shape[0]
        else:
            self.na = A.shape[0]
        self.singleA, self.ncomp = singleA, ncomp
        self.A, self.B = A, B
        self.nb = B.shape[0]
        self.nall = self.na + self.nb
        if M is not None:
            self.M = M
            self.nm = M.shape[0]
            self.nall += self.nm
        self.constr = hasattr(self, 'M')
        if self.singleA:
            self.matvec = self.matvec3_singleA if self.constr else self.matvec2_singleA
        else:
            self.matvec = self.matvec3 if self.constr else self.matvec2
    def _dot_A(self, b):
        if self.singleA:
            w = np.empty_like(b)
            for i in range(self.ncomp): w[i::self.ncomp] = self.A.dot(b[i::self.ncomp])
            return w
        return self.A.dot(b)
    def copy(self):
        M = None
        if hasattr(self, 'M'): M = self.M.copy()
        return SaddlePointSystem(self.A.copy(), self.B.copy(), M=M, singleA=self.singleA, ncomp=self.ncomp)
    def scale_A(self):
        assert not self.constr
        DA = self.A.diagonal()
        assert np.all(DA>0)
        if self.singleA:
            AD = diagmatrix2systemdiagmatrix(1/DA, self.ncomp)
        else:
            AD = sparse.diags(1/DA, offsets=(0), shape=self.A.shape)
        DS = (self.B@AD@self.B.T).diagonal()
        print(f"scale_A {DA.max()/DA.min()=:8.2f} {DS.max()/DS.min()=:8.2f}")
        assert np.all(DS>0)
        nb = self.B.shape[0]
        self.vs = sparse.diags(np.power(AD.diagonal(), 0.5), offsets=(0), shape=AD.shape)
        self.ps = sparse.diags(np.power(DS, -0.5), offsets=(0), shape=(nb,nb))
        # print(f"{self.vs.data=}\n{self.ps.data=}")
        if self.singleA:
            # vss = sparse.dia_array((np.power(DA, -0.5), (0)), shape=self.A.shape)
            vss = sparse.diags(np.power(DA, -0.5), offsets=(0), shape=self.A.shape)
            self.A = vss @ self.A @ vss
        else:
            self.A = self.vs@self.A@self.vs
        if not np.allclose(self.A.diagonal(), np.ones(self.A.shape[0])):
            raise ValueError(f"not ones on diagona\n{self.A.diagonal()=}")
        self.B = self.ps@self.B@self.vs
    def scale_vec(self, b):
        # print(f"scale_rhs")
        bv, bp = b[:self.na], b[self.na:]
        b[:self.na] = self.vs@bv
        b[self.na:] = self.ps@bp
    # def scale_sol(self, u):
    #     print(f"scale_sol")
    #     v, p = u[:self.na], u[self.na:]
    #     u[:self.na] = self.vs@v
    #     u[self.na:] = self.ps@p
    def dot(self, x):
        if hasattr(self, "M"): return self.matvec3(x)
        return self.matvec2(x)
    def matvec3(self, x):
        v, p, lam = x[:self.na], x[self.na:self.na+self.nb], x[self.na+self.nb:]
        w = self.A.dot(v) - self.B.T.dot(p)
        q = self.B.dot(v)+ self.M.T.dot(lam)
        return np.hstack([w, q, self.M.dot(p)])
    def matvec2(self, x):
        v, p = x[:self.na], x[self.na:]
        # w = self.A.dot(v) - self.B.T.dot(p)
        w = self._dot_A(v) - self.B.T.dot(p)
        q = self.B.dot(v)
        return np.hstack([w, q])
    def matvec3_singleA(self, x):
        v, p, lam = x[:self.na], x[self.na:self.na+self.nb], x[self.na+self.nb:]
        w = - self.B.T.dot(p)
        for icomp in range(self.ncomp): w[icomp::self.ncomp] += self.A.dot(v[icomp::self.ncomp])
        q = self.B.dot(v)+ self.M.T.dot(lam)
        return np.hstack([w, q, self.M.dot(p)])
    def matvec2_singleA(self, x):
        v, p = x[:self.na], x[self.na:]
        w = - self.B.T.dot(p)
        for icomp in range(self.ncomp): w[icomp::self.ncomp] += self.A.dot(v[icomp::self.ncomp])
        q = self.B.dot(v)
        return np.hstack([w, q])
    def to_single_matrix(self):
        nullP = sparse.dia_matrix((np.zeros(self.nb), 0), shape=(self.nb, self.nb))
        if self.singleA:
            A = matrix2systemdiagonal(self.A, self.ncomp)
        else:
            A = self.A
        A1 = sparse.hstack([A, -self.B.T])
        A2 = sparse.hstack([self.B, nullP])
        Aall = sparse.vstack([A1, A2])
        if not hasattr(self, 'M'):
            return Aall.tocsr()
        nullV = sparse.coo_matrix((1, self.na)).tocsr()
        ML = sparse.hstack([nullV, self.M])
        Abig = sparse.hstack([Aall, ML.T])
        nullL = sparse.dia_matrix((np.zeros(1), 0), shape=(1, 1))
        Cbig = sparse.hstack([ML, nullL])
        Aall = sparse.vstack([Abig, Cbig])
        return Aall.tocsr()

#-------------------------------------------------------------------#
def _getSolver(args):
    if not isinstance(args, dict): raise ValueError(f"*** args must be a dict")
    if not 'method' in args: raise ValueError(f"*** needs 'method' in args\ngiven: {args}")
    method = args.pop('method')
    matrix = args.pop('matrix', None)
    if method in scipysolvers or method in pyamgsolvers or method in othersolvers:
        if matrix is not None:
            return ScipySolve(matrix=matrix, method=method, **args)
        else:
            return ScipySolve(method=method, **args)
    elif method == "spsolve":
        # if matrix is None: raise ValueError(f"matrix is None for {args=}")
        return ScipySpSolve(matrix=matrix)
    elif method == "pyamg":
        # if matrix is None: raise ValueError(f"matrix is None for {args=}")
        return Pyamg(matrix, **args)
    else:
        raise ValueError(f"unknwown {method=} not in ['spsolve', 'pyamg', {pyamgsolvers}]")
#-------------------------------------------------------------------#
def getLinearSolver(**kwargs):
    """
    :param kwargs: if args is dict build the correspong solver
    otherwise if args is list, choose the best solver in the list
    :return:
    """
    args = kwargs.pop('args')
    if isinstance(args, dict):
        if len(kwargs): raise ValueError(f"*** unused keys {kwargs}")
        return _getSolver(args)
    assert isinstance(args, list)
    maxiter = args.pop('maxiter', 50)
    verbose = args.pop('verbose', 0)
    reduction = args.pop('reduction', 0.01)
    rtol = args.pop('rtol') if 'rtol' in args else 0.1*reduction
    solvers = {}
    for arg in args:
        solvers[arg] = _getSolver(arg)
        n = solvers[arg].shape[0]
    b = np.random.random(n)
    b /= np.linalg.norm(b)
    analysis = {}
    for solvername, solver in solvers.items():
        t0 = time.time()
        res = solver.testsolve(b=b, maxiter=maxiter, rtol=rtol)
        t = time.time() - t0
        monotone = np.all(np.diff(res) < 0)
        if len(res)==1:
            if res[0] > rtol:
                print(f"no convergence in {solvername=} {res=}")
                continue
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
    if len(analysis)==0: raise ValueError('*** no working solver found')
    ibest = np.argmin([v[1] for v in analysis.values()])
    solverbest = list(analysis.keys())[ibest]
    if verbose:
        print(f"{solverbest=}")
    return solvers[solverbest], analysis[solverbest][0]
#=================================================================#
class ScipySpSolve():
    def __init__(self, **kwargs):
        self.matrix = kwargs.pop('matrix', None)
        self.niter = 1
    def solve(self, A=None, b=None, maxiter=None, rtol=None, atol=None, x0=None, verbose=None):
        if A is None: A=self.matrix
        if hasattr(A, 'to_single_matrix'):
            A = A.to_single_matrix()
        return splinalg.spsolve(A, b)
    def testsolve(self, b, maxiter, rtol):
        splinalg.spsolve(self.matrix, b)
        return [0]
#=================================================================#
class IterativeSolver():
    def __repr__(self):
        return f"{self.method}_{self.maxiter}_{self.rtol}"
    def __init__(self, **kwargs):
        self.args = {}
        self.scale = kwargs.pop('scale', False)
        # print(f"{self.scale=} {self.__class__.__name__}")
        self.atol = kwargs.pop('atol', 1e-14)
        self.rtol = kwargs.pop('rtol', 1e-8)
        self.maxiter = kwargs.pop('maxiter', 100)
        # if 'counter' in kwargs:
        disp = kwargs.pop('disp', 0)
        self.counter = tools.iterationcounter.IterationCounter(name=kwargs.pop('counter','')+str(self), disp=disp)
        self.args['callback'] = self.counter
    def solve(self, A=None, b=None, maxiter=None, rtol=None, atol=None, x0=None, verbose=None):
        # print(f"{maxiter=} {self.maxiter=}")
        # print(f"{rtol=} {self.args=}")
        if maxiter is None: maxiter = self.maxiter
        if rtol is None: rtol = self.rtol
        # print(f"{self.__class__.__name__} {rtol=} {maxiter=}")
        if hasattr(self, 'counter'):
            self.counter.reset()
            self.args['callback'] = self.counter
        if self.scale and A and hasattr(A,'scale_A'):
            A.scale_vec(b)
        self.args['b'] = b
        self.args['maxiter'] = maxiter
        self.args['x0'] = x0
        self.args['tol'] = rtol
        res  = self.solver(**self.args)
        # print(f"{self.counter.niter=} {len(res)=} {type(res)=} {res[0]=} {res[1]=}")
        if hasattr(self, 'counter'):
            self.niter = self.counter.niter
        else:
            self.niter = -1
        sol = res[0] if isinstance(res, tuple) else res
        if self.scale and A and hasattr(A,'scale_A'):
            A.scale_vec(sol)
        return sol

    def testsolve(self, b, maxiter, rtol):
        counter = tools.iterationcounter.IterationCounterWithRes(name=str(self), callback_type='x', disp=0, b=b, A=self.matvec)
        args = self.args.copy()
        args['callback'] = counter
        args['maxiter'] = maxiter
        args['tol'] = rtol
        args['b'] = b
        res = self.solver(**args)
        return counter.history
#=================================================================#
class ScipySolve(IterativeSolver):
    def __init__(self, **kwargs):
        self.method = kwargs.pop('method')
        super().__init__(**kwargs)
        # if self.method in strangesolvers: raise ValueError(f"method '{self.method}' is i strange scipy solver")
        if "prec" in kwargs:
            self.M = kwargs.pop("prec")
        if "matrix" in kwargs:
            self.matvec = kwargs.pop('matrix')
            if not "matvecprec" in kwargs:
                fill_factor = kwargs.pop("fill_factor", 2)
                drop_tol = kwargs.pop("fill_factor", 0.01)
                spilu = splinalg.spilu(self.matvec.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
                self.M = splinalg.LinearOperator(self.matvec.shape, lambda x: spilu.solve(x))
        else:
            if not 'n' in kwargs: raise ValueError(f"need 'n' if no matrix given")
            n = kwargs.get('n')
            # raise ValueError(f"@@@@{n=}")
            self.matvec = splinalg.LinearOperator(shape=(n, n), matvec=kwargs.pop('matvec'))
        if "matvecprec" in kwargs:
            n = kwargs.get('n')
            self.M = splinalg.LinearOperator(shape=(n, n), matvec=kwargs.pop('matvecprec'))
        else:
            self.M = None
        # self.args = {"A": self.matvec, "M":self.M, "atol":self.atol}
        self.args['A'] = self.matvec
        self.args['M'] = self.M
        if self.method in scipysolvers:
            self.solver = eval('splinalg.'+self.method[6:])
            self.args['atol'] = self.atol
        elif self.method in pyamgsolvers:
            import pyamg
            self.solver = eval('pyamg.krylov.' + self.method[6:])
        elif self.method == 'idr':
            # import scipy.sparse.linalg.isolve
            import idrs
            self.solver = eval('idrs.idrs')
        else:
            raise ValueError("*** unknown {self.method=}")
        name = self.method
        if self.method=='scipy_gcrotmk':
            self.args['m'] = kwargs.pop('m', 5)
            self.args['truncate'] = kwargs.pop('truncate', 'smallest')
            self.solver = splinalg.gcrotmk
            name += '_' + str(self.args['m'])
#=================================================================#
class Pyamg(IterativeSolver):
    def __repr__(self):
        s = super().__repr__()
        return s + f"pyamg_{self.type}_{self.smoother}_{str(self.accel)}"
    def __init__(self, A, **kwargs):
        try:
            import pyamg
        except:
            raise ImportError(f"*** pyamg not found ***")
        assert A is not None
        self.method = 'pyamg'
        nsmooth = kwargs.pop('nsmooth', 1)
        # self.smoother = kwargs.pop('smoother', 'schwarz')
        self.smoother = kwargs.pop('smoother', 'gauss_seidel')
        symmetric = kwargs.pop('symmetric', False)
        self.type = kwargs.pop('pyamgtype', 'aggregation')
        self.accel = kwargs.pop('accel', None)
        if self.accel == 'none': self.accel=None
        # pyamgargs = {'B': pyamg.solver_configuration(A, verb=False)['B']}
        pyamgargs = {}
        smoother = (self.smoother, {'sweep': 'symmetric', 'iterations': nsmooth})
        if symmetric:
            smooth = ('energy', {'krylov': 'cg'})
        else:
            smooth = ('energy', {'krylov': 'fgmres'})
            pyamgargs['symmetry'] = 'nonsymmetric'
        pyamgargs['presmoother'] = smoother
        pyamgargs['postsmoother'] = smoother
        # pyamgargs['smooth'] = smooth
        pyamgargs['coarse_solver'] = 'splu'
        if self.type == 'aggregation':
            self.mlsolver = pyamg.smoothed_aggregation_solver(A, **pyamgargs)
        elif self.type == 'rootnode':
            self.mlsolver = pyamg.rootnode_solver(A, **pyamgargs)
        else:
            raise ValueError(f"unknown {self.type=}")
        self.solver = self.mlsolver.solve
        #        cycle : {'V','W','F','AMLI'}
        super().__init__(**kwargs)
        self.args['cycle'] = 'V'
        self.args['accel'] = self.accel
