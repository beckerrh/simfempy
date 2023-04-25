import numpy as np
import scipy.sparse as sparse
from . import linalg

#=================================================================#
class SaddlePointPreconditioner():
    """
    """
    def __repr__(self):
        s =  f"{self.method=}\n{self.type=}"
        if hasattr(self,'SV'): s += f"\n{self.SV=}"
        if hasattr(self,'SP'): s += f"\n{self.SP=}"
        return s
    def _get_schur_of_diag(self, ret_diag=False):
        if self.AS.singleA:
            AD = linalg.diagmatrix2systemdiagmatrix(1/self.AS.A.diagonal(), self.AS.ncomp)
        else:
            AD = sparse.diags(1 / self.AS.A.diagonal(), offsets=(0), shape=self.AS.A.shape)
        if ret_diag: return self.AS.B @ AD @ self.AS.B.T, AD
        return self.AS.B @ AD @ self.AS.B.T

    def __init__(self, AS, **kwargs):
        self.AS = AS
        method = kwargs.pop('method','full')
        self.method = method
        solver_p = kwargs.pop('solver_p', None)
        solver_v = kwargs.pop('solver_v', None)
        constr = hasattr(AS, 'M')
        self.nv = self.AS.na
        self.nvp = self.AS.na + AS.nb
        self.nall = self.nvp
        if constr: self.nall += AS.nm
        # print(f"{AS=} {self.nv=} {self.nvp=} {self.nall=} {solver_v=} {solver_p=}")
        if method == 'diag':
            self.matvecprec = self.pmatvec3_diag if constr else self.pmatvec2_diag
        elif method == 'triup':
            self.matvecprec = self.pmatvec3_triup if constr else self.pmatvec2_triup
        elif method == 'tridown':
            self.matvecprec = self.pmatvec3_tridown if constr else self.pmatvec2_tridown
        elif method == 'full':
            self.matvecprec = self.pmatvec3_full if constr else self.pmatvec2_full
        elif method[:3] == 'hss':
            ms = method.split('_')
            if len(ms) != 2: raise ValueError(f"*** needs 'hass_alpha'")
            self.alpha = float(ms[1])
            # solver_p['type'] = f"diag_{self.alpha**2}"
            # solver_p['method'] = f"pyamg"
            self.matvecprec = self.pmatvec3_hss if constr else self.pmatvec2_hss
        else:
            raise ValueError(f"*** unknwon {method=}\npossible values: 'diag', 'triup', 'tridown', 'full', hss_alpha")
        if not isinstance(AS, linalg.SaddlePointSystem):
            raise ValueError(f"*** resuired arguments: AS (SaddlePointSystem)")
        if solver_v is None:
            solver_v = {'method': 'pyamg', 'maxiter': 10, 'disp':0}
        assert solver_p == None
        if solver_p is None:
            solver_p = {'method': 'pyamg', 'maxiter': 10, 'disp':0}
        if isinstance(solver_v,dict):
            solver_v['matrix'] = AS.A
        else:
            for s in solver_v:
                s['matrix'] = AS.A
        solver_v['counter'] = '\tV '
        solver_v['matrix'] = self.AS.A
        solver_p['counter'] = '\tP '
        solver_p['matrix'] = self._get_schur_of_diag()
        # solver_p['matrix'] = self.AS.B @ self.AS.B.T
        # print(f"{solver_p['matrix'].shape=} {AD.shape=}")
        self.SP = linalg.getLinearSolver(args=solver_p)
        self.SV = linalg.getLinearSolver(args=solver_v)
        # self.type = solver_p['type']
        # # print(f"{self.type=}")
        # if self.type == 'scale':
        #     self.SP = linalg.DiagonalScaleSolver(coeff = 1/solver_p['coeff'])
        #     return
        # solver_p['counter'] = '\tP '
        # if self.type[:4] =='diag':
        #     ts = self.type.split('_')
        #     if len(ts)>1:
        #         alpha = float(ts[1])
        #         solver_p['matrix'] = AS.B@ AS.B.T + alpha*sparse.identity(AS.B.shape[0])
        #     else:
        #         # AD = sparse.diags(1 / AS.A.diagonal(), offsets=(0), shape=AS.A.shape)
        #         # solver_p['matrix'] = AS.B @ AD @ AS.B.T
        #         solver_p['matrix'] = self._get_schur_of_diag()
        # elif self.type[:5] == 'schur':
        #     ts = self.type.split('|')
        #     if len(ts)>1:
        #         prec = ts[1]
        #         if prec == 'diag':
        #             # AD = sparse.diags(1 / AS.A.diagonal(), offsets=(0), shape=AS.A.shape)
        #             args = {'method':'pyamg', 'maxiter':1}
        #             # args['matrix'] = AS.B @ AD @ AS.B.T
        #             args['matrix'] = self._get_schur_of_diag()
        #             solver_p['prec'] = linalg.getSolver(args=args)
        #         elif prec == 'scale':
        #             AD = sparse.diags(1 / AS.A.diagonal(), offsets=(0), shape=AS.A.shape)
        #             solver_p['prec'] = linalg.DiagonalScaleSolver(coeff = AD.diagonal())
        #         else:
        #             raise ValueError(f"unknwon {self.type=} {solver_p=}")
        #     # else:
        #     #     raise ValueError(f"unknwon {self.type=} {solver_p=}")
        #     solver_p['matvec'] = self.schurmatvec
        #     solver_p['n'] = AS.B.shape[0]
        #     # print(f"## {solver_p=}")
        # else:
        #     raise ValueError(f"*** {self.type=} not in [schur|diag/scale, diag, diag_alpha] {self.method=} {solver_p=}")
        # self.SP = linalg.getLinearSolver(args=solver_p)
    def _solve_v(self,b):
        if self.AS.singleA:
            w = np.empty_like(b)
            for i in range(self.AS.ncomp): w[i::self.AS.ncomp] = self.SV.solve(b=b[i::self.AS.ncomp])
            return w
        return self.SV.solve(b=b)
    def schurmatvec(self, x):
        v = self.AS.B.T.dot(x)
        v2 = self._solve_v(v)
        return self.AS.B.dot(v2)
    def pmatvec2_diag(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self._solve_v(v)
        q = self.SP.solve(b=p)
        return np.hstack([w, q])
    def pmatvec3_diag(self, x):
        v, p, lam = x[:self.nv], x[self.nv:self.nvp], x[self.nvp:]
        w = self._solve_v(v)
        q = self.SP.solve(b=p)
        mu = self.MP.solve(lam)
        return np.hstack([w, q, mu])
    def pmatvec2_triup(self, x):
        v, p = x[:self.nv], x[self.nv:]
        q = self.SP.solve(b=p)
        w = self._solve_v(v+self.AS.B.T.dot(q))
        return np.hstack([w, q])
    def pmatvec2_tridown(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self._solve_v(v)
        q = self.SP.solve(b=p-self.AS.B.dot(w))
        return np.hstack([w, q])
    def pmatvec2_full(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self._solve_v(v)
        q = self.SP.solve(b=p-self.AS.B.dot(w))
        h = self.AS.B.T.dot(q)
        w += self._solve_v(h)
        return np.hstack([w, q])
    def pmatvec2_hss(self, x):
        v, p = x[:self.nv], x[self.nv:]
        q = self.SP.solve(b=p-1/self.alpha*self.AS.B.dot(v))
        w = self._solve_v(1/self.alpha*v + self.AS.B.T.dot(q))
        return np.hstack([w, q])
#=================================================================#
class BraessSarazin(SaddlePointPreconditioner):
    """
    Instead of
    --------
    A -B.T
    B  0
    --------
    solve
    --------
    alpha*diag(A)  -B.T
    B               0
    --------
    S = B*diag(A)^{-1}*B.T
    S p = alpha*g - B C^{-1}f
    C v = 1/alpha*f + 1/alpha*B.T*p
    """
    def __repr__(self):
        return self.__class__.__name__ + f"{self.alpha=}"
        return s
    def __init__(self, AS, **kwargs):
        self.AS = AS
        self.alpha = kwargs.pop('alpha',10)
        solver_p = kwargs.pop('solver_p', {})
        solver_v = kwargs.pop('solver_v', {})
        constr = hasattr(AS, 'M')
        assert not constr
        self.nv = self.AS.na
        self.nvp = self.AS.na + AS.nb
        self.nall = self.nvp
        if constr: self.nall += AS.nm
        # print(f"{AS=} {self.nv=} {self.nvp=} {self.nall=} {solver_v=} {solver_p=}")
        if not isinstance(AS, linalg.SaddlePointSystem) or not isinstance(solver_p, (list,dict)) or not isinstance(solver_v,(list,dict)):
            raise ValueError(f"*** resuired arguments: AS (SaddlePointSystem), solver_p, solver_v (dicts of arguments ")
        solver_p['counter'] = '\tP '
        solver_p = {'method': 'pyamg', 'maxiter': 3}
        solver_p['matrix'], AD = self._get_schur_of_diag(ret_diag=True)
        # print(f"{solver_p['matrix'].shape=} {AD.shape=}")
        self.SP = linalg.getLinearSolver(args=solver_p)
        self.SV = linalg.DiagonalScaleSolver(AD.diagonal())

    def matvecprec(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self.SV.solve(b=v)/self.alpha
        q = self.SP.solve(b=self.alpha*p-self.AS.B.dot(w))
        h = self.AS.B.T.dot(q)
        w += self.SV.solve(h)/self.alpha
        return np.hstack([w, q])
#=================================================================#
class Chorin(SaddlePointPreconditioner):
    """
    Instead of
    --------
    A -B.T
    B  0
    --------
    solve
    --------
    A  -B.T
    0   B@B.T
    --------
    """
    def __repr__(self):
        return self.__class__.__name__
        return s
    def __init__(self, AS, **kwargs):
        self.AS = AS
        solver_p = kwargs.pop('solver_p', None)
        solver_v = kwargs.pop('solver_v', None)
        constr = hasattr(AS, 'M')
        assert not constr
        self.nv = self.AS.na
        self.nvp = self.AS.na + AS.nb
        self.nall = self.nvp
        if constr: self.nall += AS.nm
        # print(f"{AS=} {self.nv=} {self.nvp=} {self.nall=} {solver_v=} {solver_p=}")
        if not isinstance(AS, linalg.SaddlePointSystem):
            raise ValueError(f"*** resuired arguments: AS (SaddlePointSystem), solver_p, solver_v (dicts of arguments ")
        if solver_v is None:
            solver_v = {'method': 'pyamg', 'maxiter': 1, 'disp':0}
        if solver_p is None:
            solver_p = {'method': 'pyamg', 'maxiter': 1, 'disp':0}
        solver_v['matrix'] = self.AS.A
        solver_v['counter'] = '\tV '
        solver_p['counter'] = '\tP '
        solver_p['matrix'] = self._get_schur_of_diag()
        # solver_p['matrix'] = self.AS.B @ self.AS.B.T
        # print(f"{solver_p['matrix'].shape=} {AD.shape=}")
        self.SP = linalg.getLinearSolver(args=solver_p)
        self.SV = linalg.getLinearSolver(args=solver_v)

    def matvecprec2(self, x):
        v, p = x[:self.nv], x[self.nv:]
        q = self.SP.solve(b=p)
        w = self._solve_v(b=v+self.AS.B.T.dot(q))
        return np.hstack([w, q])
    def matvecprec(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self._solve_v(v)
        q = self.SP.solve(b=p-self.AS.B@w)
        w += self.AS.B.T@q
        return np.hstack([w, q])
