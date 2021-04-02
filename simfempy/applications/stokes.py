import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from simfempy import fems
from simfempy.applications.application import Application
from simfempy.tools.analyticalfunction import analyticalSolution

#=================================================================#
class Stokes(Application):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.femv = fems.cr1sys.CR1sys(self.ncomp)
        self.femp = fems.d0.D0()
        self.mu = kwargs.pop('mu', 1)
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.femv.setMesh(self.mesh)
        self.femp.setMesh(self.mesh)
        self.mucell = np.full(self.mesh.ncells, self.mu)
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
        self.bdrydata = self.femv.prepareBoundary(colorsdirichlet, colorsflux)
        self.pmean = self.problemdata.bdrycond.type.values() == len(self.problemdata.bdrycond.type)*"Dirichlet"
        assert not self.pmean

    def defineAnalyticalSolution(self, exactsolution, random=True):
        dim = self.mesh.dimension
        # print(f"defineAnalyticalSolution: {dim=} {self.ncomp=}")
        if exactsolution=="Linear":
            exactsolution = ["Linear", "Constant"]
        v = analyticalSolution(exactsolution[0], dim, dim, random)
        p = analyticalSolution(exactsolution[1], dim, 1, random)
        return v,p
    def dirichletfct(self):
        solexact = self.problemdata.solexact
        v,p = solexact
        def _solexactdirv(x, y, z):
            return [v[icomp](x, y, z) for icomp in range(self.ncomp)]
        def _solexactdirp(x, y, z, nx, ny, nz):
            return p(x, y, z)
        return _solexactdirv, _solexactdirp
    def defineRhsAnalyticalSolution(self, solexact):
        v,p = solexact
        def _fctrhsv(x, y, z):
            rhsv = np.zeros(shape=(self.ncomp, x.shape[0]))
            mu = self.mu
            for i in range(self.ncomp):
                rhsv[i] -= mu * v[i].dd(i, i, x, y, z)
                rhsv[i] += p.d(i, x, y, z)
            return rhsv
        def _fctrhsp(x, y, z):
            rhsp = np.zeros(x.shape[0])
            for i in range(self.ncomp):
                rhsp = v[i].d(i, x, y, z)
            return rhsp
        return _fctrhsv, _fctrhsp
    def defineNeumannAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        def _fctneumannv(x, y, z, nx, ny, nz):
            v, p = solexact
            rhsv = np.zeros(shape=(self.ncomp, x.shape[0]))
            normals = nx, ny, nz
            mu = self.mu
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhsv[i] -= mu  * v[i].d(j, x, y, z) * normals[j]
                rhsv[i] += p(x, y, z) * normals[i]
            return rhsv
        def _fctneumannp(x, y, z, nx, ny, nz):
            v, p = solexact
            rhsp = np.zeros(shape=x.shape[0])
            normals = nx, ny, nz
            for i in range(self.ncomp):
                rhsp -= v[i](x, y, z) * normals[i]
            return rhsp
        return _fctneumannv, _fctneumannp
    def solve(self, iter, dirname): return self.static(iter, dirname)
    def computeRhs(self, b=None, u=None, coeff=1, coeffmass=None):
        bv = np.zeros(self.femv.nunknowns() * self.ncomp)
        bp = np.zeros(self.femp.nunknowns())
        rhsv, rhsp = self.problemdata.params.fct_glob['rhs']
        if rhsv: self.femv.computeRhsCells(bv, rhsv)
        if rhsp: self.femp.computeRhsCells(bp, rhsp)
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsneu = self.problemdata.bdrycond.colorsOfType("Neumann")
        # for k, v in self.problemdata.bdrycond.fct.items():
        #     print(f"{k} {v}")
        bdryfctv = {k:v[0] for k,v in self.problemdata.bdrycond.fct.items()}
        bdryfctp = {k:v[1] for k,v in self.problemdata.bdrycond.fct.items()}
        self.femv.computeRhsBoundary(bv, colorsneu, bdryfctv)
        self.femp.computeRhsBoundary(bp, colorsdir, bdryfctp)
        if u is not None: (uv,up) = u
        else: (uv,up) = (None,None)
        bv, uv, self.bdrydata = self.femv.vectorBoundary(bv, uv, bdryfctv, self.bdrydata)
        return (bv,bp), (uv,up)
    def computeMatrix(self):
        A = self.femv.computeMatrixLaplace(self.mucell)
        A, self.bdrydata = self.femv.matrixBoundary(A, self.bdrydata)
        # colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        B = self.femv.computeMatrixDivergence(self.bdrydata.facesdirall)
        # print(f"{A.todense()=}")
        # print(f"{B.todense()=}")
        return A, B
    def postProcess(self, u):
        v,p =  u
        data = {'point':{}, 'cell':{}, 'global':{}}
        for icomp in range(self.ncomp):
            data['point'][f'V_{icomp:02d}'] = self.femv.fem.tonode(v[icomp::self.ncomp])
        data['cell']['P'] = p
        if self.problemdata.solexact:
            err, e = self.femv.computeErrorL2(self.problemdata.solexact[0], v)
            data['global']['error_V_L2'] = np.sum(err)
            err, e = self.femp.computeErrorL2(self.problemdata.solexact[1], p)
            data['global']['error_P_L2'] = err
        return data

    def _to_single_matrix(self, Ain):
        import scipy.sparse
        ncells, nfaces = self.mesh.ncells, self.mesh.nfaces
        assert not self.pmean
        # print("Ain", Ain)
        A, B = Ain
        nullP = scipy.sparse.dia_matrix((np.zeros(ncells), 0), shape=(ncells, ncells))
        A1 = scipy.sparse.hstack([A, -B.T])
        A2 = scipy.sparse.hstack([B, nullP])
        Aall = scipy.sparse.vstack([A1, A2])
        return Aall.tocsr()

    def linearSolver(self, Ain, bin, uin=None, solver='umf', verbose=0):
        ncells, nfaces, ncomp = self.mesh.ncells, self.mesh.nfaces, self.ncomp
        if solver == 'umf':
            Aall = self._to_single_matrix(Ain)
            # print(f"{Aall.diagonal()=}")
            ball = np.hstack((bin[0],bin[1]))
            print(f"{ball.shape=}")
            uall =  splinalg.spsolve(Aall, ball, permc_spec='COLAMD')
            print(f"{uall.shape=}")
            return (uall[:nfaces*ncomp],uall[nfaces*ncomp:]), 1
        elif solver == 'gmres':
            nfaces, ncells, ncomp, pstart = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.pstart
            counter = simfempy.tools.iterationcounter.IterationCounter(name=solver)
            if self.pmean:
                A, B, C = Ain
                nall = ncomp*nfaces + ncells + 1
                BP = scipy.sparse.diags(1/self.mesh.dV, offsets=(0), shape=(ncells, ncells))
                CP = splinalg.inv(C*BP*C.T)
                import pyamg
                config = pyamg.solver_configuration(A, verb=False)
                API = pyamg.rootnode_solver(A, B=config['B'], smooth='energy')
                def amult(x):
                    v, p, lam = x[:pstart], x[pstart:pstart+ncells], x[pstart+ncells:]
                    w = -B.T.dot(p)
                    for i in range(ncomp):
                        w[i*nfaces: (i+1)*nfaces] += A.dot(v[i*nfaces: (i+1)*nfaces])
                    q = B.dot(v)+C.T.dot(lam).ravel()
                    return np.hstack([w, q, C.dot(p)])
                Amult = splinalg.LinearOperator(shape=(nall, nall), matvec=amult)
                def pmult(x):
                    v, p, lam = x[:pstart], x[pstart:pstart+ncells], x[pstart+ncells:]
                    w = np.zeros_like(v)
                    for i in range(ncomp):
                        w[i*nfaces: (i+1)*nfaces] = API.solve(v[i*nfaces: (i+1)*nfaces], maxiter=1, tol=1e-16)
                    q = BP.dot(p-B.dot(w))
                    mu = CP.dot(lam-C.dot(q)).ravel()
                    q += BP.dot(C.T.dot(mu).ravel())
                    h = B.T.dot(q)
                    for i in range(ncomp):
                        w[i*nfaces: (i+1)*nfaces] += API.solve(h[i*nfaces: (i+1)*nfaces], maxiter=1, tol=1e-16)
                    return np.hstack([w, q, mu])
                P = splinalg.LinearOperator(shape=(nall, nall), matvec=pmult)
                u, info = splinalg.lgmres(Amult, bin, M=P, callback=counter, atol=1e-14, tol=1e-14, inner_m=10, outer_k=4)
                if info: raise ValueError("no convergence info={}".format(info))
                return u, counter.niter
            else:
                A, B = Ain
                nall = ncomp*nfaces + ncells
                BP = scipy.sparse.diags(1/self.mesh.dV, offsets=(0), shape=(ncells, ncells))
                import pyamg
                config = pyamg.solver_configuration(A, verb=False)
                API = pyamg.rootnode_solver(A, B=config['B'], smooth='energy')
                def amult(x):
                    v, p= x[:pstart], x[pstart:pstart+ncells]
                    w = -B.T.dot(p)
                    for i in range(ncomp):
                        w[i*nfaces: (i+1)*nfaces] += A.dot(v[i*nfaces: (i+1)*nfaces])
                    q = B.dot(v)
                    return np.hstack([w, q])
                Amult = splinalg.LinearOperator(shape=(nall, nall), matvec=amult)
                def pmult(x):
                    v, p = x[:pstart], x[pstart:pstart+ncells]
                    w = np.zeros_like(v)
                    for i in range(ncomp):
                        w[i*nfaces: (i+1)*nfaces] = API.solve(v[i*nfaces: (i+1)*nfaces], maxiter=1, tol=1e-16)
                    q = BP.dot(p-B.dot(w))
                    h = B.T.dot(q)
                    for i in range(ncomp):
                        w[i*nfaces: (i+1)*nfaces] += API.solve(h[i*nfaces: (i+1)*nfaces], maxiter=1, tol=1e-16)
                    return np.hstack([w, q])
                P = splinalg.LinearOperator(shape=(nall, nall), matvec=pmult)
                u, info = splinalg.lgmres(Amult, bin, M=P, callback=counter, atol=1e-14, tol=1e-14, inner_m=10, outer_k=4)
                if info: raise ValueError("no convergence info={}".format(info))
                return u, counter.niter
        else:
            raise ValueError(f"unknown solve '{solver=}'")

#=================================================================#
if __name__ == '__main__':
    raise NotImplementedError("Pas encore de test")
