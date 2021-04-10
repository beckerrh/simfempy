import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from simfempy import fems
from simfempy.applications.application import Application
from simfempy.tools.analyticalfunction import analyticalSolution

#=================================================================#
class StokesBase(Application):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dirichletmethod = kwargs.pop('dirichletmethod', 'none')
        self.femv = fems.cr1sys.CR1sys(self.ncomp, dirichletmethod=dirichletmethod)
        self.femp = fems.d0.D0()
    def setMesh(self, mesh):
        super().setMesh(mesh)
        assert self.ncomp==self.mesh.dimension
        self.femv.setMesh(self.mesh)
        self.femp.setMesh(self.mesh)
        self.mucell = self.compute_cell_vector_from_params('mu', self.problemdata.params)
        self.pmean = list(self.problemdata.bdrycond.type.values()) == len(self.problemdata.bdrycond.type)*['Dirichlet']
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
        return _solexactdirv
    def defineRhsAnalyticalSolution(self, solexact):
        v,p = solexact
        mu = self.problemdata.params.scal_glob['mu']
        def _fctrhsv(x, y, z):
            rhsv = np.zeros(shape=(self.ncomp, x.shape[0]))
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhsv[i] -= mu * v[i].dd(j, j, x, y, z)
                rhsv[i] += p.d(i, x, y, z)
            # print(f"{rhsv=}")
            return rhsv
        def _fctrhsp(x, y, z):
            rhsp = np.zeros(x.shape[0])
            for i in range(self.ncomp):
                rhsp += v[i].d(i, x, y, z)
            return rhsp
        return _fctrhsv, _fctrhsp
    def defineNeumannAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        mu = self.problemdata.params.scal_glob['mu']
        def _fctneumannv(x, y, z, nx, ny, nz):
            v, p = solexact
            rhsv = np.zeros(shape=(self.ncomp, x.shape[0]))
            normals = nx, ny, nz
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhsv[i] += mu  * v[i].d(j, x, y, z) * normals[j]
                rhsv[i] -= p(x, y, z) * normals[i]
            return rhsv
        def _fctneumannp(x, y, z, nx, ny, nz):
            v, p = solexact
            rhsp = np.zeros(shape=x.shape[0])
            normals = nx, ny, nz
            for i in range(self.ncomp):
                rhsp -= v[i](x, y, z) * normals[i]
            return rhsp
        return _fctneumannv
    def solve(self, iter, dirname):
        return self.static(iter, dirname)
    def postProcess(self, u):
        if self.pmean:
            v,p,lam =  u
            print(f"{lam=}")
        else: v,p =  u
        data = {'point':{}, 'cell':{}, 'global':{}}
        for icomp in range(self.ncomp):
            data['point'][f'V_{icomp:01d}'] = self.femv.fem.tonode(v[icomp::self.ncomp])
        data['cell']['P'] = p
        if self.problemdata.solexact:
            err, e = self.femv.computeErrorL2(self.problemdata.solexact[0], v)
            data['global']['error_V_L2'] = np.sum(err)
            err, e = self.femp.computeErrorL2(self.problemdata.solexact[1], p)
            data['global']['error_P_L2'] = err
        if self.problemdata.postproc:
            types = ["bdry_pmean", "bdry_nflux"]
            for name, type in self.problemdata.postproc.type.items():
                colors = self.problemdata.postproc.colors(name)
                if type == types[0]:
                    data['global'][name] = self.femp.computeBdryMean(p, colors)
                elif type == types[1]:
                    data['global'][name] = self.computeBdryNormalFlux(v, p, colors)
                else:
                    raise ValueError(f"unknown postprocess type '{type}' for key '{name}'\nknown types={types=}")
        return data
    def _to_single_matrix(self, Ain):
        ncells, nfaces = self.mesh.ncells, self.mesh.nfaces
        # print("Ain", Ain)
        if self.pmean:
            A, B, C = Ain
        else:
            A, B = Ain
        nullP = sparse.dia_matrix((np.zeros(ncells), 0), shape=(ncells, ncells))
        A1 = sparse.hstack([A, -B.T])
        A2 = sparse.hstack([B, nullP])
        Aall = sparse.vstack([A1, A2])
        if not self.pmean:
            return Aall.tocsr()
        ncomp = self.ncomp
        rows = np.zeros(ncomp*nfaces, dtype=int)
        cols = np.arange(0, ncomp*nfaces)
        nullV = sparse.coo_matrix((np.zeros(ncomp*nfaces), (rows, cols)), shape=(1, ncomp*nfaces)).tocsr()
        CL = sparse.hstack([nullV, C])
        Abig = sparse.hstack([Aall,CL.T])
        nullL = sparse.dia_matrix((np.zeros(1), 0), shape=(1, 1))
        Cbig = sparse.hstack([CL,nullL])
        Aall = sparse.vstack([Abig, Cbig])
        return Aall.tocsr()
    def linearSolver(self, Ain, bin, uin=None, solver='umf', verbose=0):
        ncells, nfaces, ncomp = self.mesh.ncells, self.mesh.nfaces, self.ncomp
        if solver == 'umf':
            Aall = self._to_single_matrix(Ain)
            if self.pmean:
                ball = np.hstack((bin[0],bin[1],bin[2]))
            else: ball = np.hstack((bin[0],bin[1]))
            uall =  splinalg.spsolve(Aall, ball, permc_spec='COLAMD')
            if self.pmean: return (uall[:nfaces*ncomp],uall[nfaces*ncomp:nfaces*ncomp+ncells],uall[-1]), 1
            else: return (uall[:nfaces*ncomp],uall[nfaces*ncomp:]), 1
        elif solver == 'gmres':
            from simfempy import tools
            nfaces, ncells, ncomp = self.mesh.nfaces, self.mesh.ncells, self.ncomp
            counter = tools.iterationcounter.IterationCounter(name=solver)
            if self.pmean:
                A, B, C = Ain
                nall = ncomp*nfaces + ncells + 1
                BP = sparse.diags(1/self.mesh.dV, offsets=(0), shape=(ncells, ncells))
                CP = splinalg.inv(C*BP*C.T)
                import pyamg
                config = pyamg.solver_configuration(A, verb=False)
                API = pyamg.rootnode_solver(A, B=config['B'], smooth='energy')
                def amult(x):
                    v, p, lam = x[:ncomp*nfaces], x[ncomp*nfaces:ncomp*nfaces+ncells], x[-1]*np.ones(1)
                    w = -B.T.dot(p)
                    w += A.dot(v)
                    q = B.dot(v)+C.T.dot(lam)
                    return np.hstack([w, q, C.dot(p)])
                Amult = splinalg.LinearOperator(shape=(nall, nall), matvec=amult)
                def pmult(x):
                    v, p, lam = x[:ncomp*nfaces], x[ncomp*nfaces:ncomp*nfaces+ncells], x[-1]*np.ones(1)
                    w = API.solve(v, maxiter=1, tol=1e-16)
                    q = BP.dot(p-B.dot(w))
                    mu = CP.dot(lam-C.dot(q)).ravel()
                    q += BP.dot(C.T.dot(mu).ravel())
                    h = B.T.dot(q)
                    w += API.solve(h, maxiter=1, tol=1e-16)
                    return np.hstack([w, q, mu])
                P = splinalg.LinearOperator(shape=(nall, nall), matvec=pmult)
                b = np.hstack([bin[0], bin[1], bin[2]])
                u, info = splinalg.lgmres(Amult, b, M=P, callback=counter, atol=1e-14, tol=1e-14, inner_m=10, outer_k=4)
                if info: raise ValueError("no convergence info={}".format(info))
                return (u[:ncomp*nfaces], u[ncomp*nfaces:ncomp*nfaces+ncells], u[-1]*np.ones(1)), counter.niter
            else:
                A, B = Ain
                nall = ncomp*nfaces + ncells
                BP = sparse.diags(1/self.mesh.dV, offsets=(0), shape=(ncells, ncells))
                import pyamg
                config = pyamg.solver_configuration(A, verb=False)
                API = pyamg.rootnode_solver(A, B=config['B'], smooth='energy')
                def amult(x):
                    v, p = x[:ncomp*nfaces], x[ncomp*nfaces:ncomp*nfaces+ncells]
                    w = -B.T.dot(p)
                    w += A.dot(v)
                    q = B.dot(v)
                    return np.hstack([w, q])
                Amult = splinalg.LinearOperator(shape=(nall, nall), matvec=amult)
                def pmult(x):
                    v, p = x[:ncomp*nfaces], x[ncomp*nfaces:ncomp*nfaces+ncells]
                    w = API.solve(v, maxiter=1, tol=1e-16)
                    q = BP.dot(p-B.dot(w))
                    h = B.T.dot(q)
                    w += API.solve(h, maxiter=1, tol=1e-16)
                    return np.hstack([w, q])
                P = splinalg.LinearOperator(shape=(nall, nall), matvec=pmult)
                b = np.hstack([bin[0], bin[1]])
                u, info = splinalg.lgmres(Amult, b, M=P, callback=counter, atol=1e-14, tol=1e-14, inner_m=10, outer_k=4)
                if info: raise ValueError("no convergence info={}".format(info))
                return (u[:ncomp*nfaces], u[ncomp*nfaces:ncomp*nfaces+ncells]), counter.niter
        else:
            raise ValueError(f"unknown solve '{solver=}'")

#=================================================================#
if __name__ == '__main__':
    raise NotImplementedError("Pas encore de test")
