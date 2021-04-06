import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from simfempy import fems
from simfempy.applications.application import Application
from simfempy.tools.analyticalfunction import analyticalSolution

#=================================================================#
class StokesN(Application):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.femv = fems.cr1sys.CR1sys(self.ncomp)
        self.femp = fems.d0.D0()
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.femv.setMesh(self.mesh)
        self.femp.setMesh(self.mesh)
        self.mucell = self.compute_cell_vector_from_params('mu', self.problemdata.params)
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
        self.bdrydata = self.femv.prepareBoundary(colorsdirichlet, colorsflux)
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
        return _solexactdirv, _solexactdirp
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
        return _fctneumannv, _fctneumannp
    def solve(self, iter, dirname): return self.static(iter, dirname)
    def computeRhs(self, b=None, u=None, coeff=1, coeffmass=None):
        bv = np.zeros(self.femv.nunknowns() * self.ncomp)
        bp = np.zeros(self.femp.nunknowns())
        rhsv, rhsp = self.problemdata.params.fct_glob['rhs']
        if rhsv: self.femv.computeRhsCells(bv, rhsv)
        if rhsp: self.femp.computeRhsCells(bp, rhsp)
        # print(f"{bv=}")
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsneu = self.problemdata.bdrycond.colorsOfType("Neumann")
        bdryfctv = {k:v[0] for k,v in self.problemdata.bdrycond.fct.items()}
        bdryfctp = {k:v[1] for k,v in self.problemdata.bdrycond.fct.items()}
        self.femv.computeRhsBoundary(bv, colorsneu, bdryfctv)
        # self.femp.computeRhsBoundary(bp, colorsdir, bdryfctp)
        self.computeRhsNitsche((bv,bp), colorsdir, self.problemdata.bdrycond.fct)
        b, u, self.bdrydata = self.vectorBoundary((bv, bp), u, bdryfctv)
        if not self.pmean: return b,u
        if hasattr(self.problemdata,'solexact'):
            p = self.problemdata.solexact[1]
            pmean = self.femp.computeMean(p)
        else: pmean=0
        print(f"{pmean=}")
        return (bv,bp,pmean), (u[0], u[1], 0)
    def computeMatrix(self):
        A = self.femv.computeMatrixLaplace(self.mucell)
        B = self.femv.computeMatrixDivergence()
        A, B, self.bdrydata = self.matrixBoundary(A, B)
        if not self.pmean:
            return A, B
        ncells = self.mesh.ncells
        rows = np.zeros(ncells, dtype=int)
        cols = np.arange(0, ncells)
        C = sparse.coo_matrix((self.mesh.dV, (rows, cols)), shape=(1, ncells)).tocsr()
        return A,B,C
    def postProcess(self, u):
        if self.pmean:
            v,p,lam =  u
            print(f"{lam=}")
        else: v,p =  u
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
    def computeRhsNitsche(self, b, colorsdir, bdryfct):
        bv, bp = b
        xf, yf, zf = self.mesh.pointsf.T
        for color in colorsdir:
            faces = self.mesh.bdrylabels[color]
            cells = self.mesh.cellsOfFaces[faces,0]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS,axis=1)
            # normalsS = normalsS/dS[:,np.newaxis]
            if color in bdryfct.keys():
                bfctv, bfctp = bdryfct[color]
                dirichv = np.hstack([bfctv(xf[faces], yf[faces], zf[faces])])
                bp[cells] -= np.einsum('kn,nk->n', dirichv, normalsS[:,:self.ncomp])

    def vectorBoundary(self, b, u, bdryfctv):
        bv, bp = b
        if u is None:
            uv = np.zeros_like(bv)
            up = np.zeros_like(bp)
        else:
            uv, up = u
            assert uv.shape == bv.shape
            assert up.shape == bp.shape
        xf, yf, zf = self.mesh.pointsf.T
        facesdirall, facesinner, colorsdir, facesdirflux = self.bdrydata.facesdirall, self.bdrydata.facesinner, self.bdrydata.colorsdir, self.bdrydata.facesdirflux
        nfaces, ncells, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp
        self.bdrydata.bsaved = []
        for icomp in range(ncomp):
            self.bdrydata.bsaved.append({})
            for key, faces in facesdirflux.items():
                self.bdrydata.bsaved[icomp][key] = bv[icomp + ncomp * faces]
        if self.femv.fem.dirichletmethod == 'trad':
            for color in colorsdir:
                faces = self.mesh.bdrylabels[color]
                if color in bdryfctv.keys():
                    dirichlets = bdryfctv[color](xf[faces], yf[faces], zf[faces])
                    for icomp in range(ncomp):
                        bv[icomp + ncomp * faces] = dirichlets[icomp]
                        uv[icomp + ncomp * faces] = bv[icomp + ncomp * faces]
                else:
                    for icomp in range(ncomp):
                        bv[icomp + ncomp * faces] = 0
                        uv[icomp + ncomp * faces] = bv[icomp + ncomp * faces]
            indin = np.repeat(ncomp * facesinner, ncomp)
            for icomp in range(ncomp): indin[icomp::ncomp] += icomp
            inddir = np.repeat(ncomp * facesdirall, ncomp)
            for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
            bv[indin] -= self.bdrydata.A_inner_dir * bv[inddir]
            # bp -= self.bdrydata.B_inner_dir * bv[inddir]
        else:
            raise NotImplementedError()
        return (bv,bp), (uv,up), self.bdrydata
    def matrixBoundary(self, A, B):
        A, self.bdrydata = self.femv.matrixBoundary(A, self.bdrydata)
        facesdirall, facesinner, colorsdir, facesdirflux = self.bdrydata.facesdirall, self.bdrydata.facesinner, self.bdrydata.colorsdir, self.bdrydata.facesdirflux
        nfaces, ncells, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp
        self.bdrydata.Bsaved = {}
        for key, faces in facesdirflux.items():
            nb = faces.shape[0]
            helpB = sparse.dok_matrix((ncomp*nfaces, ncomp*nb))
            for icomp in range(ncomp):
                # for i in range(nb): helpB[icomp*nfaces + faces[i], icomp*nb + i] = 1
                for i in range(nb): helpB[icomp + ncomp*faces[i], icomp + ncomp*i] = 1
            self.bdrydata.Bsaved[key] = B.dot(helpB)
        # self.bdrydata.A_inner_dir = A[facesinner, :][:, facesdirall]
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        self.bdrydata.B_inner_dir = B[:,:][:,inddir]
        help = np.ones((ncomp * nfaces))
        help[inddir] = 0
        help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
        B = B.dot(help)
        return A,B, self.bdrydata

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
            else: return (uall[:nfaces*ncomp],uall[nfaces*ncomp:nfaces*ncomp]), 1
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
