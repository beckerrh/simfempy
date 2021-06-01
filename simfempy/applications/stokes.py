import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from simfempy import fems, tools, solvers
from simfempy.applications.application import Application
from simfempy.tools.analyticalfunction import analyticalSolution
from simfempy.tools import npext
from functools import partial

#=================================================================#
class Stokes(Application):
    """
    """
    def __format__(self, spec):
        if spec=='-':
            repr = f"{self.femv=} {self.femp=}"
            repr += f"\tlinearsolver={self.linearsolver}"
            return repr
        return self.__repr__()
    def __init__(self, **kwargs):
        self.dirichlet_nitsche = 4
        self.dirichletmethod = kwargs.pop('dirichletmethod', 'nitsche')
        self.problemdata = kwargs.pop('problemdata')
        self.ncomp = self.problemdata.ncomp
        self.femv = fems.cr1sys.CR1sys(self.ncomp)
        self.femp = fems.d0.D0()
        super().__init__(**kwargs)
    def _zeros(self):
        nv = self.mesh.dimension*self.mesh.nfaces
        n = nv+self.mesh.ncells
        if self.pmean: n += 1
        return np.zeros(n)
    def _split(self, x):
        nv = self.mesh.dimension*self.mesh.nfaces
        ind = [nv]
        if self.pmean: ind.append(nv+self.mesh.ncells)
        # print(f"{ind=} {np.split(x, ind)=}")
        return np.split(x, ind)
    def setMesh(self, mesh):
        super().setMesh(mesh)
        assert self.ncomp==self.mesh.dimension
        self.femv.setMesh(self.mesh)
        self.femp.setMesh(self.mesh)
        self.mucell = self.compute_cell_vector_from_params('mu', self.problemdata.params)
        # self.pmean = list(self.problemdata.bdrycond.type.values()) == len(self.problemdata.bdrycond.type)*['Dirichlet']
        self.pmean = not ('Neumann' in self.problemdata.bdrycond.type.values())
        if self.dirichletmethod=='strong':
            assert 'Navier' not in self.problemdata.bdrycond.type.values()
            colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
            colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
            self.bdrydata = self.femv.prepareBoundary(colorsdirichlet, colorsflux)
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
        # def _solexactdirv(x, y, z):
        #     return [v[icomp](x, y, z) for icomp in range(self.ncomp)]
        def _solexactdirp(x, y, z, nx, ny, nz):
            return p(x, y, z)
        from functools import partial
        def _solexactdirv(x, y, z, icomp):
            # print(f"{icomp=}")
            return v[icomp](x, y, z)
        return [partial(_solexactdirv, icomp=icomp) for icomp in range(self.ncomp)]
        # return _solexactdirv
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
    def postProcess(self, u):
        if self.pmean: v, p, lam = self._split(u)
        else: v, p = self._split(u)
        # if self.pmean:
        #     v,p,lam =  u
        #     print(f"{lam=}")
        # else: v,p =  u
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
                    if self.dirichletmethod=='strong':
                        data['global'][name] = self.computeBdryNormalFluxStrong(v, p, colors)
                    else:
                        data['global'][name] = self.computeBdryNormalFluxNitsche(v, p, colors)
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
        nullV = sparse.coo_matrix((1, ncomp*nfaces)).tocsr()
        # rows = np.zeros(ncomp*nfaces, dtype=int)
        # cols = np.arange(0, ncomp*nfaces)
        # nullV = sparse.coo_matrix((np.zeros(ncomp*nfaces), (rows, cols)), shape=(1, ncomp*nfaces)).tocsr()
        CL = sparse.hstack([nullV, C])
        Abig = sparse.hstack([Aall,CL.T])
        nullL = sparse.dia_matrix((np.zeros(1), 0), shape=(1, 1))
        Cbig = sparse.hstack([CL,nullL])
        Aall = sparse.vstack([Abig, Cbig])
        return Aall.tocsr()
    def matrixVector(self, Ain, x):
        ncells, nfaces, ncomp = self.mesh.ncells, self.mesh.nfaces, self.ncomp
        if self.pmean:
            A, B, C = Ain
            v, p, lam = x[:ncomp*nfaces], x[ncomp*nfaces:ncomp*nfaces+ncells], x[-1]*np.ones(1)
            w = A.dot(v) - B.T.dot(p)
            q = B.dot(v)+C.T.dot(lam)
            return np.hstack([w, q, C.dot(p)])
        else:
            A, B = Ain
            v, p = x[:ncomp*nfaces], x[ncomp*nfaces:]
            w = A.dot(v) - B.T.dot(p)
            q = B.dot(v)
            return np.hstack([w, q])
    def getPrecMult(self, Ain, AP, SP):
        A, B = Ain[0], Ain[1]
        ncells, nfaces, ncomp = self.mesh.ncells, self.mesh.nfaces, self.ncomp
        # mu = self.problemdata.params.scal_glob['mu']
        # BP = sparse.diags(1/self.mesh.dV*mu, offsets=(0), shape=(ncells, ncells))
        # Ainv = sparse.linalg.spilu(A)
        # Ainv = sparse.diags(1/A.diagonal(), offsets=(0), shape=(nfaces*ncomp, nfaces*ncomp))
        # S = (B*Ainv*B.T).diagonal()
        # BP = splinalg.inv(B*Ainv*B.T)
        # BP = sparse.diags(1/S, offsets=(0), shape=(ncells, ncells))
        if self.pmean: 
            C = Ain[2]
            BPCT = SP.solve(C.T.toarray())
            # print(f"{C.dot(BPCT)=}")
            # CP = splinalg.inv(C.dot(BPCT))
            CP = sparse.coo_matrix(1/C.dot(BPCT))
        if self.pmean: 
            def pmult(x):
                v, p, lam = x[:ncomp*nfaces], x[ncomp*nfaces:ncomp*nfaces+ncells], x[-1]*np.ones(1)
                # return np.hstack([API.solve(v, maxiter=1, tol=1e-16), BP.dot(p), CP.dot(lam)])
                w = AP.solve(v)
                w = AP.solve(v)
                q = SP.solve(p-B.dot(w))
                mu = CP.dot(lam-C.dot(q)).ravel()
                q -= BPCT.dot(mu).ravel()
                h = B.T.dot(q)
                w += AP.solve(h)
                return np.hstack([w, q, mu])
        else:
            def pmult(x):
                v, p = x[:ncomp*nfaces], x[ncomp*nfaces:ncomp*nfaces+ncells]
                w = AP.solve(v)
                q = SP.solve(p-B.dot(w))
                h = B.T.dot(q)
                w += AP.solve(h)
                return np.hstack([w, q])
        return pmult
    def getVelocitySolver(self, A):
        return solvers.cfd.VelcoitySolver(A)
    def getPressureSolver(self, A, B, AP):
        mu = self.problemdata.params.scal_glob['mu']
        return solvers.cfd.PressureSolverDiagonal(self.mesh, mu)    
    def linearSolver(self, Ain, bin, uin=None, solver='umf', verbose=0, atol=1e-14, rtol=1e-10):
        ncells, nfaces, ncomp = self.mesh.ncells, self.mesh.nfaces, self.ncomp
        if solver == 'umf':
            Aall = self._to_single_matrix(Ain)
            uall =  splinalg.spsolve(Aall, bin, permc_spec='COLAMD')
            return uall, 1
        elif solver[:4] == 'iter':
            ssolver = solver.split('_')
            method=ssolver[1] if len(ssolver)>1 else 'lgmres'
            disp=int(ssolver[2]) if len(ssolver)>2 else 0
            nall = ncomp*nfaces + ncells
            if self.pmean: nall += 1
            matvec = partial(self.matrixVector, Ain)
            AP = self.getVelocitySolver(Ain[0])
            SP = self.getPressureSolver(Ain[0], Ain[1], AP)
            matvecprec=self.getPrecMult(Ain, AP, SP)
            S = solvers.cfd.SystemSolver(n=nall, matvec=matvec, matvecprec=matvecprec, method=method, disp=disp, atol=atol, rtol=rtol)
            return S.solve(b=bin, x0=uin)
        else:
            raise ValueError(f"unknown solve '{solver=}'")
    def computeRhs(self, b=None, u=None, coeffmass=None):
        b = self._zeros()
        bs  = self._split(b)
        bv,bp = bs[0], bs[1]
        # if self.pmean: bv, bp, bmean = self._split(b)
        # else: bv, bp = self._split(b)
        # bv = np.zeros(self.femv.nunknowns() * self.ncomp)
        # bp = np.zeros(self.femp.nunknowns())
        if 'rhs' in self.problemdata.params.fct_glob:
            rhsv, rhsp = self.problemdata.params.fct_glob['rhs']
            if rhsv: self.femv.computeRhsCells(bv, rhsv)
            if rhsp: self.femp.computeRhsCells(bp, rhsp)
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsneu = self.problemdata.bdrycond.colorsOfType("Neumann")
        colorsnav = self.problemdata.bdrycond.colorsOfType("Navier")
        self.femv.computeRhsBoundary(bv, colorsneu, self.problemdata.bdrycond.fct)
        if self.dirichletmethod=='strong':
            self.vectorBoundary((bv, bp), self.problemdata.bdrycond.fct, self.bdrydata, self.dirichletmethod)
        else:
            vdir = self.femv.interpolateBoundary(colorsdir, self.problemdata.bdrycond.fct)
            self.computeRhsBdryNitscheDirichlet((bv,bp), colorsdir, vdir, self.mucell)
            self.computeRhsBdryNitscheNavier((bv,bp), colorsnav, vdir, self.mucell)
        if not self.pmean: return b
        if self.problemdata.solexact is not None:
            p = self.problemdata.solexact[1]
            bmean = self.femp.computeMean(p)
        else: bmean=0
        b[-1] = bmean
        return b
    def computeForm(self, u):
        d = np.zeros_like(u)
        if self.pmean: 
            v, p, lam = self._split(u)
            dv, dp, dlam = self._split(d)
        else: 
            v, p = self._split(u)
            dv, dp = self._split(d)
        # d2 = self.matrixVector(self.A, u)
        self.femv.computeFormLaplace(self.mucell, dv, v)
        self.femv.computeFormDivGrad(dv, dp, v, p)
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        if self.dirichletmethod == 'strong':
            self.femv.formBoundary(dv, self.bdrydata, self.dirichletmethod)
        else:
            self.computeFormBdryNitsche(dv, dp, v, p, colorsdir, self.mucell)
        if self.pmean:
            self.computeFormMeanPressure(dp, dlam, p, lam)
        # if not np.allclose(d,d2):
        #     raise ValueError(f"{d=}\n{d2=}")
        return d
    def computeMatrix(self, u=None):
        A = self.femv.computeMatrixLaplace(self.mucell)
        B = self.femv.computeMatrixDivergence()
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsnav = self.problemdata.bdrycond.colorsOfType("Navier")
        if self.dirichletmethod == 'strong':
            A, B = self.matrixBoundary(A, B, self.bdrydata, self.dirichletmethod)
        else:
            #TODO eviter le retour de A,B
            A, B = self.computeMatrixBdryNitscheDirichlet(A, B, colorsdir, self.mucell)
            A, B = self.computeMatrixBdryNitscheNavier(A, B, colorsnav, self.mucell)
        if not self.pmean:
            return [A, B]
        ncells = self.mesh.ncells
        rows = np.zeros(ncells, dtype=int)
        cols = np.arange(0, ncells)
        C = sparse.coo_matrix((self.mesh.dV, (rows, cols)), shape=(1, ncells)).tocsr()
        return [A,B,C]
    def computeFormMeanPressure(self,dp, dlam, p, lam):
        dlam += self.mesh.dV.dot(p)
        dp += lam*self.mesh.dV
    def computeBdryNormalFluxNitsche(self, v, p, colors):
        nfaces, ncells, ncomp = self.mesh.nfaces, self.mesh.ncells, self.ncomp
        bdryfct = self.problemdata.bdrycond.fct
        flux, omega = np.zeros(shape=(len(colors),ncomp)), np.zeros(len(colors))
        xf, yf, zf = self.mesh.pointsf.T
        cellgrads = self.femv.fem.cellgrads
        facesOfCell = self.mesh.facesOfCells
        mucell = self.mucell
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            cells = self.mesh.cellsOfFaces[faces,0]
            normalsS = self.mesh.normals[faces][:,:ncomp]
            dS = np.linalg.norm(normalsS, axis=1)
            if color in bdryfct:
                # bfctv, bfctp = bdryfct[color]
                bfctv = bdryfct[color]
                # dirichv = np.hstack([bfctv(xf[faces], yf[faces], zf[faces])])
                dirichv = np.vstack([f(xf[faces], yf[faces], zf[faces]) for f in bfctv])
            flux[i] -= np.einsum('f,fk->k', p[cells], normalsS)
            indfaces = self.mesh.facesOfCells[cells]
            ind = npext.positionin(faces, indfaces).astype(int)
            for icomp in range(ncomp):
                vicomp = v[icomp+ ncomp*facesOfCell[cells]]
                flux[i,icomp] = np.einsum('fj,f,fi,fji->', vicomp, mucell[cells], normalsS, cellgrads[cells, :, :ncomp])
                vD = v[icomp+ncomp*faces]
                if color in bdryfct:
                    vD -= dirichv[icomp]
                flux[i,icomp] -= self.dirichlet_nitsche*np.einsum('f,fi,fi->', vD * mucell[cells], normalsS, cellgrads[cells, ind, :ncomp])
            # for icomp in range(ncomp):
            #     mat2 = np.einsum('fj,f->fj', mat, dirichv[icomp])
            #     np.add.at(bv, icomp + ncomp * indfaces, -mat2)
            # ind = npext.positionin(faces, indfaces).astype(int)
            # for icomp in range(ncomp):
            #     bv[icomp + ncomp * faces] += self.dirichlet_nitsche * np.choose(ind, mat.T) * dirichv[icomp]
        return flux.T
    def computeRhsBdryNitscheDirichlet(self, b, colorsdir, vdir, mucell, coeff=1):
        bv, bp = b
        ncomp  = self.ncomp
        faces = self.mesh.bdryFaces(colorsdir)
        cells = self.mesh.cellsOfFaces[faces,0]
        normalsS = self.mesh.normals[faces][:,:ncomp]
        np.add.at(bp, cells, -np.einsum('nk,nk->n', coeff*vdir[faces], normalsS))
        self.femv.computeRhsNitscheDiffusion(bv, mucell, colorsdir, vdir, ncomp)
    def computeRhsBdryNitscheNavier(self, b, colors, vdir, mucell, coeff=1):
        if len(colors): raise NotImplementedError("trop tot")
    def computeRhsBdryNitscheOld(self, b, colorsdir, bdryfct, mucell, coeff=1):
        bv, bp = b
        xf, yf, zf = self.mesh.pointsf.T
        nfaces, ncells, dim, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension, self.ncomp
        cellgrads = self.femv.fem.cellgrads
        for color in colorsdir:
            faces = self.mesh.bdrylabels[color]
            cells = self.mesh.cellsOfFaces[faces,0]
            normalsS = self.mesh.normals[faces][:,:ncomp]
            dS = np.linalg.norm(normalsS,axis=1)
            # normalsS = normalsS/dS[:,np.newaxis]
            if not color in bdryfct.keys(): continue
            bfctv = bdryfct[color]
            # dirichv = np.hstack([bfctv(xf[faces], yf[faces], zf[faces])])
            dirichv = np.vstack([f(xf[faces], yf[faces], zf[faces]) for f in bfctv])
            # print(f"{dirichv.shape=} {normalsS.shape=}")
            bp[cells] -= np.einsum('kn,nk->n', coeff*dirichv, normalsS)
            mat = np.einsum('f,fi,fji->fj', coeff*mucell[cells], normalsS, cellgrads[cells, :, :dim])
            indfaces = self.mesh.facesOfCells[cells]
            for icomp in range(ncomp):
                mat2 = np.einsum('fj,f->fj', mat, dirichv[icomp])
                np.add.at(bv, icomp+ncomp*indfaces, -mat2)
            ind = npext.positionin(faces, indfaces).astype(int)
            for icomp in range(ncomp):
                bv[icomp+ncomp*faces] += self.dirichlet_nitsche * np.choose(ind, mat.T)*dirichv[icomp]
        # print(f"{bv.shape=} {bp.shape=}")
        if len(colors): raise NotImplementedError("trop tot")
    def computeFormBdryNitsche(self, dv, dp, v, p, colorsdir, mu):
        ncomp, dim  = self.femv.ncomp, self.mesh.dimension
        self.femv.computeFormNitscheDiffusion(dv, v, mu, colorsdir, ncomp)
        faces = self.mesh.bdryFaces(colorsdir)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :self.ncomp]
        for icomp in range(ncomp):
            r = np.einsum('f,f->f', p[cells], normalsS[:,icomp])
            np.add.at(dv[icomp::ncomp], faces, r)
            r = np.einsum('f,f->f', normalsS[:,icomp], v[icomp::ncomp][faces])
            np.add.at(dp, cells, -r)
    def computeMatrixBdryNitscheDirichlet(self, A, B, colorsdir, mucell):
        nfaces, ncells, ncomp, dim  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp, self.mesh.dimension
        A += self.femv.computeMatrixNitscheDiffusion(mucell, colorsdir, ncomp)
        faces = self.mesh.bdryFaces(colorsdir)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :self.ncomp]
        indfaces = np.repeat(ncomp * faces, ncomp)
        for icomp in range(ncomp): indfaces[icomp::ncomp] += icomp
        cols = indfaces.ravel()
        rows = cells.repeat(ncomp).ravel()
        mat = normalsS.ravel()
        BN = sparse.coo_matrix((mat, (rows, cols)), shape=(ncells, ncomp*nfaces)).tocsr()
        B -= BN
        return A,B
    def computeMatrixBdryNitscheNavier(self, A, B, colors, mucell):
        if len(colors): raise NotImplementedError("trop tot")
        return A,B
    def computeMatrixBdryNitscheOld(self, A, B, colorsdir, mucell):
        nfaces, ncells, ncomp, dim  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp, self.mesh.dimension
        A += self.femv.computeMatrixNitscheDiffusion(mucell, colorsdir, ncomp)
        # nloc = dim+1
        # nlocncomp = ncomp * nloc
        # cellgrads = self.femv.fem.cellgrads
        faces = self.mesh.bdryFaces(colorsdir)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :self.ncomp]
        indfaces = np.repeat(ncomp * faces, ncomp)
        for icomp in range(ncomp): indfaces[icomp::ncomp] += icomp
        cols = indfaces.ravel()
        rows = cells.repeat(ncomp).ravel()
        mat = normalsS.ravel()
        BN = sparse.coo_matrix((mat, (rows, cols)), shape=(ncells, ncomp*nfaces)).tocsr()

        # mat = np.einsum('f,fi,fji->fj', mucell[cells], normalsS, cellgrads[cells, :, :dim])
        # mats = mat.repeat(ncomp)
        # dofs = self.mesh.facesOfCells[cells, :]
        # cols = np.repeat(ncomp * dofs, ncomp).reshape(dofs.shape[0], nloc, ncomp)
        # for icomp in range(ncomp): cols[:,:,icomp::ncomp] += icomp
        # indfaces = np.repeat(ncomp * faces, ncomp*nloc).reshape(dofs.shape[0],nloc,ncomp)
        # for icomp in range(ncomp): indfaces[:,:,icomp::ncomp] += icomp
        # cols = cols.reshape(dofs.shape[0],nloc*ncomp)
        # rows = indfaces.reshape(dofs.shape[0],nloc*ncomp)
        # AN = sparse.coo_matrix((mats, (rows.ravel(), cols.ravel())), shape=(ncomp*nfaces, ncomp*nfaces)).tocsr()
        # AD = sparse.diags(AN.diagonal(), offsets=(0), shape=(ncomp*nfaces, ncomp*nfaces))
        # A = A- AN -AN.T + self.dirichlet_nitsche*AD
        # print(f"{AN.todense()=}")
        # print(f"{AD.toarray()=}")
        # print(f"{A.toarray()=}")
        # print(f"{A=}")
        B -= BN
        return A,B
    def vectorBoundary(self, b, bdryfctv, bdrydata, method):
        bv, bp = b
        bv = self.femv.vectorBoundary(bv, bdryfctv, bdrydata, method)
        facesdirall, facesinner, colorsdir, facesdirflux = bdrydata.facesdirall, bdrydata.facesinner, bdrydata.colorsdir, bdrydata.facesdirflux
        nfaces, ncells, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp
        bdrydata.bsaved = {}
        for key, faces in facesdirflux.items():
            indfaces = np.repeat(ncomp * faces, ncomp)
            for icomp in range(ncomp): indfaces[icomp::ncomp] += icomp
            bdrydata.bsaved[key] = bv[indfaces]
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        #suppose strong-trad
        bp -= bdrydata.B_inner_dir * bv[inddir]
        return (bv,bp)
    def matrixBoundary(self, A, B, bdrydata, method):
        A = self.femv.matrixBoundary(A, bdrydata, method)
        facesdirall, facesinner, colorsdir, facesdirflux = bdrydata.facesdirall, bdrydata.facesinner, bdrydata.colorsdir, bdrydata.facesdirflux
        nfaces, ncells, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp
        bdrydata.Bsaved = {}
        for key, faces in facesdirflux.items():
            nb = faces.shape[0]
            helpB = sparse.dok_matrix((ncomp*nfaces, ncomp*nb))
            for icomp in range(ncomp):
                for i in range(nb): helpB[icomp + ncomp*faces[i], icomp + ncomp*i] = 1
            bdrydata.Bsaved[key] = B.dot(helpB)
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        bdrydata.B_inner_dir = B[:,:][:,inddir]
        help = np.ones((ncomp * nfaces))
        help[inddir] = 0
        help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
        B = B.dot(help)
        return A,B
    def computeBdryNormalFluxStrong(self, v, p, colors):
        nfaces, ncells, ncomp, bdrydata  = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.bdrydata
        flux, omega = np.zeros(shape=(ncomp,len(colors))), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            As = bdrydata.Asaved[color]
            Bs = bdrydata.Bsaved[color]
            res = bdrydata.bsaved[color] - As * v + Bs.T * p
            for icomp in range(ncomp):
                flux[icomp, i] = np.sum(res[icomp::ncomp])
            # print(f"{flux=}")
            #TODO flux Stokes Dirichlet strong wrong
        return flux

#=================================================================#
if __name__ == '__main__':
    raise NotImplementedError("Pas encore de test")
