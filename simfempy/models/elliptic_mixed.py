from simfempy.models.elliptic_base import EllipticBase

# ================================================================= #
class EllipticMixed(EllipticBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def meshSet(self):
        super().meshSet()
        self.pureneumann = not ('Dirichlet' in self.problemdata.bdrycond.type.values() or 'Robin' in self.problemdata.bdrycond.type.values())
        # print(f"{self.pureneumann=} {self.dirichletmethod=}")
        self.rt.setMesh(self.mesh)
        self.d0.setMesh(self.mesh)
        colorsneumann = self.problemdata.bdrycond.colorsOfType("Neumann")
        self.bdrydata = self.rt.prepareBoundary(colorsneumann)
        self.divcoeffinv = 1/self.kheatcell
    def tofemvector(self, u):
        assert 0

        # def sol_to_data(self, u):
        #     nfaces, dim = self.mesh.nfaces, self.mesh.dimension
        #     v = self.vectorview.get_part(0, u)
        #     p = self.vectorview.get_part(1, u)
        #     # sigma, u = uin[0], uin[1]
        #     # sigma, u = u[:nfaces], u[nfaces:]
        #     data = {'point': {}, 'cell': {}, 'global': {}}
        #     # point_data, side_data, cell_data, global_data = {}, {}, {}, {}
        #     data['cell']['p'] = p
        #     vc = self.rt.toCell(v)
        #     pn = self.rt.reconstruct(p, vc, self.divcoeffinv)
        #     data['point']['pn'] = pn
        #     for i in range(dim):
        #         data['cell']['v{:1d}'.format(i)] = vc[:, i]
        #     return data

        return fems.femvector.FemVector(data = u, vectorview=self.vectorview, fems=[self.fem])
    def getNcomps(self, mesh):
        return [1, 1]
    def getSystemSize(self):
        ns = [self.rt.nunknowns(), self.d0.nunknowns()]
        return ns
    # def computeForm(self, u, coeffmass=None):
    #     raise NotImplementedError(f"computeForm for rt")
    def computeMatrix(self, u=None, coeffmass=None):
        # print("Hallo computeMatrix")
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        A = self.rt.constructMass(diffinvcell=self.divcoeffinv)
        B = self.rt.constructDiv()
        A += self.rt.computeBdryMassMatrix(colorsrobin, bdrycond.param)
        if self.hasconvection:
            raise NotImplementedError(f"convection for rt")
        if coeffmass is not None:
            raise NotImplementedError(f"recation for rt")
        if self.pureneumann:
            A, B, self.bdrydata = self.rt.matrixNeumann(A, B, self.bdrydata)
        positions = [[{'pos':(0,0)}], [{'pos':(1,0)}, {'pos':(0,1), 'trp':True, 'scl':1}]]
        return matrixsystem.MatrixSystem(self.vectorview, [A,B], positions)
        raise ValueError(f"{M=}")
        return saddle_point.SaddlePointSystem(self.vectorview, [A, B], singleA=False)
        # return A, B
    def computeRhs(self, b=None, coeffmass=None, u=None):
        assert b == None
        b = np.zeros(self.vectorview.n())
        bsides = self.vectorview.get_part(0, b)
        bcells = self.vectorview.get_part(1, b)
        # bsides = np.zeros(self.mesh.nfaces)
        # bcells = np.zeros(self.mesh.ncells)
        bdrycond = self.problemdata.bdrycond
        colorsrobin = bdrycond.colorsOfType("Robin")
        colorsdirrobin = bdrycond.colorsOfType(["Dirichlet","Robin"])
        colorsneu = bdrycond.colorsOfType("Neumann")
        if 'rhs' in self.problemdata.params.fct_glob:
            fp1 = self.d0.interpolate(self.problemdata.params.fct_glob['rhs'])
            self.d0.massDot(bcells, fp1)
        for color in colorsdirrobin:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS, axis=1)
            normalsS = normalsS / dS[:, np.newaxis]
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T
            try:
                ud = bdrycond.fct[color](xf, yf, zf, nx, ny, nz)
            except:
                ud = bdrycond.fct[color](xf, yf, zf)
            if color in colorsrobin: dS /= bdrycond.param[color]
            bsides[faces] += dS * ud
            if self.hasconvection:
                faces = self.mesh.bdrylabels[color]
                # normalsS = self.mesh.normals[faces]
                # dS = np.linalg.norm(normalsS, axis=1)
                # normalsS = normalsS / dS[:, np.newaxis]
                # xf, yf, zf = self.mesh.pointsf[faces].T
                # beta = np.array(self.convection(xf, yf, zf))
                # bn = np.einsum("ij,ij->j", beta, normalsS.T)
                # # print("bn", bn)
                # bn[bn<=0] = 0
                cells = self.mesh.cellsOfFaces[faces,0]
                # bcells[cells] += bn*ud*dS

                bn = self.convdata.betart[faces]
                bcells[cells] += bn*ud

        help = np.zeros(self.mesh.nfaces)
        for color in colorsneu:
            if not color in bdrycond.fct or not bdrycond.fct[color]: continue
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS, axis=1)
            normalsS = normalsS / dS[:, np.newaxis]
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T
            help[faces] += bdrycond.fct[color](xf, yf, zf, nx, ny, nz)
        # print(f"{self.bdrydata=}")
        # print(f"{type(self.bdrydata.A_inner_neum)=}")
        if self.pureneumann:
            bsides[self.bdrydata.facesinner] -= self.bdrydata.A_inner_neum*help[self.bdrydata.facesneumann]
            bsides[self.bdrydata.facesneumann] += self.bdrydata.A_neum_neum*help[self.bdrydata.facesneumann]
            bcells -= self.bdrydata.B_inner_neum*help[self.bdrydata.facesneumann]
        return b
        # return bsides, bcells
    def postProcess(self, u):
        assert 0
        nfaces, dim =  self.mesh.nfaces, self.mesh.dimension
        v = self.vectorview.get_part(0, u)
        p = self.vectorview.get_part(1, u)
        vc = self.rt.toCell(v)
        pn = self.rt.reconstruct(p, vc, self.divcoeffinv)
        data = {}
        if self.application.exactsolution:
            errp, errs, errpn, pe, ve = self.computeError(self.application.exactsolution, p, vc, pn)
            data['cell']={}
            data['cell']['err_p'] = np.abs(pe - p)
            for i in range(dim):
                data['cell']['err_v{:1d}'.format(i)] = np.abs(ve[i] - vc[:,i])
            data['scalar']={}
            data['scalar']['err_L2c'] = errp
            data['scalar']['err_L2n'] = errpn
            data['scalar']['err_Flux'] = errs
        if self.problemdata.postproc:
            types = ["bdry_mean", "bdry_fct", "bdry_nflux", "pointvalues", "meanvalues", "linemeans"]
            for name, type in self.problemdata.postproc.type.items():
                colors = self.problemdata.postproc.colors(name)
                if type == types[0]:
                    data['scalar'][name] = self.computeBdryMean(pn, colors)
                elif type == types[1]:
                    data['scalar'][name] = self.fem.computeBdryFct(u, colors)
                elif type == types[2]:
                    data['scalar'][name] = self.computeBdryDn(v, colors)
                elif type == types[3]:
                    data['scalar'][name] = self.fem.computePointValues(u, colors)
                elif type == types[4]:
                    data['scalar'][name] = self.fem.computeMeanValues(u, colors)
                elif type == types[5]:
                    data['scalar'][name] = self.fem.computeLineValues(u, colors)
                else:
                    raise ValueError(f"unknown postprocess type '{type}' for key '{name}'\nknown types={types=}")
        return data
    def computeBdryDn(self, u, colors):
        flux, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            flux[i] = np.sum(dS*u[faces])
        return flux
        return flux/omega
    def computeBdryMean(self, pn, colors):
        mean, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            mean[i] = np.sum(dS*np.mean(pn[self.mesh.faces[faces]],axis=1))
        return mean/omega
    def computeError(self, solexact, p, vc, pn):
        nfaces, dim =  self.mesh.nfaces, self.mesh.dimension
        xc, yc, zc = self.mesh.pointsc.T
        pex = solexact(xc, yc, zc)
        errp = np.sqrt(np.sum((pex-p)**2* self.mesh.dV))
        errv = 0
        vexx=[]
        for i in range(dim):
            solxi = self.kheatcell*solexact.d(i, xc, yc, zc)
            errv += np.sum( (solxi-vc[:,i])**2* self.mesh.dV)
            vexx.append(solxi)
        errv = np.sqrt(errv)
        x, y, z = self.mesh.points.T
        errpn = solexact(x, y, z) - pn
        errpn = errpn**2
        errpn= np.mean(errpn[self.mesh.cells], axis=1)
        errpn = np.sqrt(np.sum(errpn* self.mesh.dV))
        return errp, errv, errpn, pex, vexx

    def _to_single_matrix(self, Ain):
        A, B = Ain
        ncells = self.mesh.ncells
        help = np.zeros((ncells))
        help = scipy.sparse.dia_matrix((help, 0), shape=(ncells, ncells))
        A1 = scipy.sparse.hstack([A, B.T])
        A2 = scipy.sparse.hstack([B, help])
        Aall = scipy.sparse.vstack([A1, A2])
        return Aall.tocsr()
    # def computelinearSolver(self, A):
    #     if isinstance(self.linearsolver,str):
    #         args = {'method': self.linearsolver}
    #     else:
    #         args = self.linearsolver.copy()
    #     return simfempy.linalg.linalg.getLinearSolver(**args)
    # def solvelinear(self, Ain, bin, u=None, solver = None, verbose=0):
    #     if solver is None: solver = self.linearsolver
    #     if solver == 'spsolve':
    #         # print("bin", bin)
    #         Aall = self._to_single_matrix(Ain)
    #         b = np.concatenate((bin[0], bin[1]))
    #         u =  splinalg.spsolve(Aall, b, permc_spec='COLAMD')
    #         # print("u", u)
    #         return u, 1
    #     elif solver == 'gmres':
    #         nfaces, ncells = self.mesh.nfaces, self.mesh.ncells
    #         counter = simfempy.tools.iterationcounter.IterationCounter(name=solver, verbose=verbose)
    #         # Aall = self._to_single_matrix(Ain)
    #         # M2 = splinalg.spilu(Aall, drop_tol=0.2, fill_factor=2)
    #         # M_x = lambda x: M2.solve(x)
    #         # M = splinalg.LinearOperator(Aall.shape, M_x)
    #         A, B = Ain
    #         A, B = A.tocsr(), B.tocsr()
    #         D = scipy.sparse.diags(1/A.diagonal(), offsets=(0), shape=(nfaces,nfaces))
    #         S = -B*D*B.T
    #         import pyamg
    #         config = pyamg.solver_configuration(S, verb=False)
    #         ml = pyamg.rootnode_solver(S, B=config['B'], smooth='energy')
    #         # Ailu = splinalg.spilu(A, drop_tol=0.2, fill_factor=2)
    #         def amult(x):
    #             v,p = x[:nfaces],x[nfaces:]
    #             return np.hstack( [A.dot(v) + B.T.dot(p), B.dot(v)])
    #         Amult = splinalg.LinearOperator(shape=(nfaces+ncells,nfaces+ncells), matvec=amult)
    #         def pmult(x):
    #             v,p = x[:nfaces],x[nfaces:]
    #             w = D.dot(v)
    #             # w = Ailu.solve(v)
    #             q = ml.solve(p - B.dot(w), maxiter=1, tol=1e-16)
    #             w = w - D.dot(B.T.dot(q))
    #             # w = w - Ailu.solve(B.T.dot(q))
    #             return np.hstack( [w, q] )
    #         P = splinalg.LinearOperator(shape=(nfaces+ncells,nfaces+ncells), matvec=pmult)
    #         u,info = splinalg.lgmres(Amult, bin, M=P, callback=counter, atol=1e-12, tol=1e-12, inner_m=10, outer_k=4)
    #         if info: raise ValueError("no convergence info={}".format(info))
    #         # print("u", u)
    #         return u, counter.niter
    #     else:
    #         raise NotImplementedError("solver '{}' ".format(solver))
