import numpy as np
import scipy.sparse
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg
from simfempy import solvers
from simfempy import fems
import simfempy.tools.analyticalsolution
import simfempy.tools.iterationcounter

# =================================================================#
class Stokes(solvers.solver.Solver):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linearsolver = 'gmres'
        if 'problem' in kwargs:
            function = kwargs.pop('problem').split('_')[1]
            if function == 'Linear':
                # self.solexact.append(simfempy.tools.analyticalsolution.AnalyticalSolution('x+y'))
                self.solexact.append(simfempy.tools.analyticalsolution.AnalyticalSolution('0'))
            elif function == 'Quadratic':
                self.solexact[0] = simfempy.tools.analyticalsolution.AnalyticalSolution('x*x-2*y*x')
                self.solexact[1] = simfempy.tools.analyticalsolution.AnalyticalSolution('-2*x*y+y*y')
                self.solexact.append(simfempy.tools.analyticalsolution.AnalyticalSolution('x+y'))
            else:
                raise NotImplementedError("unknown function '{}'".format(function))
        if 'fem' in kwargs:
            fem = kwargs.pop('fem')
        else:
            fem = 'cr1'
        if fem == 'p1':
            self.femv = fems.femp1sys.FemP1()
            raise NotImplementedError("p1 not ready")
        elif fem == 'cr1':
            self.femp = fems.femd0.FemD0()
            self.femv = fems.femcr1sys.FemCR1()
        else:
            raise ValueError("unknown fem '{}'".format(fem))
        if 'mu' in kwargs:
            self.mu = kwargs.pop('mu')
            self.mu = np.vectorize(self.mu)
        else:
            self.mu = np.vectorize(lambda x: 1.0)
        if 'method' in kwargs:
            self.method = kwargs.pop('method')
        else:
            self.method = "trad"
        self.rhsmethod = kwargs.pop('rhsmethod')
        if self.rhsmethod == "rt":
            self.femrt = fems.femrt0.FemRT0()

    def setMesh(self, mesh):
        # t0 = time.time()
        self.mesh = mesh
        self.femv.setMesh(self.mesh, self.ncomp)
        self.femp.setMesh(self.mesh)
        self.pmean = True
        colorsdir = []
        for color, type in self.bdrycond.type.items():
            if type == "Dirichlet": colorsdir.append(color)
            else: self.pmean = False
        self.bdrydata = self.femv.prepareBoundary(colorsdir, self.postproc)
        self.mucell = self.mu(self.mesh.cell_labels)
        # t1 = time.time()
        # self.timer['setmesh'] = t1 - t0
        self.pstart = self.ncomp*self.mesh.nfaces
        if self.rhsmethod == "rt":
            self.femrt.setMesh(self.mesh)
            self.massrt = self.femrt.constructMass()
        if self.solexact:
            def fctv(x,y,z):
                rhs = np.zeros(shape=(self.ncomp, x.shape[0]))
                for i in range(self.ncomp):
                    for j in range(self.ncomp):
                        rhs[i] -= self.mucell[0] * self.solexact[i].dd(j, j, x, y, z)
                    rhs[i] += self.solexact[self.ncomp].d(i, x, y, z)
                return rhs
            def fctp(x,y,z):
                rhs = np.zeros(x.shape[0])
                for i in range(self.ncomp):
                    rhs += self.solexact[i].d(i, x, y, z)
                return rhs
            self.rhs = fctv, fctp

    def solve(self, iter, dirname):
        return self.solveLinear()

    def computeRhs(self):
        ncomp, nfaces, ncells = self.ncomp, self.mesh.nfaces, self.mesh.ncells
        if self.pmean: nall = nfaces * ncomp + self.mesh.ncells +1
        else: nall = nfaces * ncomp + self.mesh.ncells
        b = np.zeros(nall)
        rhsv, rhsp = self.rhs
        if self.rhsmethod=='rt':
            xf, yf, zf = self.femv.pointsf[:, 0], self.femv.pointsf[:, 1], self.femv.pointsf[:, 2]
            rhsall = np.array(rhsv(xf, yf, zf))
            normals, sigma, dV = self.mesh.normals, self.mesh.sigma, self.mesh.dV
            dS = linalg.norm(normals, axis=1)
            rhsn = np.zeros(nfaces)
            for i in range(ncomp):
                rhsn += rhsall[i] * normals[:,i] / dS
            rhsn = self.massrt*rhsn
            for i in range(ncomp):
                b[i * nfaces:(i + 1) * nfaces] = rhsn*normals[:,i]/dS

            # xc, yc, zc = self.mesh.pointsc.T
            # rhsall = np.array(rhsv(xc, yc, zc))
            # ncells, nfaces, facesOfCells = self.mesh.ncells, self.mesh.nfaces, self.mesh.facesOfCells
            # normals, sigma, dV = self.mesh.normals, self.mesh.sigma, self.mesh.dV
            # scale = 1/self.mesh.dimension
            # dfaces = np.zeros(nfaces)
            # for icell in range(ncells):
            #     for ii, iface in enumerate(facesOfCells[icell]):
            #         inode = self.mesh.simplices[icell, ii]
            #         for j in range(ncomp):
            #             dfaces[iface] += sigma[icell, ii] * rhsall[j][icell] * (
            #                         self.mesh.pointsc[icell, j] - self.mesh.points[inode, j]) * scale
            # for i in range(ncomp):
            #     b[i * nfaces:(i + 1) * nfaces][:] += normals[:, i] * dfaces
            if rhsp:
                xc, yc, zc = self.mesh.pointsc.T
                b[self.pstart:self.pstart+ncells] = self.mesh.dV*np.array(rhsp(xc, yc, zc))
        elif self.rhsmethod=='cr':
            x, y, z = self.femv.pointsf[:, 0], self.femv.pointsf[:, 1], self.femv.pointsf[:, 2]
            bfaces = rhsv(x,y,z)
            for i in range(ncomp):
                b[i * nfaces:(i + 1) * nfaces] = self.femv.massmatrix * bfaces[i]
            if rhsp:
                xc, yc, zc = self.mesh.pointsc.T
                b[self.pstart:self.pstart + ncells] = self.mesh.dV * np.array(rhsp(xc, yc, zc))
        else:
            raise ValueError("don't know rhsmethod='{}'".format(self.rhsmethod))

        # elif self.solexact:
        #     x, y, z = self.femv.pointsf[:,0], self.femv.pointsf[:,1], self.femv.pointsf[:,2]
        #     bfaces2 = self.rhs(x,y,z)
        #     for i in range(ncomp):
        #         bfaces = np.zeros(self.mesh.nfaces)
        #         for j in range(ncomp):
        #             bfaces -= self.mucell[0] * self.solexact[i].dd(j, j, x, y, z)
        #         bfaces += self.solexact[ncomp].d(i, x, y, z)
        #         assert np.allclose(bfaces, bfaces2[i])
        #         b[i*nfaces:(i+1)*nfaces] = self.femv.massmatrix * bfaces
        #     xc, yc, zc = self.mesh.pointsc[:, 0], self.mesh.pointsc[:, 1], self.mesh.pointsc[:, 2]
        #     bcells = np.zeros(self.mesh.ncells)
        #     for i in range(ncomp):
        #         bcells += self.solexact[i].d(i, xc, yc, zc)
        #     b[self.pstart:self.pstart+ncells] = self.mesh.dV*bcells

            # for i in range(ncomp):
            #     for icell in range(ncells):
            #         for ii, iface in enumerate(facesOfCells[icell]):
            #             inode = self.mesh.simplices[icell,ii]
            #             d=0
            #             for j in range(ncomp):
            #                 d += sigma[icell,ii]*rhsall[j][icell]*(self.mesh.pointsc[icell,j] - self.mesh.points[inode,j])*scale
            #             b[i*nfaces:(i+1)*nfaces][iface] += normals[iface,i]*d

        normals = self.mesh.normals
        for color, faces in self.mesh.bdrylabels.items():
            condition = self.bdrycond.type[color]
            if condition == "Neumann":
                normalsS = normals[faces]
                dS = linalg.norm(normalsS,axis=1)
                muS = self.mucell[self.mesh.cellsOfFaces[faces, 0]]
                xf, yf, zf = self.femv.pointsf[faces,0], self.femv.pointsf[faces,1], self.femv.pointsf[faces,2]
                nx, ny, nz = normalsS[:,0]/dS, normalsS[:,1]/dS, normalsS[:,2]/dS
                if self.solexact:
                    for i in range(ncomp):
                        bS = np.zeros(faces.shape[0])
                        for j in range(ncomp):
                            bS += muS * self.solexact[i].d(j, xf, yf, zf) * normalsS[:, j]
                        bS -= muS * self.solexact[ncomp](xf, yf, zf) * normalsS[:, i]
                        indices = i*nfaces + faces
                        b[indices] += bS
                else:
                    if not color in self.bdrycond.fct.keys(): continue
                    neumann = self.bdrycond.fct[color]
                    neumanns = neumann(xf, yf, zf, nx, ny, nz, lamS, muS)
                    for i in range(ncomp):
                        bS = dS * neumanns[i]
                        indices = i*nfaces + faces
                        b[indices] += bS
                # print("bS.shape", bS.shape)
                # print("indices.shape", indices.shape)

        # from ..meshes import plotmesh
        # plotmesh.meshWithData(self.mesh, point_data={"b_{:1d}".format(i):b[i::self.ncomp] for i in range(self.ncomp)})
        return b

    def matrix(self):
        nfaces, facesOfCells = self.mesh.nfaces, self.mesh.facesOfCells
        nnodes, ncells, ncomp, dV = self.mesh.nnodes, self.mesh.ncells, self.ncomp, self.mesh.dV
        nloc, cellgrads = self.femv.nloc, self.femv.cellgrads
        matA = np.zeros(ncells*nloc*nloc).reshape(ncells, nloc, nloc)
        for i in range(ncomp):
            matA += np.einsum('nk,nl->nkl', cellgrads[:, :, i], cellgrads[:, :, i])
        matA = ( matA.T*dV*self.mucell).T.flatten()
        cols = np.tile(facesOfCells, nloc).reshape(ncells, nloc, nloc)
        rows = cols.swapaxes(1, 2)
        # print("matA.shape", matA.shape)
        # print("cols.shape", cols.shape)
        A = scipy.sparse.coo_matrix((matA, (rows.reshape(-1), cols.reshape(-1))), shape=(nfaces, nfaces)).tocsr()

        rowsB = np.repeat(np.arange(ncells), ncomp*nloc).reshape(ncells * nloc, ncomp)
        # print("rowsB", rowsB)

        # colsB = np.repeat( ncomp*facesOfCells, ncomp).reshape(ncells * nloc, ncomp) + np.arange(ncomp)
        colsB = np.repeat( facesOfCells, ncomp).reshape(ncells * nloc, ncomp) + nfaces*np.arange(ncomp)
        # print("colsB", colsB.reshape(ncells, nloc, ncomp)[0])

        # matB = np.zeros(ncells*nloc*ncomp, dtype=float).reshape(ncells, nloc, ncomp)
        matB = (cellgrads[:,:,:ncomp].T*dV).T
        # print("matB", matB[0])
        B = scipy.sparse.coo_matrix((matB.reshape(-1), (rowsB.reshape(-1), colsB.reshape(-1))), shape=(ncells, nfaces*ncomp)).tocsr()
        print("A=", A)
        raise NotImplementedError
        # print("B", B[0])
        if self.pmean:
            rows = np.zeros(ncells, dtype=int)
            cols = np.arange(0, ncells)
            C = scipy.sparse.coo_matrix((dV, (rows, cols)), shape=(1, ncells)).tocsr()
            return (A,B,C)
        return (A,B)

    def boundary(self, Aall, b, u):
        if self.pmean:
            A, B, C = Aall
        else:
            A, B = Aall
        # print("A", A.todense())
        # print("B", B.todense())
        x, y, z = self.femv.pointsf[:, 0], self.femv.pointsf[:, 1], self.femv.pointsf[:, 2]
        facesdirall, facesinner, colorsdir, facesdirflux = self.bdrydata
        nfaces, ncells, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp
        self.bsaved = []
        self.Asaved = {}
        self.Bsaved = {}
        for key, faces in facesdirflux.items():
            nb = faces.shape[0]
            help = scipy.sparse.dok_matrix((nb, nfaces))
            for i in range(nb): help[i, faces[i]] = 1
            self.Asaved[key] = help.dot(A)
            # helpB = np.zeros((ncomp*nfaces))
            # for icomp in range(ncomp):
            #     helpB[icomp*nfaces + facesdirall] = 0
            # helpB = scipy.sparse.dia_matrix((helpB, 0), shape=(ncomp*nfaces, ncomp*nfaces))
            helpB = scipy.sparse.dok_matrix((ncomp*nfaces, ncomp*nb))
            for icomp in range(ncomp):
                for i in range(nb): helpB[icomp*nfaces + faces[i], icomp*nb + i] = 1
            self.Bsaved[key] = B.dot(helpB)
        self.A_inner_dir = A[facesinner, :][:, facesdirall]
        for icomp in range(ncomp):
            self.bsaved.append({})
            for key, faces in facesdirflux.items():
                self.bsaved[icomp][key] = b[icomp*nfaces + faces]
        if self.method == 'trad':
            for color in colorsdir:
                faces = self.mesh.bdrylabels[color]
                dirichlet = self.bdrycond.fct[color]
                dirs = dirichlet(x[faces], y[faces], z[faces])
                # print("dirs", dirs)
                for icomp in range(ncomp):
                    b[icomp*nfaces + faces] = dirs[icomp]
                    u[icomp*nfaces + faces] = b[icomp*nfaces + faces]
            for icomp in range(ncomp):
                indin = icomp*nfaces + facesinner
                inddir = icomp*nfaces + facesdirall
                b[indin] -= self.A_inner_dir * b[inddir]
                # print("B.indices", B.indices)
                # print("b[self.pstart:]", b[self.pstart:])
                # print("inddir", inddir)
                # print("B[:,:][:,inddir]", B[:,:][:,inddir])
                b[self.pstart:self.pstart+ncells] -= B[:,:][:,inddir] * b[inddir]

            help = np.ones((nfaces))
            help[facesdirall] = 0
            help = scipy.sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            A = help.dot(A.dot(help))
            help = np.zeros((nfaces))
            help[facesdirall] = 1.0
            help = scipy.sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            A += help
        else:
            for color in colorsdir:
                faces = self.mesh.bdrylabels[color]
                dirichlet = self.bdrycond.fct[color]
                dirs = dirichlet(x[faces], y[faces], z[faces])
                for icomp in range(ncomp):
                    u[icomp*nfaces + faces] = dirs[icomp]
                    b[icomp*nfaces + faces] = 0
            self.A_dir_dir = A[facesdirall, :][:, facesdirall]
            for icomp in range(ncomp):
                indin = icomp*nfaces + facesinner
                inddir = icomp*nfaces + facesdirall
                b[indin] -= self.A_inner_dir * u[inddir]
                b[inddir] += self.A_dir_dir * u[inddir]
            help = np.ones((nfaces))
            help[facesdirall] = 0
            help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            help2 = np.zeros((nfaces))
            help2[facesdirall] = 1
            help2 = sparse.dia_matrix((help2, 0), shape=(nfaces, nfaces))
            A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        # B
        help = np.ones((ncomp * nfaces))
        for icomp in range(ncomp):
            help[icomp*nfaces + facesdirall] = 0
        help = scipy.sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
        B = B.dot(help)
        # print("B", B[0])
        # print("after A", A.todense())
        # print("after B", B.todense())
        if self.pmean:
            return (A, B, C), b, u
        return (A,B), b, u

    def computeBdryMean(self, u, key, data):
        nfaces, ncomp  = self.mesh.nfaces, self.femv.ncomp
        colors = [int(x) for x in data.split(',')]
        mean, omega = [0 for i in range(self.ncomp)], 0
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            for i in range(ncomp):
                mean[i] += np.sum(dS * np.mean(u[i*nfaces + self.mesh.faces[faces]], axis=1))
        return mean

    def computeBdryDn(self, u, key, data):
        nfaces, ncells, ncomp, pstart  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp, self.pstart
        # colors = [int(x) for x in data.split(',')]
        # omega = 0
        # for color in colors:
        #     omega += np.sum(linalg.norm(self.mesh.normals[self.mesh.bdrylabels[color]],axis=1))
        # print("###",self.bsaved[key].shape)
        # print("###",(self.Asaved[key] * u).shape)
        flux = []
        pcontrib = self.Bsaved[key].T*u[pstart: pstart+ncells]
        nb = self.Bsaved[key].shape[1]//ncomp
        for icomp in range(ncomp):
            res = self.bsaved[icomp][key] - self.Asaved[key] * u[icomp*nfaces:(icomp+1)*nfaces]
            # print("self.Asaved.shape", self.Asaved[key].shape)
            # print("self.Bsaved.shape", self.Bsaved[key].shape)
            # print("self.bsaved[icomp][key].shape", self.bsaved[icomp][key].shape)
            # print("res.shape", res.shape)
            # print("pcontrib.shape", pcontrib.shape)
            # print("nfaces", nfaces)
            flux.append(np.sum(res+pcontrib[icomp*nb:(icomp+1)*nb]))
        return flux

    def computeErrorL2V(self, solex, uh):
        nfaces, ncomp  = self.mesh.nfaces, self.femv.ncomp
        x, y, z = self.femv.pointsf[:,0], self.femv.pointsf[:,1], self.femv.pointsf[:,2]
        e = []
        err = []
        for icomp in range(self.ncomp):
            e.append(solex[icomp](x, y, z) - uh[icomp*nfaces:(icomp+1)*nfaces])
            err.append(np.sqrt(np.dot(e[icomp], self.femv.massmatrix * e[icomp])))
        return err, e

    def postProcess(self, u):
        nfaces, ncomp  = self.mesh.nfaces, self.femv.ncomp
        info = {}
        cell_data = {}
        point_data = {}
        for icomp in range(ncomp):
            point_data['V_{:02d}'.format(icomp)] = self.femv.tonode(u[icomp*nfaces:(icomp+1)*nfaces])
        p = u[self.pstart:self.pstart+self.mesh.ncells]
        cell_data['P'] = p
        if self.solexact:
            info['error'] = {}
            errv, ev = self.computeErrorL2V(self.solexact, u[:self.pstart])
            errp, ep = self.femp.computeErrorL2(self.solexact[-1], p)
            info['error']['L2-V'] = np.sum(errv)
            info['error']['L2-P'] = np.sum(errp)
            for icomp in range(self.ncomp):
                point_data['E_V{:02d}'.format(icomp)] = self.femv.tonode(ev[icomp])
            cell_data['E_P'] = ep
        info['postproc'] = {}
        for key, val in self.postproc.items():
            type, data = val.split(":")
            if type == "bdrymean":
                mean = self.computeBdryMean(u, key, data)
                assert len(mean) == self.ncomp
                for icomp in range(self.ncomp):
                    info['postproc']["{}_{:02d}".format(key, icomp)] = mean[icomp]
            elif type == "bdrydn":
                flux = self.computeBdryDn(u, key, data)
                assert len(flux) == self.ncomp
                for icomp in range(self.ncomp):
                    info['postproc']["{}_{:02d}".format(key, icomp)] = flux[icomp]
            else:
                raise ValueError("unknown postprocess {}".format(key))
        return point_data, cell_data, info

    def _to_single_matrix(self, Ain):
        import scipy.sparse
        # print("Ain", Ain)
        if self.pmean:
            A, B, C = Ain
        else:
            A, B = Ain
        ncells, nfaces, = self.mesh.ncells, self.mesh.nfaces
        nullV = scipy.sparse.dia_matrix((np.zeros(nfaces), 0), shape=(nfaces, nfaces))
        nullP = scipy.sparse.dia_matrix((np.zeros(ncells), 0), shape=(ncells, ncells))
        if self.ncomp==2:
            A1 = scipy.sparse.hstack([A, nullV])
            A2 = scipy.sparse.hstack([nullV, A])
            AV = scipy.sparse.vstack([A1, A2])
        else:
            A1 = scipy.sparse.hstack([A, nullV, nullV])
            A2 = scipy.sparse.hstack([nullV, A, nullV])
            A3 = scipy.sparse.hstack([nullV, nullV, A])
            AV = scipy.sparse.vstack([A1, A2, A3])
        A1 = scipy.sparse.hstack([AV, -B.T])
        A2 = scipy.sparse.hstack([B, nullP])
        Aall = scipy.sparse.vstack([A1, A2])

        if self.pmean:
            rows = np.zeros(nfaces, dtype=int)
            cols = np.arange(0, nfaces)
            nullV = sparse.coo_matrix((np.zeros(nfaces), (rows, cols)), shape=(1, nfaces)).tocsr()
            if self.ncomp == 2:
                CL = scipy.sparse.hstack([nullV, nullV, C])
            else:
                CL = scipy.sparse.hstack([nullV, nullV, nullV, C])
            Abig = scipy.sparse.hstack([Aall,CL.T])
            nullL = scipy.sparse.dia_matrix((np.zeros(1), 0), shape=(1, 1))
            Cbig = scipy.sparse.hstack([CL,nullL])
            Aall = scipy.sparse.vstack([Abig, Cbig])

        return Aall.tocsr()

    def linearSolver(self, Ain, bin, u=None, solver='umf'):
        if solver == 'umf':
            Aall = self._to_single_matrix(Ain)
            u =  splinalg.spsolve(Aall, bin, permc_spec='COLAMD')
            return u
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
                return u
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
                return u
        else:
            raise ValueError("unknown solve '{}'".format(solver))
