import numpy as np
import scipy.sparse
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg
from simfempy import solvers
from simfempy import fems
import simfempy.tools.analyticalsolution
import simfempy.tools.iterationcounter

# =================================================================#
class Stokes(solvers.solver.Application):
    """
    """
    def generatePoblemData(self, **kwargs):
        self.ncomp = self.mesh.dimension
        return super().generatePoblemData(**kwargs)

    def defineAnalyticalSolution(self, exactsolution, random=True):
        solexact = super().defineAnalyticalSolution(exactsolution=exactsolution, random=random)
        if exactsolution == 'Linear':
            # solexact.append(simfempy.tools.analyticalsolution.AnalyticalSolution('x+y'))
            solexact.append(simfempy.tools.analyticalsolution.AnalyticalSolution3d('0'))
        elif exactsolution == 'Quadratic':
            # solexact[0] = simfempy.tools.analyticalsolution.AnalyticalSolution('x*x-2*y*x')
            # solexact[1] = simfempy.tools.analyticalsolution.AnalyticalSolution('-2*x*y+y*y')
            solexact.append(simfempy.tools.analyticalsolution.AnalyticalSolution3d('x+y'))
        else:
            raise NotImplementedError("unknown function '{}'".format(exactsolution))
        return solexact

    def defineRhsAnalyticalSolution(self, solexact):
        print("self.mu", self.mu(0))
        def _fctv(x, y, z):
            rhs = np.zeros(shape=(self.ncomp, x.shape[0]))
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhs[i] -= self.mu(0)* solexact[i].dd(j, j, x, y, z)
                rhs[i] += solexact[self.ncomp].d(i, x, y, z)
            return rhs
        def _fctp(x, y, z):
            rhs = np.zeros(x.shape[0])
            for i in range(self.ncomp):
                rhs += solexact[i].d(i, x, y, z)
            return rhs
        return _fctv, _fctp

    def defineNeumannAnalyticalSolution(self, solexact):
        def _fctneumann(x, y, z, nx, ny, nz):
            rhs = np.zeros(shape=(self.ncomp, x.shape[0]))
            normals = nx, ny, nz
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhs[i] += self.mu(0) * solexact[i].d(j, x, y, z) * normals[j]
                rhs[i] -= solexact[self.ncomp](x, y, z) * normals[i]
            return rhs
        return _fctneumann

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # if 'geometry' in kwargs:
        #     return
        self.linearsolver = 'gmres'
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
        if hasattr(self,'problemdata') and hasattr(self.problemdata,'mu'):
            self.mu = np.vectorize(self.problemdata.mu)
        else:
            self.mu = np.vectorize(lambda x: 1.0)
        if 'method' in kwargs:
            self.method = kwargs.pop('method')
        else:
            self.method = "trad"
        if 'rhsmethod' in kwargs:
            self.problemdata.rhsmethod = kwargs.pop('rhsmethod')
            if self.problemdata.rhsmethod == "rt":
                self.femrt = fems.femrt0.FemRT0()

    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.femv.setMesh(self.mesh, self.ncomp)
        self.femp.setMesh(self.mesh)
        self.pmean = self.problemdata.bdrycond.type.values() == len(self.problemdata.bdrycond.type)*"Dirichlet"
        self.bdrydata = self.femv.prepareBoundary(self.problemdata.bdrycond, self.problemdata.postproc)
        self.mucell = self.mu(self.mesh.cell_labels)
        self.pstart = self.ncomp*self.mesh.nfaces
        if self.problemdata.rhsmethod == "rt":
            self.femrt.setMesh(self.mesh)
            self.massrt = self.femrt.constructMass()

    def solve(self, iter, dirname):
        return self.solveLinear()

    def computeRhs(self, u=None):
        ncomp, nfaces, ncells = self.ncomp, self.mesh.nfaces, self.mesh.ncells
        if self.pmean: nall = nfaces * ncomp + self.mesh.ncells +1
        else: nall = nfaces * ncomp + self.mesh.ncells
        b = np.zeros(nall)
        rhsv, rhsp = self.problemdata.rhs
        xf, yf, zf = self.mesh.pointsf.T
        xc, yc, zc = self.mesh.pointsc.T
        if rhsp:
            b[self.pstart:self.pstart + ncells] = self.mesh.dV * np.array(rhsp(xc, yc, zc))
        rhsall = np.array(rhsv(xf, yf, zf))
        if self.problemdata.rhsmethod=='rt':
            normals, sigma, dV = self.mesh.normals, self.mesh.sigma, self.mesh.dV
            dS = linalg.norm(normals, axis=1)
            rhsn = np.zeros(nfaces)
            for i in range(ncomp):
                rhsn += rhsall[i] * normals[:,i] / dS
            rhsn = self.massrt*rhsn
            for i in range(ncomp):
                b[i * nfaces:(i + 1) * nfaces] = rhsn*normals[:,i]/dS
        elif self.problemdata.rhsmethod=='cr':
            for i in range(ncomp):
                b[i * nfaces:(i + 1) * nfaces] = self.femv.massmatrix * rhsall[i]
        else:
            raise ValueError("don't know rhsmethod='{}'".format(self.problemdata.rhsmethod))
        normals = self.mesh.normals
        for color, faces in self.mesh.bdrylabels.items():
            condition = self.problemdata.bdrycond.type[color]
            if condition == "Neumann":
                normalsS = normals[faces]
                dS = linalg.norm(normalsS,axis=1)
                xf, yf, zf = self.mesh.pointsf[faces].T
                nx, ny, nz = normalsS[:,0]/dS, normalsS[:,1]/dS, normalsS[:,2]/dS
                if not color in self.problemdata.bdrycond.fct.keys(): continue
                neumanns = self.problemdata.bdrycond.fct[color](xf, yf, zf, nx, ny, nz)
                for i in range(ncomp):
                    bS = dS * neumanns[i]
                    indices = i*nfaces + faces
                    b[indices] += bS
        return self.vectorDirichlet(b, u)

    def matrix(self):
        nfaces, facesOfCells = self.mesh.nfaces, self.mesh.facesOfCells
        nnodes, ncells, ncomp, dV = self.mesh.nnodes, self.mesh.ncells, self.ncomp, self.mesh.dV
        nloc, cellgrads = self.femv.nloc, self.femv.cellgrads
        matA = np.zeros(ncells*nloc*nloc).reshape(ncells, nloc, nloc)
        for i in range(ncomp):
            matA += np.einsum('nk,nl->nkl', cellgrads[:, :, i], cellgrads[:, :, i])
        matA = ( matA.T*dV*self.mucell).T.ravel()
        cols = np.tile(facesOfCells, nloc).reshape(ncells, nloc, nloc)
        rows = cols.swapaxes(1, 2)
        A = scipy.sparse.coo_matrix((matA, (rows.reshape(-1), cols.reshape(-1))), shape=(nfaces, nfaces)).tocsr()
        rowsB = np.repeat(np.arange(ncells), ncomp*nloc).reshape(ncells * nloc, ncomp)
        colsB = np.repeat( facesOfCells, ncomp).reshape(ncells * nloc, ncomp) + nfaces*np.arange(ncomp)
        matB = (cellgrads[:,:,:ncomp].T*dV).T
        B = scipy.sparse.coo_matrix((matB.reshape(-1), (rowsB.reshape(-1), colsB.reshape(-1))), shape=(ncells, nfaces*ncomp)).tocsr()
        (A,B) = self.matrixDirichlet((A,B))
        if self.pmean:
            rows = np.zeros(ncells, dtype=int)
            cols = np.arange(0, ncells)
            C = scipy.sparse.coo_matrix((dV, (rows, cols)), shape=(1, ncells)).tocsr()
            return (A,B,C)
        return (A,B)

    def vectorDirichlet(self, b, u):
        if u is None: u = np.zeros_like(b)
        else: assert u.shape == b.shape
        xf, yf, zf = self.mesh.pointsf.T
        facesdirall, facesinner, colorsdir, facesdirflux = self.bdrydata.facesdirall, self.bdrydata.facesinner, self.bdrydata.colorsdir, self.bdrydata.facesdirflux
        nfaces, ncells, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp
        self.bdrydata.bsaved = []
        for icomp in range(ncomp):
            self.bdrydata.bsaved.append({})
            for key, faces in facesdirflux.items():
                self.bdrydata.bsaved[icomp][key] = b[icomp*nfaces + faces]
        if self.method == 'trad':
            for color in colorsdir:
                faces = self.mesh.bdrylabels[color]
                dirichlet = self.problemdata.bdrycond.fct[color]
                dirs = dirichlet(xf[faces], yf[faces], zf[faces])
                # print("dirs", dirs)
                for icomp in range(ncomp):
                    b[icomp*nfaces + faces] = dirs[icomp]
                    u[icomp*nfaces + faces] = b[icomp*nfaces + faces]
            for icomp in range(ncomp):
                indin = icomp*nfaces + facesinner
                inddir = icomp*nfaces + facesdirall
                b[indin] -= self.bdrydata.A_inner_dir * b[inddir]
                b[self.pstart:self.pstart+ncells] -= self.bdrydata.B_inner_dir[icomp] * b[inddir]
        else:
            for color in colorsdir:
                faces = self.mesh.bdrylabels[color]
                dirichlet = self.problemdata.bdrycond.fct[color]
                dirs = dirichlet(xf[faces], yf[faces], zf[faces])
                for icomp in range(ncomp):
                    u[icomp*nfaces + faces] = dirs[icomp]
                    b[icomp*nfaces + faces] = 0
            for icomp in range(ncomp):
                indin = icomp*nfaces + facesinner
                inddir = icomp*nfaces + facesdirall
                b[indin] -= self.bdrydata.A_inner_dir * u[inddir]
                b[inddir] += self.bdrydata.A_dir_dir * u[inddir]
        return b, u

    def matrixDirichlet(self, Aall):
        A, B = Aall
        facesdirall, facesinner, colorsdir, facesdirflux = self.bdrydata.facesdirall, self.bdrydata.facesinner, self.bdrydata.colorsdir, self.bdrydata.facesdirflux
        nfaces, ncells, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp
        self.bdrydata.Asaved = {}
        self.bdrydata.Bsaved = {}
        for key, faces in facesdirflux.items():
            nb = faces.shape[0]
            help = scipy.sparse.dok_matrix((nb, nfaces))
            for i in range(nb): help[i, faces[i]] = 1
            self.bdrydata.Asaved[key] = help.dot(A)
            helpB = scipy.sparse.dok_matrix((ncomp*nfaces, ncomp*nb))
            for icomp in range(ncomp):
                for i in range(nb): helpB[icomp*nfaces + faces[i], icomp*nb + i] = 1
            self.bdrydata.Bsaved[key] = B.dot(helpB)
        self.bdrydata.A_inner_dir = A[facesinner, :][:, facesdirall]
        self.bdrydata.B_inner_dir = []
        for icomp in range(ncomp):
            inddir = icomp * nfaces + facesdirall
            self.bdrydata.B_inner_dir.append(B[:,:][:,inddir])
        if self.method == 'trad':
            help = np.ones((nfaces))
            help[facesdirall] = 0
            help = scipy.sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            A = help.dot(A.dot(help))
            help = np.zeros((nfaces))
            help[facesdirall] = 1.0
            help = scipy.sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            A += help
        else:
            self.bdrydata.A_dir_dir = A[facesdirall, :][:, facesdirall]
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
        return (A,B)

    def boundary(self, Aall, b, u):
        if self.pmean:
            A, B, C = Aall
        else:
            A, B = Aall
        # print("A", A.todense())
        # print("B", B.todense())
        xf, yf, zf = self.mesh.pointsf.T
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
                dirichlet = self.problemdata.bdrycond.fct[color]
                dirs = dirichlet(xf[faces], yf[faces], zf[faces])
                # print("dirs", dirs)
                for icomp in range(ncomp):
                    b[icomp*nfaces + faces] = dirs[icomp]
                    u[icomp*nfaces + faces] = b[icomp*nfaces + faces]
            for icomp in range(ncomp):
                indin = icomp*nfaces + facesinner
                inddir = icomp*nfaces + facesdirall
                b[indin] -= self.A_inner_dir * b[inddir]
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
                dirichlet = self.problemdata.bdrycond.fct[color]
                dirs = dirichlet(xf[faces], yf[faces], zf[faces])
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

    def computeBdryMean(self, u, data):
        nfaces, ncomp  = self.mesh.nfaces, self.femv.ncomp
        colors = [int(x) for x in data.split(',')]
        mean, omega = np.zeros(shape=(ncomp,len(colors))), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            for icomp in range(ncomp):
                mean[icomp,i] += np.sum(dS * u[icomp*nfaces + faces])
        return mean/omega
        # colors = [int(x) for x in data.split(',')]
        # mean, omega = [0 for i in range(self.ncomp)], 0
        # for color in colors:
        #     faces = self.mesh.bdrylabels[color]
        #     normalsS = self.mesh.normals[faces]
        #     dS = linalg.norm(normalsS, axis=1)
        #     for i in range(ncomp):
        #         mean[i] += np.sum(dS * u[i*nfaces + faces])
        # return mean

    def computeBdryDn(self, u, data, bdrydata, bdrycond):
        nfaces, ncells, ncomp, pstart  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp, self.pstart
        colors = [int(x) for x in data.split(',')]
        flux, omega = np.zeros(shape=(ncomp,len(colors))), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            if bdrycond.type[color] == "Dirichlet":
                As = bdrydata.Asaved[color]
                Bs = bdrydata.Bsaved[color]
                pcontrib = Bs.T * u[pstart: pstart + ncells]
                nb = Bs.shape[1] // ncomp
                for icomp in range(ncomp):
                    bs = bdrydata.bsaved[icomp][color]
                    res = bs - As * u[icomp * nfaces:(icomp + 1) * nfaces]
                    flux[icomp, i] = np.sum(res+pcontrib[icomp*nb:(icomp+1)*nb])
            else:
                raise NotImplementedError("computeBdryDn for condition '{}'".format(bdrycond.type[color]))
        return flux

        # flux = []
        # pcontrib = self.bdrydata.Bsaved[key].T*u[pstart: pstart+ncells]
        # nb = self.bdrydata.Bsaved[key].shape[1]//ncomp
        # for icomp in range(ncomp):
        #     res = self.bdrydata.bsaved[icomp][key] - self.bdrydata.Asaved[key] * u[icomp*nfaces:(icomp+1)*nfaces]
        #     flux.append(np.sum(res+pcontrib[icomp*nb:(icomp+1)*nb]))
        # return flux

    def computeErrorL2V(self, solex, uh):
        nfaces, ncomp  = self.mesh.nfaces, self.femv.ncomp
        xf, yf, zf = self.mesh.pointsf.T
        e = []
        err = []
        for icomp in range(self.ncomp):
            e.append(solex[icomp](xf, yf, zf) - uh[icomp*nfaces:(icomp+1)*nfaces])
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
        if self.problemdata.solexact:
            info['error'] = {}
            errv, ev = self.computeErrorL2V(self.problemdata.solexact, u[:self.pstart])
            errp, ep = self.femp.computeErrorL2(self.problemdata.solexact[-1], p)
            info['error']['L2-V'] = np.sum(errv)
            info['error']['L2-P'] = np.sum(errp)
            for icomp in range(self.ncomp):
                point_data['E_V{:02d}'.format(icomp)] = self.femv.tonode(ev[icomp])
            cell_data['E_P'] = ep
        info['postproc'] = {}
        for key, val in self.problemdata.postproc.items():
            type, data = val.split(":")
            if type == "bdrymean":
                mean = self.computeBdryMean(u, data)
                assert len(mean) == self.ncomp
                for icomp in range(self.ncomp):
                    info['postproc']["{}_{:02d}".format(key, icomp)] = mean[icomp]
            elif type == "bdrydn":
                flux = self.computeBdryDn(u, data, self.bdrydata, self.problemdata.bdrycond)
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
            return u, 1
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
            raise ValueError("unknown solve '{}'".format(solver))
