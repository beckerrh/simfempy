from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import time
import numpy as np
from simfempy import solvers
from simfempy import fems
import simfempy.tools.analyticalsolution
import scipy.sparse as sparse
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg


# =================================================================#
class Stokes(solvers.solver.Solver):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'problem' in kwargs:
            function = kwargs.pop('problem').split('_')[1]
            if function == 'Linear':
                self.solexact.append(simfempy.tools.analyticalsolution.AnalyticalSolution('0'))
            elif function == 'Quadratic':
                self.solexact.append(simfempy.tools.analyticalsolution.AnalyticalSolution('x'))
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

    def setMesh(self, mesh):
        t0 = time.time()
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
        t1 = time.time()
        self.timer['setmesh'] = t1 - t0
        self.pstart = self.ncomp*self.mesh.nfaces

    def solve(self, iter, dirname):
        return self.solveLinear()

    def computeRhs(self):
        ncomp, nfaces, ncells = self.ncomp, self.mesh.nfaces, self.mesh.ncells
        if self.pmean: nall = nfaces * ncomp + self.mesh.ncells +1
        else: nall = nfaces * ncomp + self.mesh.ncells
        b = np.zeros(nall)
        if self.solexact or self.rhs:
            x, y, z = self.femv.pointsf[:,0], self.femv.pointsf[:,1], self.femv.pointsf[:,2]
            for i in range(ncomp):
                if self.solexact:
                    bfaces = np.zeros(self.mesh.nfaces)
                    for j in range(ncomp):
                        bfaces -= self.mucell[0] * self.solexact[i].dd(j, j, x, y, z)
                    bfaces += self.solexact[ncomp].d(i, x, y, z)
                else:
                    bfaces = self.rhs[i](x, y, z)
                b[i*nfaces:(i+1)*nfaces] = self.femv.massmatrix * bfaces
        if self.solexact:
            xc, yc, zc = self.mesh.pointsc[:, 0], self.mesh.pointsc[:, 1], self.mesh.pointsc[:, 2]
            bcells = np.zeros(self.mesh.ncells)
            for i in range(ncomp):
                bcells += self.solexact[i].d(i, xc, yc, zc)
            b[self.pstart:self.pstart+ncells] = self.mesh.dV*bcells

        normals = self.mesh.normals
        for color, faces in self.mesh.bdrylabels.items():
            condition = self.bdrycond.type[color]
            if condition == "Neumann":
                scale = 1 / self.mesh.dimension
                normalsS = normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                xS = np.mean(self.mesh.points[self.mesh.faces[faces]], axis=1)
                muS = self.mucell[self.mesh.cellsOfFaces[faces, 0]]
                x1, y1, z1 = xS[:, 0], xS[:, 1], xS[:, 2]
                nx, ny, nz = normalsS[:, 0] / dS, normalsS[:, 1] / dS, normalsS[:, 2] / dS
                if self.solexact:
                    for i in range(ncomp):
                        bS = np.zeros(xS.shape[0])
                        for j in range(ncomp):
                            bS += scale * muS * self.solexact[i].d(j, x1, y1, z1) * normalsS[:, j]
                            bS += scale * muS * self.solexact[j].d(i, x1, y1, z1) * normalsS[:, j]
                        indices = i*nfaces + self.mesh.faces[faces]
                        np.add.at(b, indices.T, bS)
                else:
                    if not color in self.bdrycond.fct.keys(): continue
                    neumann = self.bdrycond.fct[color]
                    neumanns = neumann(x1, y1, z1, nx, ny, nz, lamS, muS)
                    for i in range(ncomp):
                        bS = scale * dS * neumanns[i]
                        indices = i*nfaces + self.mesh.faces[faces]
                        np.add.at(b, indices.T, bS)
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
        A = sparse.coo_matrix((matA, (rows.reshape(-1), cols.reshape(-1))), shape=(nfaces, nfaces)).tocsr()

        rowsB = np.repeat(np.arange(ncells), ncomp*nloc).reshape(ncells * nloc, ncomp)
        # print("rowsB", rowsB)

        # colsB = np.repeat( ncomp*facesOfCells, ncomp).reshape(ncells * nloc, ncomp) + np.arange(ncomp)
        colsB = np.repeat( facesOfCells, ncomp).reshape(ncells * nloc, ncomp) + nfaces*np.arange(ncomp)
        # print("colsB", colsB.reshape(ncells, nloc, ncomp)[0])

        # matB = np.zeros(ncells*nloc*ncomp, dtype=float).reshape(ncells, nloc, ncomp)
        matB = (cellgrads[:,:,:ncomp].T*dV).T
        # print("matB", matB[0])
        B = sparse.coo_matrix((matB.reshape(-1), (rowsB.reshape(-1), colsB.reshape(-1))), shape=(ncells, nfaces*ncomp)).tocsr()
        # print("B", B[0])
        return (A,B)

    def boundary(self, Aall, b, u):
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
            help = sparse.dok_matrix((nb, nfaces))
            for i in range(nb): help[i, faces[i]] = 1
            self.Asaved[key] = help.dot(A)
            helpB = np.zeros((ncomp*nfaces))
            for icomp in range(ncomp):
                helpB[icomp*nfaces + facesdirall] = 0
            helpB = sparse.dia_matrix((helpB, 0), shape=(ncomp*nfaces, ncomp*nfaces))
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
            help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            A = help.dot(A.dot(help))
            help = np.zeros((nfaces))
            help[facesdirall] = 1.0
            help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
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
        help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
        B = B.dot(help)
        # print("B", B[0])
        # print("after A", A.todense())
        # print("after B", B.todense())
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
        nfaces, ncomp  = self.mesh.nfaces, self.femv.ncomp
        # colors = [int(x) for x in data.split(',')]
        # omega = 0
        # for color in colors:
        #     omega += np.sum(linalg.norm(self.mesh.normals[self.mesh.bdrylabels[color]],axis=1))
        # print("###",self.bsaved[key].shape)
        # print("###",(self.Asaved[key] * u).shape)
        flux = []
        for icomp in range(ncomp):
            res = self.bsaved[icomp][key] - self.Asaved[key] * u[icomp*nfaces:(icomp+1)*nfaces]
            flux.append(np.sum(res[icomp*nfaces:(icomp+1)*nfaces]))
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
        info['timer'] = self.timer
        info['runinfo'] = self.runinfo
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
            rows = np.zeros(ncells, dtype=int)
            cols = np.arange(0, ncells)
            C = sparse.coo_matrix((self.mesh.dV, (rows, cols)), shape=(1, ncells)).tocsr()
            # print("C", C)
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

    def linearSolver(self, Ain, b, u=None, solver='umf'):
        if solver == 'umf':
            # print("Ain", Ain)
            Aall = self._to_single_matrix(Ain)
            # print("self.pstart", self.pstart)
            # Adense = Aall.todense()
            # print("Adense", Adense)
            # np.savetxt("Adense", Adense, fmt='%4.1f')
            # np.savetxt("b", b, fmt='%8.4f')
            # import numpy.linalg
            # print("Ainv", numpy.linalg.inv(Adense))
            # B = Adense[self.pstart:,:self.pstart]
            # BT = Adense[:self.pstart,self.pstart:]
            # print("B", B)
            # print("BT", BT)
            # print("B*BT", numpy.linalg.inv(B*BT))
            # raise ValueError
            u =  splinalg.spsolve(Aall, b, permc_spec='COLAMD')
            # np.savetxt("u", u, fmt='%8.4f')
            return u
        else:
            raise ValueError("unknown solve '{}'".format(solver))

#----------------------------------------------------------------#
def test_analytic(problem="Analytic_Sinus", geomname = "unitsquare", verbose=5):
    import simfempy.tools.comparerrors
    postproc = {}
    bdrycond =  simfempy.applications.boundaryconditions.BoundaryConditions()
    if geomname == "unitsquare":
        problem += "_2d"
        ncomp = 2
        h = [2, 1, 0.5, 0.25, 0.125]
        bdrycond.type[1000] = "Dirichlet"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Dirichlet"
        bdrycond.type[1003] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:1000,1002"
        postproc['bdrydn'] = "bdrydn:1001,1003"
    if geomname == "unitcube":
        problem += "_3d"
        ncomp = 3
        h = [2]
        bdrycond.type[100] = "Dirichlet"
        bdrycond.type[105] = "Dirichlet"
        bdrycond.type[101] = "Dirichlet"
        bdrycond.type[102] = "Dirichlet"
        bdrycond.type[103] = "Dirichlet"
        bdrycond.type[104] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:100,105"
        postproc['bdrydn'] = "bdrydn:101,102,103,104"
    compares = {}
    for fem in ['cr1']:
        compares[fem] = Stokes(problem=problem, bdrycond=bdrycond, postproc=postproc,fem=fem, ncomp=ncomp)
    comp = simfempy.tools.comparerrors.CompareErrors(compares, verbose=verbose)
    result = comp.compare(geomname=geomname, h=h)
    return result[3]['error']['L2-V']


#================================================================#
if __name__ == '__main__':
    # test_analytic(problem="Analytic_Linear", geomname = "unitcube")
    test_analytic(problem="Analytic_Quadratic")
