from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import time
import numpy as np
from fempy import solvers
from fempy import fems
import fempy.tools.analyticalsolution
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
            self.solexact.append(fempy.tools.analyticalsolution.AnalyticalSolution('0'))
        if 'fem' in kwargs:
            fem = kwargs.pop('fem')
        else:
            fem = 'cr1'
        if fem == 'p1':
            self.femv = fems.femp1sys.FemP1()
            raise NotImplementedError("cr1 not ready")
        elif fem == 'cr1':
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
        colorsdir = []
        for color, type in self.bdrycond.type.items():
            if type == "Dirichlet": colorsdir.append(color)
        self.bdrydata = self.femv.prepareBoundary(colorsdir, self.postproc)
        self.mucell = self.mu(self.mesh.cell_labels)
        t1 = time.time()
        self.timer['setmesh'] = t1 - t0

    def solve(self, iter, dirname):
        return self.solveLinear()

    def computeRhs(self):
        ncomp, nfaces = self.ncomp, self.mesh.nfaces
        b = np.zeros(nfaces * ncomp + self.mesh.ncells)
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
                b[i:nfaces * ncomp:ncomp] = self.femv.massmatrix * bfaces
        if self.solexact:
            xc, yc, zc = self.mesh.pointsc[:, 0], self.mesh.pointsc[:, 1], self.mesh.pointsc[:, 2]
            bcells = np.zeros(self.mesh.ncells)
            for i in range(ncomp):
                bcells += self.solexact[i].d(i, xc, yc, zc)
            b[nfaces * ncomp::] = self.mesh.dV*bcells

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
                        indices = i + ncomp * self.mesh.faces[faces]
                        np.add.at(b, indices.T, bS)
                else:
                    if not color in self.bdrycond.fct.keys(): continue
                    neumann = self.bdrycond.fct[color]
                    neumanns = neumann(x1, y1, z1, nx, ny, nz, lamS, muS)
                    for i in range(ncomp):
                        bS = scale * dS * neumanns[i]
                        indices = i + ncomp * self.mesh.faces[faces]
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
        print("matA.shape", matA.shape, "cols.shape", cols.shape)
        A = sparse.coo_matrix((matA, (rows.reshape(-1), cols.reshape(-1))), shape=(nfaces, nfaces)).tocsr()

        rows = np.repeat(facesOfCells, ncomp).reshape(ncells * nloc, ncomp) + np.arange(ncomp)
        raise NotImplementedError("rows.shape", rows.shape)

        self.rows = self.rows.reshape(ncells, nlocncomp).repeat(nlocncomp).reshape(ncells, nlocncomp, nlocncomp)
        self.cols = self.rows.swapaxes(1, 2)
        # self.cols = self.cols.flatten()
        # self.rows = self.rows.flatten()
        self.cols = self.cols.reshape(-1)
        self.rows = self.rows.reshape(-1)

        matB = np.zeros(shape=(ncells*nloc*ncomp), dtype=float).reshape(shape=(ncells, 1, ncomp*nloc))
        B = sparse.coo_matrix((matB, (rows, cols)), shape=(ncells, nfaces*ncomp)).tocsr()

        return (A,B)

        nnodes, ncells, ncomp, dV = self.mesh.nnodes, self.mesh.ncells, self.ncomp, self.mesh.dV
        mat = np.zeros(shape=rows.shape, dtype=float).reshape(ncells, ncomp * nloc, ncomp * nloc)
        for i in range(ncomp):
            for j in range(self.ncomp):
                mat[:, i::ncomp, j::ncomp] += (
                            np.einsum('nk,nl->nkl', cellgrads[:, :, i], cellgrads[:, :, j]).T * dV * self.lamcell).T
                mat[:, i::ncomp, j::ncomp] += (
                            np.einsum('nk,nl->nkl', cellgrads[:, :, j], cellgrads[:, :, i]).T * dV * self.mucell).T
                mat[:, i::ncomp, i::ncomp] += (
                            np.einsum('nk,nl->nkl', cellgrads[:, :, j], cellgrads[:, :, j]).T * dV * self.mucell).T
        A = sparse.coo_matrix((mat.flatten(), (rows, cols)), shape=(ncomp * nnodes, ncomp * nnodes)).tocsr()
        # if self.method == "sym":
        # rows, cols = A.nonzero()
        # A[cols, rows] = A[rows, cols]
        return A

    def boundary(self, A, b, u=None):
        x, y, z = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.points[:, 2]
        nnodes, ncomp = self.mesh.nnodes, self.ncomp
        nodedirall, nodesinner, nodesdir, nodesdirflux = self.bdrydata
        # print("nodedirall", nodedirall)
        # print("nodesinner", nodesinner)
        self.bsaved = {}
        self.Asaved = {}
        for key, nodes in nodesdirflux.items():
            ind = np.repeat(ncomp * nodes, ncomp)
            for icomp in range(ncomp): ind[icomp::ncomp] += icomp
            self.bsaved[key] = b[ind]
        for key, nodes in nodesdirflux.items():
            nb = nodes.shape[0]
            help = sparse.dok_matrix((ncomp * nb, ncomp * nnodes))
            for icomp in range(ncomp):
                for i in range(nb): help[icomp + ncomp * i, icomp + ncomp * nodes[i]] = 1
            self.Asaved[key] = help.dot(A)
        if u is None: u = np.asarray(b)
        indin = np.repeat(ncomp * nodesinner, ncomp)
        for icomp in range(ncomp): indin[icomp::ncomp] += icomp
        inddir = np.repeat(ncomp * nodedirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        if self.method == 'trad':
            for color, nodes in nodesdir.items():
                if color in self.bdrycond.fct.keys():
                    dirichlets = self.bdrycond.fct[color](x[nodes], y[nodes], z[nodes])
                    for icomp in range(ncomp):
                        b[icomp + ncomp * nodes] = dirichlets[icomp]
                        u[icomp + ncomp * nodes] = b[icomp + ncomp * nodes]
                else:
                    for icomp in range(ncomp):
                        b[icomp + ncomp * nodes] = 0
                        u[icomp + ncomp * nodes] = b[icomp + ncomp * nodes]
            b[indin] -= A[indin, :][:, inddir] * b[inddir]
            help = np.ones((ncomp * nnodes))
            help[inddir] = 0
            help = sparse.dia_matrix((help, 0), shape=(ncomp * nnodes, ncomp * nnodes))
            A = help.dot(A.dot(help))
            help = np.zeros((ncomp * nnodes))
            help[inddir] = 1.0
            help = sparse.dia_matrix((help, 0), shape=(ncomp * nnodes, ncomp * nnodes))
            A += help
        else:
            for color, nodes in nodesdir.items():
                dirichlets = self.bdrycond.fct[color](x[nodes], y[nodes], z[nodes])
                for icomp in range(ncomp):
                    u[icomp + ncomp * nodes] = dirichlets[icomp]
                    b[icomp + ncomp * nodes] = 0
            b[indin] -= A[indin, :][:, inddir] * u[inddir]
            b[inddir] = A[inddir, :][:, inddir] * u[inddir]
            help = np.ones((ncomp * nnodes))
            help[inddir] = 0
            help = sparse.dia_matrix((help, 0), shape=(ncomp * nnodes, ncomp * nnodes))
            help2 = np.zeros((ncomp * nnodes))
            help2[inddir] = 1
            help2 = sparse.dia_matrix((help2, 0), shape=(ncomp * nnodes, ncomp * nnodes))
            A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        return A.tobsr(), b, u

    def computeBdryMean(self, u, key, data):
        colors = [int(x) for x in data.split(',')]
        mean, omega = [0 for i in range(self.ncomp)], 0
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            for i in range(self.ncomp):
                mean[i] += np.sum(dS * np.mean(u[i + self.ncomp * self.mesh.faces[faces]], axis=1))
        return mean

    def computeBdryDn(self, u, key, data):
        # colors = [int(x) for x in data.split(',')]
        # omega = 0
        # for color in colors:
        #     omega += np.sum(linalg.norm(self.mesh.normals[self.mesh.bdrylabels[color]],axis=1))
        # print("###",self.bsaved[key].shape)
        # print("###",(self.Asaved[key] * u).shape)
        res = self.bsaved[key] - self.Asaved[key] * u
        flux = []
        for icomp in range(self.ncomp):
            flux.append(np.sum(res[icomp::self.ncomp]))
        return flux

    def postProcess(self, u):
        info = {}
        cell_data = {}
        point_data = {}
        for icomp in range(self.ncomp):
            point_data['U_{:02d}'.format(icomp)] = self.fem.tonode(u[icomp::self.ncomp])
        if self.solexact:
            info['error'] = {}
            err, e = self.fem.computeErrorL2(self.solexact, u)
            info['error']['L2'] = np.sum(err)
            for icomp in range(self.ncomp):
                point_data['E_{:02d}'.format(icomp)] = self.fem.tonode(e[icomp])
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

    def linearSolver(self, A, b, u=None, solver='umf'):
        # print("A is symmetric ? ", is_symmetric(A))
        if solver == 'umf':
            return splinalg.spsolve(A, b, permc_spec='COLAMD')
        # elif solver == 'scipy-umf_mmd':
        #     return splinalg.spsolve(A, b, permc_spec='MMD_ATA')
        elif solver in ['gmres', 'lgmres', 'bicgstab', 'cg']:
            M2 = splinalg.spilu(A, drop_tol=0.2, fill_factor=2)
            M_x = lambda x: M2.solve(x)
            M = splinalg.LinearOperator(A.shape, M_x)
            counter = fempy.solvers.solver.IterationCounter(name=solver)
            args = ""
            if solver == 'lgmres': args = ', inner_m=20, outer_k=4'
            cmd = "u = splinalg.{}(A, b, M=M, callback=counter {})".format(solver, args)
            exec(cmd)
            return u
        elif solver == 'pyamg':
            import pyamg
            config = pyamg.solver_configuration(A, verb=False)
            # ml = pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='energy')
            # ml = pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='jacobi')
            ml = pyamg.rootnode_solver(A, B=config['B'], smooth='energy')
            # print("ml", ml)
            res = []
            # if u is not None: print("u norm", np.linalg.norm(u))
            u = ml.solve(b, x0=u, tol=1e-12, residuals=res, accel='gmres')
            print("pyamg {:3d} ({:7.1e})".format(len(res), res[-1] / res[0]))
            return u
        else:
            raise ValueError("unknown solve '{}'".format(solver))

#----------------------------------------------------------------#
def test_analytic(problem="Analytic_Sinus", geomname = "unitsquare", verbose=5):
    import fempy.tools.comparerrors
    postproc = {}
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions()
    if geomname == "unitsquare":
        problem += "_2d"
        ncomp = 2
        h = [0.5, 0.25, 0.125, 0.06, 0.03, 0.015, 0.008]
        bdrycond.type[1000] = "Dirichlet"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Dirichlet"
        bdrycond.type[1003] = "Dirichlet"
        postproc['bdrymean'] = "bdrymean:1000,1002"
        postproc['bdrydn'] = "bdrydn:1001,1003"
    if geomname == "unitcube":
        problem += "_3d"
        ncomp = 3
        h = [2, 1, 0.5, 0.25, 0.125, 0.08]
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
    comp = fempy.tools.comparerrors.CompareErrors(compares, verbose=verbose)
    result = comp.compare(geomname=geomname, h=h)
    return result[3]['error']['L2']


#================================================================#
if __name__ == '__main__':
    test_analytic(problem="Analytic_Linear")
