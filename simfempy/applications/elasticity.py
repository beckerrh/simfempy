import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg
from simfempy import solvers
from simfempy import fems
from simfempy import tools

#=================================================================#
class Elasticity(solvers.solver.Solver):
    """
    """
    YoungPoisson = {}
    YoungPoisson["Acier"] = (210, 0.285)
    YoungPoisson["Aluminium"] = (71, 0.34)
    YoungPoisson["Verre"] = (60, 0.25)
    YoungPoisson["Beton"] = (10, 0.15)
    YoungPoisson["Caoutchouc"] = (0.2, 0.5)
    YoungPoisson["Bois"] = (7, 0.2)
    YoungPoisson["Marbre"] = (26, 0.3)

    def toLame(self, E, nu):
        return 0.5*E/(1+nu), nu*E/(1+nu)/(1-2*nu)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linearsolver = 'pyamg'
        if 'fem' in kwargs: fem = kwargs.pop('fem')
        else: fem='p1'
        if fem == 'p1':
            self.fem = fems.femp1sys.FemP1()
        elif fem == 'cr1':
            raise NotImplementedError("cr1 not ready")
            self.fem = fems.femcr1sys.FemCR1()
        else:
            raise ValueError("unknown fem '{}'".format(fem))
        if 'mu' in kwargs:
            self.mu = kwargs.pop('mu')
            self.mu = np.vectorize(self.mu)
            if 'lam' not in kwargs: raise ValueError("If mu is given, so should be lam !")
            self.lam = kwargs.pop('lam')
            self.lam = np.vectorize(self.lam)
        else:
            E, nu = self.YoungPoisson["Acier"]
            mu, lam = self.toLame(E, nu)
            self.mu = np.vectorize(lambda j: mu)
            self.lam = np.vectorize(lambda j: lam)
        if 'method' in kwargs: self.method = kwargs.pop('method')
        else: self.method="trad"

    def setMesh(self, mesh):
        self.mesh = mesh
        self.fem.setMesh(self.mesh, self.ncomp)
        colorsdir = []
        for color, type in self.bdrycond.type.items():
            if type == "Dirichlet": colorsdir.append(color)
        self.bdrydata = self.fem.prepareBoundary(colorsdir, self.postproc)
        self.mucell = self.mu(self.mesh.cell_labels)
        self.lamcell = self.lam(self.mesh.cell_labels)

    def solve(self, iter, dirname):
        return self.solveLinear()

    def computeRhs(self):
        ncomp = self.ncomp
        b = np.zeros(self.mesh.nnodes * self.ncomp)
        if self.solexact or self.rhs:
            x, y, z = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.points[:, 2]
            for i in range(ncomp):
                if self.solexact:
                    bnodes = np.zeros(self.mesh.nnodes)
                    for j in range(ncomp):
                        bnodes -= (self.lamcell[0]+self.mucell[0])*self.solexact[j].dd(i, j, x, y, z)
                        bnodes -= self.mucell[0]*self.solexact[i].dd(j, j, x, y, z)
                else:
                    bnodes = self.rhs[i](x, y, z)
                b[i::self.ncomp] = self.fem.massmatrix * bnodes
        normals = self.mesh.normals
        for color, faces in self.mesh.bdrylabels.items():
            condition = self.bdrycond.type[color]
            if condition == "Neumann":
                scale = 1 / self.mesh.dimension
                normalsS = normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                xS = np.mean(self.mesh.points[self.mesh.faces[faces]], axis=1)
                lamS = self.lamcell[self.mesh.cellsOfFaces[faces, 0]]
                muS = self.mucell[self.mesh.cellsOfFaces[faces, 0]]
                x1, y1, z1 = xS[:, 0], xS[:, 1], xS[:, 2]
                nx, ny, nz = normalsS[:, 0] / dS, normalsS[:, 1] / dS, normalsS[:, 2] / dS
                if self.solexact:
                    for i in range(self.ncomp):
                        bS = np.zeros(xS.shape[0])
                        for j in range(self.ncomp):
                            bS += scale * lamS * self.solexact[j].d(j, x1, y1, z1) * normalsS[:, i]
                            bS += scale * muS * self.solexact[i].d(j, x1, y1, z1) * normalsS[:, j]
                            bS += scale * muS * self.solexact[j].d(i, x1, y1, z1) * normalsS[:, j]
                        indices = i + self.ncomp * self.mesh.faces[faces]
                        np.add.at(b, indices.T, bS)
                else:
                    if not color in self.bdrycond.fct.keys(): continue
                    neumann = self.bdrycond.fct[color]
                    neumanns = neumann(x1, y1, z1, nx, ny, nz, lamS, muS)
                    for i in range(ncomp):
                        bS = scale*dS*neumanns[i]
                        indices = i + self.ncomp * self.mesh.faces[faces]
                        np.add.at(b, indices.T, bS)
                # print("bS.shape", bS.shape)
                # print("indices.shape", indices.shape)

        # from ..meshes import plotmesh
        # plotmesh.meshWithData(self.mesh, point_data={"b_{:1d}".format(i):b[i::self.ncomp] for i in range(self.ncomp)})
        return b

    def matrix(self):
        nnodes, ncells, ncomp, dV = self.mesh.nnodes, self.mesh.ncells, self.ncomp, self.mesh.dV
        nloc, rows, cols, cellgrads = self.fem.nloc, self.fem.rows, self.fem.cols, self.fem.cellgrads
        mat = np.zeros(shape=rows.shape, dtype=float).reshape(ncells, ncomp * nloc, ncomp * nloc)
        for i in range(ncomp):
            for j in range(self.ncomp):
                mat[:, i::ncomp, j::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, i], cellgrads[:, :, j]).T * dV * self.lamcell).T
                mat[:, i::ncomp, j::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, j], cellgrads[:, :, i]).T * dV * self.mucell).T
                mat[:, i::ncomp, i::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, j], cellgrads[:, :, j]).T * dV * self.mucell).T
        A = sparse.coo_matrix((mat.flatten(), (rows, cols)), shape=(ncomp*nnodes, ncomp*nnodes)).tocsr()
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
            for icomp in range(ncomp):ind[icomp::ncomp] += icomp
            self.bsaved[key] = b[ind]
        for key, nodes in nodesdirflux.items():
            nb = nodes.shape[0]
            help = sparse.dok_matrix((ncomp *nb, ncomp * nnodes))
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
            b[indin] -= A[indin, :][:,inddir] * b[inddir]
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
        flux  = []
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
        # info['timer'] = self.timer
        # info['runinfo'] = self.runinfo
        info['postproc'] = {}
        for key, val in self.postproc.items():
            type,data = val.split(":")
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

    def linearSolver(self, A, b, u=None, solver = 'umf'):
        # print("A is symmetric ? ", is_symmetric(A))
        if solver == 'umf':
            return splinalg.spsolve(A, b, permc_spec='COLAMD')
        # elif solver == 'scipy-umf_mmd':
        #     return splinalg.spsolve(A, b, permc_spec='MMD_ATA')
        elif solver in ['gmres','lgmres','bicgstab','cg']:
            M2 = splinalg.spilu(A, drop_tol=0.2, fill_factor=2)
            M_x = lambda x: M2.solve(x)
            M = splinalg.LinearOperator(A.shape, M_x)
            counter = tools.iterationcounter.IterationCounter(name=solver)
            args=""
            if solver == 'lgmres': args = ', inner_m=20, outer_k=4'
            cmd = "u = splinalg.{}(A, b, M=M, callback=counter {})".format(solver,args)
            exec(cmd)
            return u
        elif solver == 'pyamg':
            import pyamg
            config = pyamg.solver_configuration(A, verb=False)
            # ml = pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='energy')
            # ml = pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='jacobi')
            ml = pyamg.rootnode_solver(A, B=config['B'], smooth='energy')
            # print("ml", ml)
            res=[]
            # if u is not None: print("u norm", np.linalg.norm(u))
            u = ml.solve(b, x0=u, tol=1e-12, residuals=res, accel='gmres')
            print("pyamg {:3d} ({:7.1e})".format(len(res),res[-1]/res[0]))
            return u
        else:
            raise ValueError("unknown solve '{}'".format(solver))

        # ml = pyamg.ruge_stuben_solver(A)
        # B = np.ones((A.shape[0], 1))
        # ml = pyamg.smoothed_aggregation_solver(A, B, max_coarse=10)
        # res = []
        # # u = ml.solve(b, tol=1e-10, residuals=res)
        # u = pyamg.solve(A, b, tol=1e-10, residuals=res, verb=False,accel='cg')
        # for i, r in enumerate(res):
        #     print("{:2d} {:8.2e}".format(i,r))
        # lu = umfpack.splu(A)
        # u = umfpack.spsolve(A, b)

#=================================================================#
if __name__ == '__main__':
    print("Pas encore de test")
