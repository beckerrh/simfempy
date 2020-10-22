import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg
from simfempy import solvers
from simfempy import fems
from simfempy import tools

#=================================================================#
class Elasticity(solvers.solver.Application):
    """
    """
    YoungPoisson = {}
    YoungPoisson["Acier"] = (210, 0.285)
    YoungPoisson["Aluminium"] = (71, 0.34)
    YoungPoisson["Verre"] = (60, 0.25)
    YoungPoisson["Beton"] = (10, 0.15)
    YoungPoisson["Caoutchouc"] = (0.2, 0.49)
    YoungPoisson["Bois"] = (7, 0.2)
    YoungPoisson["Marbre"] = (26, 0.3)

    def toLame(self, E, nu):
        return 0.5*E/(1+nu), nu*E/(1+nu)/(1-2*nu)

    def material2Lame(self, material):
        E, nu = self.YoungPoisson[material]
        return self.toLame(E, nu)

    def generatePoblemData(self, **kwargs):
        self.ncomp = self.mesh.dimension
        return super().generatePoblemData(**kwargs)

    def defineRhsAnalyticalSolution(self, solexact):
        def _fctu(x, y, z):
            rhs = np.zeros(shape=(self.ncomp, x.shape[0]))
            mu, lam = self.mufct(0), self.lamfct(0)
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhs[i] -= (lam+mu) * solexact[j].dd(i, j, x, y, z)
                    rhs[i] -= mu * solexact[i].dd(j, j, x, y, z)
            return rhs
        return _fctu

    def defineNeumannAnalyticalSolution(self, solexact):
        def _fctneumann(x, y, z, nx, ny, nz):
            rhs = np.zeros(shape=(self.ncomp, x.shape[0]))
            normals = nx, ny, nz
            mu, lam = self.mufct(0), self.lamfct(0)
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhs[i] += lam * solexact[j].d(j, x, y, z) * normals[i]
                    rhs[i] += mu  * solexact[i].d(j, x, y, z) * normals[j]
                    rhs[i] += mu  * solexact[j].d(i, x, y, z) * normals[j]
            return rhs
        return _fctneumann

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # if 'geometry' in kwargs:
        #     return
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
        if 'material' in kwargs: material = kwargs.pop('material')
        else: material = "Acier"
        self.setParameters(*self.material2Lame(material))
        if 'method' in kwargs: self.method = kwargs.pop('method')
        else: self.method="trad"

    def setParameters(self, mu, lam):
        self.mu, self.lam = mu, lam
        self.mufct = np.vectorize(lambda j: mu)
        self.lamfct = np.vectorize(lambda j: lam)
        if hasattr(self,'mesh'):
            self.mucell = self.mufct(self.mesh.cell_labels)
            self.lamcell = self.lamfct(self.mesh.cell_labels)

    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.fem.setMesh(self.mesh, self.ncomp)
        self.bdrydata = self.fem.prepareBoundary(self.problemdata.bdrycond, self.problemdata.postproc)
        self.mucell = self.mufct(self.mesh.cell_labels)
        self.lamcell = self.lamfct(self.mesh.cell_labels)

    def solve(self, iter, dirname):
        return self.solveLinear()

    def computeRhs(self, u=None):
        ncomp = self.ncomp
        b = np.zeros(self.mesh.nnodes * self.ncomp)
        if self.problemdata.rhs:
            x, y, z = self.mesh.points.T
            rhsall = self.problemdata.rhs(x, y, z)
            for i in range(ncomp):
                b[i::self.ncomp] = self.fem.massmatrix * rhsall[i]
        normals = self.mesh.normals
        for color, faces in self.mesh.bdrylabels.items():
            if self.problemdata.bdrycond.type[color] != "Neumann": continue
            scale = 1 / self.mesh.dimension
            normalsS = normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            xS = np.mean(self.mesh.points[self.mesh.faces[faces]], axis=1)
            x1, y1, z1 = xS[:, 0], xS[:, 1], xS[:, 2]
            nx, ny, nz = normalsS[:, 0] / dS, normalsS[:, 1] / dS, normalsS[:, 2] / dS
            if not color in self.problemdata.bdrycond.fct.keys(): continue
            neumanns = self.problemdata.bdrycond.fct[color](x1, y1, z1, nx, ny, nz)
            for i in range(ncomp):
                bS = scale * dS * neumanns[i]
                indices = i + self.ncomp * self.mesh.faces[faces]
                np.add.at(b, indices.T, bS)
        return self.vectorDirichlet(b, u)


    def matrix(self):
        nnodes, ncells, ncomp, dV = self.mesh.nnodes, self.mesh.ncells, self.ncomp, self.mesh.dV
        nloc, rows, cols, cellgrads = self.fem.nloc, self.fem.rowssys, self.fem.colssys, self.fem.cellgrads
        mat = np.zeros(shape=rows.shape, dtype=float).reshape(ncells, ncomp * nloc, ncomp * nloc)
        for i in range(ncomp):
            for j in range(self.ncomp):
                mat[:, i::ncomp, j::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, i], cellgrads[:, :, j]).T * dV * self.lamcell).T
                mat[:, i::ncomp, j::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, j], cellgrads[:, :, i]).T * dV * self.mucell).T
                mat[:, i::ncomp, i::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, j], cellgrads[:, :, j]).T * dV * self.mucell).T
        A = sparse.coo_matrix((mat.ravel(), (rows, cols)), shape=(ncomp*nnodes, ncomp*nnodes)).tocsr()
        # if self.method == "sym":
        # rows, cols = A.nonzero()
        # A[cols, rows] = A[rows, cols]
        return self.matrixDirichlet(A).tobsr()

    def matrixDirichlet(self, A):
        nnodes, ncomp = self.mesh.nnodes, self.ncomp
        nodesdir, nodedirall, nodesinner, nodesdirflux = self.bdrydata.nodesdir, self.bdrydata.nodedirall, self.bdrydata.nodesinner, self.bdrydata.nodesdirflux
        for key, nodes in nodesdirflux.items():
            nb = nodes.shape[0]
            help = sparse.dok_matrix((ncomp *nb, ncomp * nnodes))
            for icomp in range(ncomp):
                for i in range(nb): help[icomp + ncomp * i, icomp + ncomp * nodes[i]] = 1
            self.bdrydata.Asaved[key] = help.dot(A)
        indin = np.repeat(ncomp * nodesinner, ncomp)
        for icomp in range(ncomp): indin[icomp::ncomp] += icomp
        inddir = np.repeat(ncomp * nodedirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        self.bdrydata.A_inner_dir = A[indin, :][:, inddir]
        if self.method == 'trad':
            help = np.ones((ncomp * nnodes))
            help[inddir] = 0
            help = sparse.dia_matrix((help, 0), shape=(ncomp * nnodes, ncomp * nnodes))
            A = help.dot(A.dot(help))
            help = np.zeros((ncomp * nnodes))
            help[inddir] = 1.0
            help = sparse.dia_matrix((help, 0), shape=(ncomp * nnodes, ncomp * nnodes))
            A += help
        else:
            self.bdrydata.A_dir_dir = A[inddir, :][:, inddir]
            help = np.ones((ncomp * nnodes))
            help[inddir] = 0
            help = sparse.dia_matrix((help, 0), shape=(ncomp * nnodes, ncomp * nnodes))
            help2 = np.zeros((ncomp * nnodes))
            help2[inddir] = 1
            help2 = sparse.dia_matrix((help2, 0), shape=(ncomp * nnodes, ncomp * nnodes))
            A = help.dot(A.dot(help)) + help2.dot(A.dot(help2))
        return A

    def vectorDirichlet(self, b, u):
        if u is None: u = np.zeros_like(b)
        else: assert u.shape == b.shape
        x, y, z = self.mesh.points.T
        nnodes, ncomp = self.mesh.nnodes, self.ncomp
        nodesdir, nodedirall, nodesinner, nodesdirflux = self.bdrydata.nodesdir, self.bdrydata.nodedirall, self.bdrydata.nodesinner, self.bdrydata.nodesdirflux
        for key, nodes in nodesdirflux.items():
            ind = np.repeat(ncomp * nodes, ncomp)
            for icomp in range(ncomp):ind[icomp::ncomp] += icomp
            self.bdrydata.bsaved[key] = b[ind]
        indin = np.repeat(ncomp * nodesinner, ncomp)
        for icomp in range(ncomp): indin[icomp::ncomp] += icomp
        inddir = np.repeat(ncomp * nodedirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        if self.method == 'trad':
            for color, nodes in nodesdir.items():
                if color in self.problemdata.bdrycond.fct.keys():
                    dirichlets = self.problemdata.bdrycond.fct[color](x[nodes], y[nodes], z[nodes])
                    for icomp in range(ncomp):
                        b[icomp + ncomp * nodes] = dirichlets[icomp]
                        u[icomp + ncomp * nodes] = b[icomp + ncomp * nodes]
                else:
                    for icomp in range(ncomp):
                        b[icomp + ncomp * nodes] = 0
                        u[icomp + ncomp * nodes] = b[icomp + ncomp * nodes]
            b[indin] -= self.bdrydata.A_inner_dir * b[inddir]
        else:
            for color, nodes in nodesdir.items():
                if color in self.problemdata.bdrycond.fct.keys():
                    dirichlets = self.problemdata.bdrycond.fct[color](x[nodes], y[nodes], z[nodes])
                    for icomp in range(ncomp):
                        u[icomp + ncomp * nodes] = dirichlets[icomp]
                        b[icomp + ncomp * nodes] = 0
                else:
                    for icomp in range(ncomp):
                        b[icomp + ncomp * nodes] = 0
                        u[icomp + ncomp * nodes] = b[icomp + ncomp * nodes]
            b[indin] -= self.bdrydata.A_inner_dir * u[inddir]
            b[inddir] = self.bdrydata.A_dir_dir * u[inddir]
        return b, u

    def boundary(self, A, b, u=None):
        x, y, z = self.mesh.points.T
        nnodes, ncomp = self.mesh.nnodes, self.ncomp
        nodedirall, nodesinner, nodesdir, nodesdirflux = self.bdrydata
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
        if u is None: u = np.zeros_like(b)
        indin = np.repeat(ncomp * nodesinner, ncomp)
        for icomp in range(ncomp): indin[icomp::ncomp] += icomp
        inddir = np.repeat(ncomp * nodedirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        if self.method == 'trad':
            for color, nodes in nodesdir.items():
                if color in self.problemdata.bdrycond.fct.keys():
                    dirichlets = self.problemdata.bdrycond.fct[color](x[nodes], y[nodes], z[nodes])
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
                dirichlets = self.problemdata.bdrycond.fct[color](x[nodes], y[nodes], z[nodes])
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

    # def computeBdryMean(self, u, data):
    #     colors = [int(x) for x in data.split(',')]
    #     mean, omega = np.zeros(shape=(self.ncomp,len(colors))), np.zeros(len(colors))
    #     for i,color in enumerate(colors):
    #         faces = self.mesh.bdrylabels[color]
    #         normalsS = self.mesh.normals[faces]
    #         dS = linalg.norm(normalsS, axis=1)
    #         omega[i] = np.sum(dS)
    #         for icomp in range(self.ncomp):
    #             mean[icomp,i] += np.sum(dS * np.mean(u[icomp + self.ncomp * self.mesh.faces[faces]], axis=1))
    #     return mean/omega
    #
    def computeBdryDn(self, u, data):
        colors = [int(x) for x in data.split(',')]
        flux, omega = np.zeros(shape=(len(colors),self.ncomp)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            bs, As = self.bdrydata.bsaved[color], self.bdrydata.Asaved[color]
            res = bs - As * u
            for icomp in range(self.ncomp):
                flux[i, icomp] = np.sum(res[icomp::self.ncomp])
            # else:
            #     raise NotImplementedError("computeBdryDn for condition '{}'".format(bdrycond.type[color]))
        return flux
        # colors = [int(x) for x in data.split(',')]
        # flux, omega = np.zeros(shape=(self.ncomp,len(colors))), np.zeros(len(colors))
        # for i,color in enumerate(colors):
        #     bs, As = self.bdrydata.bsaved[color], self.bdrydata.Asaved[color]
        #     res = bs - As * u
        #     for icomp in range(self.ncomp):
        #         flux[icomp, i] = np.sum(res[icomp::self.ncomp])
        # return flux
        # colors = [int(x) for x in data.split(',')]
        # omega = 0
        # for color in colors:
        #     omega += np.sum(linalg.norm(self.mesh.normals[self.mesh.bdrylabels[color]],axis=1))
        # print("###",self.bsaved[key].shape)
        # print("###",(self.Asaved[key] * u).shape)
        # res = bs - As * u
        # flux  = []
        # for icomp in range(self.ncomp):
        #     flux.append(np.sum(res[icomp::self.ncomp]))
        # return flux

    def postProcess(self, u):
        info = {}
        cell_data = {}
        point_data = {}
        for icomp in range(self.ncomp):
            point_data['U_{:02d}'.format(icomp)] = self.fem.tonode(u[icomp::self.ncomp])
        if self.problemdata.solexact:
            info['error'] = {}
            err, e = self.fem.computeErrorL2(self.problemdata.solexact, u)
            info['error']['L2'] = np.sum(err)
            for icomp in range(self.ncomp):
                point_data['E_{:02d}'.format(icomp)] = self.fem.tonode(e[icomp])
        info['postproc'] = {}
        if self.problemdata.postproc:
            for key, val in self.problemdata.postproc.items():
                type,data = val.split(":")
                if type == "bdrymean":
                    info['postproc'][key] = self.fem.computeBdryMean(u, data)
                elif type == "bdrydn":
                    info['postproc'][key] = self.computeBdryDn(u, data)
                elif type == "pointvalues":
                    info['postproc'][key] = self.fem.computePointValues(u, data)
                else:
                    raise ValueError("unknown postprocess '{}' for key '{}'".format(type, key))
        return point_data, cell_data, info

    def linearSolver(self, A, b, u=None, solver = 'umf', verbose=0):
        if not sparse.isspmatrix_bsr(A): raise ValueError("no bsr matrix")
        if solver == 'umf':
            return splinalg.spsolve(A, b, permc_spec='COLAMD'), 1
        elif solver in ['gmres','lgmres','bicgstab','cg']:
            M2 = splinalg.spilu(A, drop_tol=0.2, fill_factor=2)
            M_x = lambda x: M2.solve(x)
            M = splinalg.LinearOperator(A.shape, M_x)
            counter = tools.iterationcounter.IterationCounter(name=solver)
            args=""
            if solver == 'lgmres': args = ', inner_m=20, outer_k=4'
            cmd = "u = splinalg.{}(A, b, M=M, callback=counter {})".format(solver,args)
            exec(cmd)
            return u, counter.niter
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
            if verbose: print("pyamg {:3d} ({:7.1e})".format(len(res),res[-1]/res[0]))
            return u, len(res)
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
