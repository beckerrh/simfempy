import time
import numpy as np
import fempy.tools.analyticalsolution
from fempy import solvers
from fempy import fems
import scipy.sparse as sparse

#=================================================================#
class Elasticity(solvers.newtonsolver.NewtonSolver):
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
        solvers.newtonsolver.NewtonSolver.__init__(self)
        self.dirichlet = None
        self.neumann = None
        self.rhs = None
        self.solexact = None
        if 'fem' in kwargs: fem = kwargs.pop('fem')
        else: fem='p1'
        if fem == 'p1':
            self.fem = fems.femp1sys.FemP1()
        elif fem == 'cr1':
            print("cr1 not ready")
            import sys
            sys.exit(1)
            self.fem = fems.femcr1sys.FemCR1()
        else:
            raise ValueError("unknown fem '{}'".format(fem))            
        self.ncomp = 1
        if 'ncomp' in kwargs: self.ncomp = kwargs.pop('ncomp')
        self.bdrycond = kwargs.pop('bdrycond')
        if 'problemname' in kwargs:
            self.problemname = kwargs.pop('problemname')
        if 'problem' in kwargs:
            self.defineProblem(problem=kwargs.pop('problem'))
        if 'solexact' in kwargs:
            self.solexact = kwargs.pop('solexact')
        if self.solexact:
            for color, bc in self.bdrycond.type.items():
                def solexactall(x,y,z):
                    return [self.solexact[icomp](x,y,z) for icomp in range(self.ncomp)]
                if bc == "Dirichlet":
                    self.bdrycond.fct[color] = solexactall
                elif bc == "Neumann":
                    self.bdrycond.fct[color] = None
                else:
                    raise ValueError("unownd boundary condition {} for color {}".format(bc,color))
        if 'rhs' in kwargs:
            rhs = kwargs.pop('rhs')
            assert rhs is not None
            assert len(rhs == self.ncomp)
            self.rhs = []
            for i in range(self.ncomp):
                self.rhs[i] = np.vectorize(rhs[i])
        if 'postproc' in kwargs:
            self.postproc = kwargs.pop('postproc')
        else:
            self.postproc=None
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
        if 'show_diff' in kwargs: self.show_diff = kwargs.pop('show_diff')
        else: self.show_diff=False
        
    def defineProblem(self, problem):
        self.problemname = problem
        problemsplit = problem.split('_')
        if problemsplit[0] != 'Analytic':
            raise ValueError("unownd problem {}".format(problem))
        function = problemsplit[1]
        self.solexact = fempy.tools.analyticalsolution.randomAnalyticalSolution(function, self.ncomp)
        
    def setMesh(self, mesh):
        t0 = time.time()
        self.mesh = mesh
        self.fem.setMesh(self.mesh, self.ncomp)
        colorsdir = []
        for color, type in self.bdrycond.type.items():
            if type == "Dirichlet": colorsdir.append(color)
        self.bdrydata = self.fem.prepareBoundary(colorsdir, self.postproc)
        self.mucell = self.mu(self.mesh.cell_labels)
        self.lamcell = self.lam(self.mesh.cell_labels)
        t1 = time.time()
        self.timer['setmesh'] = t1-t0
        
    def solve(self, iter, dirname):
        return self.solveLinear()
        
    def computeRhs(self):
        b = np.zeros(self.mesh.nnodes * self.ncomp)
        if self.solexact or self.rhs:
            x, y, z = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.points[:, 2]
            for i in range(self.ncomp):
                if self.solexact:
                    bnodes = np.zeros(self.mesh.nnodes)
                    for j in range(self.ncomp):
                        bnodes -= (self.lamcell[0]+self.mucell[0])*self.solexact[j].dd(i, j, x, y, z)
                        bnodes -= self.mucell[0]*self.solexact[i].dd(j, j, x, y, z)
                else:
                    bnodes = rhs[i](x, y, z)
                b[i::self.ncomp] = self.fem.massmatrix * bnodes
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
                for k in range(self.ncomp):
                    mat[:, i::ncomp, j::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, k], cellgrads[:, :, k]).T * dV * self.mucell).T
                    mat[:, i::ncomp, k::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, k], cellgrads[:, :, j]).T * dV * self.mucell).T
        return sparse.coo_matrix((mat.flatten(), (rows, cols)), shape=(ncomp*nnodes, ncomp*nnodes)).tocsr()
        
    def boundary(self, A, b, u):
        x, y, z = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.points[:, 2]
        nnodes, ncomp = self.mesh.nnodes, self.ncomp
        nodedirall, nodesinner, nodesdir, nodesdirflux = self.bdrydata
        self.bsaved = {}
        self.Asaved = {}
        for key, nodes in nodesdirflux.items():
            ind = np.tile(ncomp * nodes, ncomp)
            for icomp in range(ncomp):ind[icomp::ncomp] += icomp
            self.bsaved[key] = b[ind]
        for key, nodes in nodesdirflux.items():
            nb = nodes.shape[0]
            help = sparse.dok_matrix((ncomp *nb, ncomp * nnodes))
            for icomp in range(ncomp):
                for i in range(nb): help[icomp + ncomp * i, icomp + ncomp * nodes[i]] = 1
            self.Asaved[key] = help.dot(A)
        if self.method == 'trad':
            for color, nodes in nodesdir.items():
                dirichlet = self.bdrycond.fct[color]
                dirichlets = dirichlet(x[nodes], y[nodes], z[nodes])
                for icomp in range(ncomp):
                    # b[icomp + ncomp * nodes] = dirichlets[icomp]
                    b[icomp + ncomp * nodes] = self.solexact[icomp](x[nodes], y[nodes], z[nodes])
                    u[icomp + ncomp * nodes] = b[icomp + ncomp * nodes]
            indin = np.tile(ncomp * nodesinner, ncomp)
            for icomp in range(ncomp): indin[icomp::ncomp] += icomp
            inddir = np.tile(ncomp * nodedirall, ncomp)
            for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
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
            raise ValueError("not written !!")
        return A, b, u

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
        for icomp, postproc in enumerate(self.postproc):
            for key, val in postproc.items():
                type,data = val.split(":")
                if type == "bdrymean":
                    info['postproc']["{}_{:02d}".format(key,icomp)] = self.fem.computeBdryMean(u, key, data, icomp)
                elif type == "bdrydn":
                    info['postproc']["{}_{:02d}".format(key,icomp)] = self.fem.computeBdryDn(u, key, data, icomp)
                else:
                    raise ValueError("unknown postprocess {}".format(key))
            if self.show_diff: cell_data['diff_{:02d}'.format(icomp)] = self.diffcell[icomp]
        return point_data, cell_data, info

#=================================================================#
if __name__ == '__main__':
    print("Pas encore de test")
