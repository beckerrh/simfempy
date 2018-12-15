import time
import numpy as np
import scipy.linalg as linalg
import fempy.tools.analyticalsolution
import scipy.sparse as sparse
from fempy import solvers
from fempy import fems

#=================================================================#
class Heat(solvers.newtonsolver.NewtonSolver):
    """
    """
    def __init__(self, **kwargs):
        solvers.newtonsolver.NewtonSolver.__init__(self)
        self.fem = fems.femp1.FemP1()
        self.dirichlet = None
        self.neumann = None
        self.rhs = None
        self.solexact = None
        self.bdrycond = kwargs.pop('bdrycond')
        self.kheat = None
        if 'rhocp' in kwargs:
            self.rhocp = np.vectorize(kwargs.pop('rhocp'))
        else:
            self.rhocp = np.vectorize(lambda i: 1234.56)
        if 'kheat' in kwargs:
            self.kheat = np.vectorize(kwargs.pop('kheat'))
        else:
            self.kheat = np.vectorize(lambda i: 0.123)
        if 'problem' in kwargs:
            self.defineProblem(problem=kwargs.pop('problem'))
        else:
            self.rhs = np.vectorize(kwargs.pop('rhs'))
        if 'postproc' in kwargs:
            self.postproc = kwargs.pop('postproc')
        else:
            self.postproc={}
        if 'method' in kwargs:
            self.method = kwargs.pop('method')
        else:
            self.method="trad"
    def defineProblem(self, problem):
        self.problem = problem
        problemsplit = problem.split('_')
        if problemsplit[0] != 'Analytic':
            raise ValueError("unownd problem {}".format(problem))
        function = problemsplit[1]
        if function == 'Linear':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('0.3 * x + 0.7 * y')
        elif function == 'Linear3d':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('0.3 * x + 0.2 * y + 0.4*z')
        elif function == 'Quadratic':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('x*x+2*y*y')
        elif function == 'Quadratic3d':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('x*x+2*y*y+3*z*z')
        elif function == 'Hubbel':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('(1-x*x)*(1-y*y)')
        elif function == 'Exponential':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('exp(x-0.7*y)')
        elif function == 'Sinus':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('sin(x+0.2*y*y)')
        elif function == 'Sinus3d':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('sin(x+0.2*y*y+0.5*z)')
        else:
            raise ValueError("unknown analytic solution: {}".format(function))
        class NeummannExact():
            def __init__(self, ex):
                self.ex = ex
            def __call__(self, x, y, z, nx, ny, nz, k):
                return k*(self.ex.x(x, y, z)*nx + self.ex.y(x, y, z)*ny + self.ex.z(x, y, z)*nz)
        class RhsExact():
            def __init__(self, ex, k):
                self.ex = ex
                self.k = k
            def __call__(self, x, y, z):
                return -self.k*(self.ex.xx(x, y, z) + self.ex.yy(x, y, z) + self.ex.zz(x, y, z))
        neumannex = np.vectorize(NeummannExact(self.solexact).__call__)
        self.rhs = np.vectorize(RhsExact(self.solexact, self.kheat).__call__)
        for color, bc in self.bdrycond.type.items():
            if bc == "Dirichlet":
                self.bdrycond.fct[color] = self.solexact
            elif bc == "Neumann":
                self.bdrycond.fct[color] = neumannex
            else:
                raise ValueError("unownd boundary condition {} for color {}".format(bc,color))
    def setMesh(self, mesh):
        t0 = time.time()
        self.mesh = mesh
        self.fem.setMesh(self.mesh)
        self.massmatrix = self.fem.massMatrix()
        colorsdir = []
        self.nodedirall = np.empty(shape=(0), dtype=int)
        for color, type in self.bdrycond.type.items():
            if type == "Dirichlet": colorsdir.append(color)
        self.nodesdir={}
        for color in colorsdir:
            edgesdir = self.mesh.bdrylabels[color]
            self.nodesdir[color] = np.unique(self.mesh.faces[edgesdir].flat[:])
            self.nodedirall = np.unique(np.union1d(self.nodedirall, self.nodesdir[color]))
        self.nodesinner = np.setdiff1d(np.arange(self.mesh.nnodes, dtype=int),self.nodedirall)
        # print("colorsdir", colorsdir)
        # print("nodesdir", self.nodesdir)
        # print("self.nodesinner", self.nodesinner)
        self.bsaved={}
        self.Asaved={}
        self.nodesdirflux={}
        for key, val in self.postproc.items():
            type,data = val.split(":")
            if type != "flux": continue
            colors = [int(x) for x in data.split(',')]
            self.nodesdirflux[key] = np.empty(shape=(0), dtype=int)
            for color in colors:
                edgesdir = self.mesh.bdrylabels[color]
                self.nodesdirflux[key] = np.unique(np.union1d(self.nodesdirflux[key], np.unique(self.mesh.faces[edgesdir].flatten())))
        self.kheatcell = self.kheat(self.mesh.cell_labels)
        self.rhocpcell = self.rhocp(self.mesh.cell_labels)
        # print("self.kheatcell", self.kheatcell)
        t1 = time.time()
        self.timer['setmesh'] = t1-t0
    def solvestatic(self):
        return self.solveLinear()
    def solve(self, iter, dirname):
        return self.solveLinear()
    def computeRhs(self):
        x, y, z = self.mesh.points[:,0], self.mesh.points[:,1], self.mesh.points[:,2]
        if self.solexact:
            bnodes = -self.solexact.xx(x, y, z) - self.solexact.yy(x, y, z)- self.solexact.zz(x, y, z)
            bnodes *= self.kheat(0)
        else:
            bnodes = self.rhs(x, y, z)
        b = self.massmatrix*bnodes
        normals =  self.mesh.normals
        for color, edges in self.mesh.bdrylabels.items():
            bdrycond = self.bdrycond.type[color]
            if bdrycond == "Neumann":
                neumann = self.bdrycond.fct[color]
                scale = 1/self.mesh.dimension
                normalsS = normals[edges]
                dS = linalg.norm(normalsS,axis=1)
                xS = np.mean(self.mesh.points[self.mesh.faces[edges]], axis=1)
                kS = self.kheatcell[self.mesh.cellsOfFaces[edges,0]]
                assert(dS.shape[0] == len(edges))
                assert(xS.shape[0] == len(edges))
                assert(kS.shape[0] == len(edges))
                x1, y1, z1 = xS[:,0], xS[:,1], xS[:,2]
                nx, ny, nz = normalsS[:,0]/dS, normalsS[:,1]/dS, normalsS[:,2]/dS
                bS =  scale * neumann(x1, y1, z1, nx, ny, nz, kS)*dS
                np.add.at(b, self.mesh.faces[edges].T, bS)
        return b
    def matrix(self):
        nnodes = self.mesh.nnodes
        A = self.fem.assemble(self.kheatcell)
        return sparse.coo_matrix((A, (self.fem.rows, self.fem.cols)), shape=(nnodes, nnodes)).tocsr()
    def boundary(self, A, b):
        x, y, z = self.mesh.points[:,0], self.mesh.points[:,1], self.mesh.points[:,2]
        nnodes = self.mesh.nnodes
        for key, nodes in self.nodesdirflux.items():
            self.bsaved[key] = b[nodes]
        for color, nodes in self.nodesdir.items():
            dirichlet = self.bdrycond.fct[color]
            b[nodes] = dirichlet(x[nodes], y[nodes], z[nodes])
        for key, nodes in self.nodesdirflux.items():
            nb = nodes.shape[0]
            help = sparse.dok_matrix((nb,nnodes))
            for i in range(nb): help[i, nodes[i]] = 1
            self.Asaved[key] = help.dot(A)
        help = np.ones((nnodes))
        help[self.nodedirall] = 0
        help = sparse.dia_matrix((help, 0), shape=(nnodes, nnodes))
        # A = help.dot(A)
        b[self.nodesinner] -= A[self.nodesinner,:][:,self.nodedirall]*b[self.nodedirall]
        A = help.dot(A.dot(help))
        help = np.zeros((nnodes))
        help[self.nodedirall] = 1.0
        help = sparse.dia_matrix((help, 0), shape=(nnodes, nnodes))
        A += help
        return A,b

    def computeMean(self, u, key, data):
        colors = [int(x) for x in data.split(',')]
        mean, omega = 0, 0
        for color in colors:
            edges = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[edges]
            dS = linalg.norm(normalsS, axis=1)
            mean += np.sum(dS*np.mean(u[self.mesh.faces[edges]],axis=1))
        return mean
    def computeFlux(self, u, key, data):
        # colors = [int(x) for x in data.split(',')]
        # omega = 0
        # for color in colors:
        #     omega += np.sum(linalg.norm(self.mesh.normals[self.mesh.bdrylabels[color]],axis=1))
        flux = np.sum(self.bsaved[key] - self.Asaved[key]*u )
        return flux
    def postProcess(self, u):
        info = {}
        cell_data = {}
        point_data = {}
        point_data['U'] = u
        if self.solexact:
            info['error'], point_data['E'] = self.computeError(self.solexact, u)
        info['timer'] = self.timer
        info['runinfo'] = self.runinfo
        info['postproc'] = {}
        for key, val in self.postproc.items():
            type,data = val.split(":")
            if type == "mean":
                info['postproc'][key] = self.computeMean(u, key, data)
            elif type == "flux":
                info['postproc'][key] = self.computeFlux(u, key, data)
            else:
                raise ValueError("unknown postprocess {}".format(key))
        cell_data['k'] = self.kheatcell
        return point_data, cell_data, info
    def computeError(self, solex, uh):
        x, y, z = self.mesh.points[:,0], self.mesh.points[:,1], self.mesh.points[:,2]
        e = solex(x, y, z) - uh
        errors = {}
        errors['L2'] = np.sqrt( np.dot(e, self.massmatrix*e) )
        return errors, e

#=================================================================#
if __name__ == '__main__':
    print("Pas encore de test")
