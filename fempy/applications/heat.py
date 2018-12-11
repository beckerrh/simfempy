import numpy as np
import scipy.linalg as linalg
import fempy.tools.analyticalsolution
import scipy.sparse as sparse
from fempy import solvers
from fempy import fems

class Heat(solvers.newtonsolver.NewtonSolver):
    """
    """
    def __init__(self, **kwargs):
        solvers.newtonsolver.NewtonSolver.__init__(self)
        self.fem = fems.femp12d.FemP12D()
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
        elif function == 'Quadratic':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('x*x+2*y*y')
        elif function == 'Hubbel':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('(1-x*x)*(1-y*y)')
        elif function == 'Exponential':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('exp(x-0.7*y)')
        elif function == 'Sinus':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('sin(x+0.2*y*y)')
        else:
            raise ValueError("unknown analytic solution: {}".format(function))
        class NeummannExact():
            def __init__(self, ex):
                self.ex = ex
            def __call__(self,x,y,nx,ny,k):
                return k*(self.ex.x(x, y)*nx + self.ex.y(x, y)*ny)
        class RhsExact():
            def __init__(self, ex, k):
                self.ex = ex
                self.k = k
            def __call__(self,x,y):
                return -self.k(x,y)*(self.ex.xx(x, y) + self.ex.yy(x, y))
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
        self.mesh = mesh
        self.mesh.computeSimpOfVert(test=False)
        nnodes, ncells = self.mesh.nnodes, self.mesh.ncells
        xc, yc, zc = self.mesh.pointsc[:,0], self.mesh.pointsc[:,1], self.mesh.pointsc[:,2]
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
        # print("colorsdir", colorsdir)
        # print("nodesdir", self.nodesdir)
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
                self.nodesdirflux[key] = np.unique(np.union1d(self.nodesdirflux[key], np.unique(self.mesh.faces[edgesdir].flat[:])))


        self.kheatcell = np.zeros(ncells)
        self.kheatcell = self.kheat(self.mesh.cell_labels)
        self.rhocpcell = np.zeros(ncells)
        self.rhocpcell = self.rhocp(self.mesh.cell_labels)
        # print("self.kheatcell", self.kheatcell)

    def solvestatic(self):
        return self.solveLinear()
    def solve(self, iter, dirname):
        return self.solveLinear()
    def computeRhs(self):
        import time
        x, y, z = self.mesh.points[:,0], self.mesh.points[:,1], self.mesh.points[:,2]
        t1 = time.time()
        if self.solexact:
            # bnodes = -self.kheat(x,y)*(self.solexact.xx(x, y) + self.solexact.yy(x, y))
            bnodes = -self.solexact.xx(x, y) - self.solexact.yy(x, y)
            bnodes *= self.kheat(0)
        else:
            bnodes = self.rhs(x, y)
        t2 = time.time()
        b = self.massmatrix*bnodes
        t3 = time.time()
        normals =  self.mesh.normals
        for color, edges in self.mesh.bdrylabels.items():
            bdrycond = self.bdrycond.type[color]
            # print("Boundary condition:", bdrycond)
            if bdrycond == "Neumann":
                neumann = self.bdrycond.fct[color]
                for ie in edges:
                    iv0 =  self.mesh.faces[ie, 0]
                    iv1 =  self.mesh.faces[ie, 1]
                    sigma = 1-2*(self.mesh.cellsOfFaces[ie, 0]==-1)
                    normal = normals[ie]
                    normal *= sigma
                    ic = self.mesh.cellsOfFaces[ie,0]
                    if ic < 0: ic = self.mesh.cellsOfFaces[ie,1]
                    xe, ye = 0.5*(x[iv0]+x[iv1]), 0.5*(y[iv0]+y[iv1])
                    d = linalg.norm(normal)
                    bn = neumann(xe, ye, normal[0]/d, normal[1]/d, self.kheatcell[ic]) * d
                    b[iv0] += 0.5 * bn
                    b[iv1] += 0.5 * bn
        # self.bsavedall = b[self.nodedirall]
        for key, nodes in self.nodesdirflux.items():
            self.bsaved[key] = b[nodes]
        for color, nodes in self.nodesdir.items():
            dirichlet = self.bdrycond.fct[color]
            b[nodes] = dirichlet(x[nodes], y[nodes])
        t4 = time.time()
        self.timer['rhs_fct'] = t2-t1
        self.timer['rhs_mult'] = t3-t2
        self.timer['rhs_bdry'] = t4-t3
        return b
    def matrix(self):
        import time
        nnodes, ncells, normals = self.mesh.nnodes, self.mesh.ncells, self.mesh.normals
        t1 = time.time()
        A = self.fem.assemble(self.kheatcell)
        t2 = time.time()
        Asp = sparse.coo_matrix((A, (self.fem.rows, self.fem.cols)), shape=(nnodes, nnodes)).tocsr()
        t3 = time.time()
        for key, nodes in self.nodesdirflux.items():
            nb = nodes.shape[0]
            help = sparse.dia_matrix((np.ones(nb),0), shape=(nb,nnodes))
            self.Asaved[key] = help.dot(Asp)
        # ndirs = self.nodedirall.shape[0]
        # help = sparse.dia_matrix((np.ones(ndirs), 0), shape=(ndirs, nnodes))
        # self.Asavedall = help.dot(Asp)
        help = np.ones((nnodes))
        help[self.nodedirall] = 0
        help = sparse.dia_matrix((help, 0), shape=(nnodes, nnodes))
        Asp = help.dot(Asp)
        help = np.zeros((nnodes))
        help[self.nodedirall] = 1.0
        help = sparse.dia_matrix((help, 0), shape=(nnodes, nnodes))
        Asp += help
        t4 = time.time()
        self.timer['mat_cells'] = t2-t1
        self.timer['mat_coo'] = t3-t2
        self.timer['mat_bdry'] = t4-t3
        return Asp
    def computeMean(self, u, key, data):
        colors = [int(x) for x in data.split(',')]
        mean, omega = 0, 0
        for color in colors:
            edges = self.mesh.bdrylabels[color]
            for ie in edges:
                normal = self.mesh.normals[ie]
                d = linalg.norm(normal)
                omega += d
                mean += d*np.mean(u[self.mesh.faces[ie, :]])
        return mean/omega
    def computeFlux(self, u, key, data):
        # colors = [int(x) for x in data.split(',')]
        # omega = 0
        # for color in colors:
        #     omega += np.sum(linalg.norm(self.mesh.normals[self.mesh.bdrylabels[color]],axis=1))
        flux = np.sum(self.bsaved[key] - self.Asaved[key]*u)
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
        e = solex(x, y) - uh
        errors = {}
        errors['L2'] = np.sqrt( np.dot(e, self.massmatrix*e) )
        return errors, e

