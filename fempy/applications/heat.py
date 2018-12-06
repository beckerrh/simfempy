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
            self.rhocp = np.vectorize(lambda x,y: 1234.56)
        if 'kheat' in kwargs:
            self.kheat = np.vectorize(kwargs.pop('kheat'))
        else:
            self.kheat = np.vectorize(lambda x,y: 0.123)
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
        nnodes, ncells, xc, yc = self.mesh.nnodes, self.mesh.ncells, self.mesh.centersx, self.mesh.centersy
        self.fem.setMesh(self.mesh)
        self.massmatrix = self.fem.massMatrix()

        colorsdir = []
        self.nodedirall = np.empty(shape=(0), dtype=int)
        for color, type in self.bdrycond.type.items():
            if type == "Dirichlet": colorsdir.append(color)
        self.nodesdir={}
        for color in colorsdir:
            edgesdir = self.mesh.bdrylabels[color]
            self.nodesdir[color] = np.unique(self.mesh.edges[edgesdir].flat[:])
            self.nodedirall = np.unique(np.union1d(self.nodedirall, self.nodesdir[color]))
        # print("colorsdir", colorsdir)
        # print("nodesdir", self.nodesdir)
        self.bsaved={}
        self.Asaved={}

        self.kheatcell = np.zeros(ncells)
        self.kheatcell = self.kheat(xc, yc)
        self.rhocpcell = np.zeros(ncells)
        self.rhocpcell = self.rhocp(xc, yc)
        # print("self.kheatcell", self.kheatcell)

    def solve(self, iter, dirname):
        return self.solveLinear()
    def computeRhs(self):
        import time
        x, y, simps, xc, yc = self.mesh.x, self.mesh.y, self.mesh.triangles, self.mesh.centersx, self.mesh.centersy
        t1 = time.time()
        if self.solexact:
            # bnodes = -self.kheat(x,y)*(self.solexact.xx(x, y) + self.solexact.yy(x, y))
            bnodes = -self.solexact.xx(x, y) - self.solexact.yy(x, y)
            bnodes *= self.kheat(x,y)
        else:
            bnodes = self.rhs(x, y)
        t2 = time.time()
        b = self.massmatrix*bnodes
        t3 = time.time()
        bdryedges, normals =  self.mesh.bdryedges, self.mesh.normals
        for color, edges in self.mesh.bdrylabels.items():
            bdrycond = self.bdrycond.type[color]
            # print("Boundary condition:", bdrycond)
            if bdrycond == "Neumann":
                neumann = self.bdrycond.fct[color]
                for ie in edges:
                    iv0 =  self.mesh.edges[ie, 0]
                    iv1 =  self.mesh.edges[ie, 1]
                    sigma = 1-2*(self.mesh.cellsOfEdge[ie, 0]==-1)
                    normal = normals[ie]
                    normal *= sigma
                    ic = self.mesh.cellsOfEdge[ie,0]
                    if ic < 0: ic = self.mesh.cellsOfEdge[ie,1]
                    xe, ye = 0.5*(x[iv0]+x[iv1]), 0.5*(y[iv0]+y[iv1])
                    d = linalg.norm(normal)
                    bn = neumann(xe, ye, normal[0]/d, normal[1]/d, self.kheatcell[ic]) * d
                    b[iv0] += 0.5 * bn
                    b[iv1] += 0.5 * bn
        for color, nodes in self.nodesdir.items():
            dirichlet = self.bdrycond.fct[color]
            self.bsaved[color] = b[nodes]
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
        for color, nodes in self.nodesdir.items():
            nb = nodes.shape[0]
            help = sparse.dia_matrix((np.ones(nb),0), shape=(nb,nnodes))
            self.Asaved[color] = help.dot(Asp)
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
    def computeMean(self, color, u):
        edges = self.mesh.bdrylabels[color]
        mean, omega = 0, 0
        for ie in edges:
            normal = self.mesh.normals[ie]
            d = linalg.norm(normal)
            omega += d
            mean += d*np.mean(u[self.mesh.edges[ie, :]])
        return mean/omega
    def computeFlux(self, color, u):
        length = np.sum(linalg.norm(self.mesh.normals[self.mesh.bdrylabels[color]],axis=1))
        return np.sum(self.bsaved[color] - self.Asaved[color]*u)/length
    def postProcess(self, u):
        info = {}
        cell_data = {}
        point_data = {}
        point_data['U'] = u
        if self.solexact:
            info['error'], point_data['E'] = self.computeError(self.solexact, u)
        info['timer'] = self.timer
        info['runinfo'] = self.runinfo
        for key, val in self.postproc.items():
            info[key] = {}
            for color in val.split(','):
                if key == "mean":
                    info[key][color] = self.computeMean(int(color),u)
                elif key == "flux":
                    info[key][color] = self.computeFlux(int(color), u)
                else:
                    raise ValueError("unknown postprocess {}".format(key))
        return point_data, cell_data, info
    def computeError(self, solex, uh):
        x, y, simps, xc, yc = self.mesh.x, self.mesh.y, self.mesh.triangles, self.mesh.centersx, self.mesh.centersy
        e = solex(x, y) - uh
        errors = {}
        errors['L2'] = np.sqrt( np.dot(e, self.massmatrix*e) )
        return errors, e

