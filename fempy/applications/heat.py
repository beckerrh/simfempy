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
    def defineProblem(self, problem):
        print("problem is {}".format(problem))
        self.problem = problem
        problemsplit = problem.split('_')
        if problemsplit[0] == 'Analytic':
            if problemsplit[1] == 'Linear':
                self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('0.3 * x + 0.7 * y')
            elif problemsplit[1] == 'Quadratic':
                self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('x*x+2*y*y')
            elif problemsplit[1] == 'Hubbel':
                self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('(1-x*x)*(1-y*y)')
            elif problemsplit[1] == 'Exponential':
                self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('exp(x-0.7*y)')
            elif problemsplit[1] == 'Sinus':
                self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('sin(x+0.2*y*y)')
            else:
                raise ValueError("unknown analytic solution: {}".format(problemsplit[1]))
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
            neumannex = NeummannExact(self.solexact)
            self.rhs = RhsExact(self.solexact, self.kheat)
            for color, bc in self.bdrycond.type.items():
                if bc == "Dirichlet":
                    self.bdrycond.fct[color] = self.solexact
                elif bc == "Neumann":
                    self.bdrycond.fct[color] = neumannex
                else:
                    raise ValueError("unownd boundary condition {} for color {}".format(bc,color))
        else:
            raise ValueError("unownd problem {}".format(problem))
    def setMesh(self, mesh):
        self.mesh = mesh
        self.mesh.computeSimpOfVert(test=False)
        nnodes, ncells, xc, yc = self.mesh.nnodes, self.mesh.ncells, self.mesh.centersx, self.mesh.centersy
        self.fem.setMesh(self.mesh)
        self.massmatrix = self.fem.massMatrix()
        edgesdir = np.empty(shape=(0), dtype=int)
        for color, edges in self.mesh.bdrylabels.items():
            bdrycond = self.bdrycond.type[color]
            if bdrycond == "Dirichlet":
                edgesdir = np.union1d(edgesdir, edges)
        # print("edgesdir", edgesdir)
        self.nodesdir = np.unique(self.mesh.edges[edgesdir].flat[:])
        # print("nodesdir", self.nodesdir)
        self.kheatcell = np.zeros(ncells)
        self.kheatcell = self.kheat(xc, yc)
        self.rhocpcell = np.zeros(ncells)
        self.rhocpcell = self.rhocp(xc, yc)
        # print("self.kheatcell", self.kheatcell)

    def solve(self):
        return self.solveLinear()
    def computeRhs(self):
        dirichlet, rhs = self.dirichlet, self.rhs
        x, y, simps, xc, yc = self.mesh.x, self.mesh.y, self.mesh.triangles, self.mesh.centersx, self.mesh.centersy
        bnodes = self.rhs(x, y)
        b = self.massmatrix*bnodes
        bdryedges, normals =  self.mesh.bdryedges, self.mesh.normals
        for color, edges in self.mesh.bdrylabels.items():
            bdrycond = self.bdrycond.type[color]
            # print("Boundary condition:", bdrycond)
            if bdrycond == "Neumann":
                neumann = self.bdrycond.fct[color]
                for ie in edges:
                    iv0 =  self.mesh.edges[ie, 0]
                    iv1 =  self.mesh.edges[ie, 1]
                    normal = normals[ie]
                    ic = self.mesh.cellsOfEdge[ie,0]
                    if ic < 0: ic = self.mesh.cellsOfEdge[ie,1]
                    xe, ye = 0.5*(x[iv0]+x[iv1]), 0.5*(y[iv0]+y[iv1])
                    d = linalg.norm(normal)
                    bn = neumann(xe, ye, normal[0]/d, normal[1]/d, self.kheatcell[ic]) * d
                    bn *= 2*(self.mesh.cellsOfEdge[ie, 1]==-1)-1
                    # print("bn", bn, "nomal", normal)
                    b[iv0] += 0.5 * bn
                    b[iv1] += 0.5 * bn
        for color, edges in self.mesh.bdrylabels.items():
            bdrycond = self.bdrycond.type[color]
            # print("Boundary condition:", bdrycond)
            if bdrycond == "Dirichlet":
                dirichlet = self.bdrycond.fct[color]
                for ie in edges:
                    iv0 = self.mesh.edges[ie, 0]
                    iv1 = self.mesh.edges[ie, 1]
                    b[iv0] = dirichlet(x[iv0], y[iv0])
                    b[iv1] = dirichlet(x[iv1], y[iv1])
        return b
    def matrix(self):
        nnodes, ncells = self.mesh.nnodes, self.mesh.ncells
        x, y, simps, xc, yc = self.mesh.x, self.mesh.y, self.mesh.triangles, self.mesh.centersx, self.mesh.centersy
        ar, normals = self.mesh.area, self.mesh.normals
        nlocal = 9
        index = np.zeros(nlocal*ncells, dtype=int)
        jndex = np.zeros(nlocal*ncells, dtype=int)
        A = np.zeros(nlocal*ncells, dtype=np.float64)
        count = 0
        for ic in range(ncells):
            mass = self.fem.elementMassMatrix(ic)
            lap = self.fem.elementLaplaceMatrix(ic)
            for ii in range(3):
                for jj in range(3):
                    index[count+3*ii+jj] = simps[ic, ii]
                    jndex[count+3*ii+jj] = simps[ic, jj]
                    A[count + 3 * ii + jj] = self.kheatcell[ic]*lap[ii,jj]
            count += nlocal
        # print("A", A)
        Asp = sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes)).tocsr()
        uno = np.ones((nnodes))
        uno[self.nodesdir] = 0
        uno = sparse.dia_matrix((uno,0), shape = (nnodes,nnodes))
        Asp = uno.dot(Asp)
        new_diag_entries = np.zeros((nnodes))
        new_diag_entries[self.nodesdir] = 1.0
        uno = sparse.dia_matrix((new_diag_entries,0), shape = (nnodes,nnodes))
        Asp += uno
        # print("Asp", Asp)
        # print("ar", ar)
        return Asp
    def postProcess(self, u, timer, nit=True, vtkbeta=True):
        info = {}
        cell_data = {}
        point_data = {}
        point_data['U'] = u
        if self.solexact:
            info['error'], point_data['E'] = self.computeError(self.solexact, u)
        info['timer'] = self.timer
        if nit: info['nit'] = nit
        return point_data, cell_data, info
    def computeError(self, solex, uh):
        x, y, simps, xc, yc = self.mesh.x, self.mesh.y, self.mesh.triangles, self.mesh.centersx, self.mesh.centersy
        e = solex(x, y) - uh
        errors = {}
        errors['L2'] = np.sqrt( np.dot(e, self.massmatrix*e) )
        return errors, e

