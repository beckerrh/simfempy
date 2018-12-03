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
        if 'problem' in kwargs:
            self.defineProblem(problem=kwargs.pop('problem'))
        else:
            self.dirichlet = np.vectorize(kwargs.pop('dirichlet'))
            self.neumann = np.vectorize(kwargs.pop('neumann'))
            self.rhs = np.vectorize(kwargs.pop('rhs'))
        if 'rhocp' in kwargs:
            self.rhocp = kwargs.pop('rhocp')
        else:
            self.rhocp = 1234.56
        if 'kheat' in kwargs:
            self.kheat = kwargs.pop('kheat')
        else:
            self.kheat = 0.123
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
            self.dirichlet = self.solexact
        else:
            raise ValueError("unownd problem {}".format(problem))
    def setMesh(self, mesh):
        self.mesh = mesh
        self.mesh.computeSimpOfVert(test=False)
        nnodes = self.mesh.nnodes
        ar, si = self.mesh.area, self.mesh.simpOfVert
        self.fem.setMesh(self.mesh)
        self.massmatrix = self.fem.massMatrix()
        edgesdir = np.empty(shape=(0), dtype=int)
        for color, edges in self.mesh.bdrylabels.items():
            bdrycond = self.bdrycond[color]
            if bdrycond == "Dirichlet":
                edgesdir = np.union1d(edgesdir, edges)
        # print("edgesdir", edgesdir)
        self.nodesdir = np.unique(self.mesh.edges[edgesdir].flat[:])
        # print("nodesdir", self.nodesdir)

    def solve(self):
        return self.solveLinear()
    def computeRhs(self):
        dirichlet, rhs = self.dirichlet, self.rhs
        x, y, simps, xc, yc = self.mesh.x, self.mesh.y, self.mesh.triangles, self.mesh.centersx, self.mesh.centersy
        ar = self.mesh.area
        if self.solexact:
            assert self.rhs is None
            bnodes = self.solexact(x, y) - self.kheat*(self.solexact.xx(x, y) + self.solexact.yy(x, y))
        elif rhs:
            assert self.solexact is None
            bnodes = self.rhs(x, y)
        else:
            raise ValueError("nor exactsolution nor rhs given")
        b = self.massmatrix*bnodes
        bdryedges, normals =  self.mesh.bdryedges, self.mesh.normals
        for color, edges in self.mesh.bdrylabels.items():
            bdrycond = self.bdrycond[color]
            # print("Boundary condition:", bdrycond)
            if bdrycond == "Neumann":
                for ie in edges:
                    iv0 =  self.mesh.edges[ie, 0]
                    iv1 =  self.mesh.edges[ie, 1]
                    xe, ye = 0.5*(x[iv0]+x[iv1]), 0.5*(y[iv0]+y[iv1])
                    if self.solexact:
                        assert self.neumann is None
                        bn = self.kheat*(self.solexact.x(xe, ye)*normals[ie][0] + self.solexact.y(xe, ye)*normals[ie][1])
                    elif self.neumann:
                        assert self.solexact is None
                        bn = self.neumann(xe, ye) * linalg.norm(normals[ie])
                    else:
                        raise ValueError("nor exactsolution nor neumann given")
                    bn *= 2*(self.mesh.cellsOfEdge[ie, 1]==-1)-1
                    # print("bn", bn, normals[ie])
                    b[iv0] += 0.5 * bn
                    b[iv1] += 0.5 * bn
        for color, edges in self.mesh.bdrylabels.items():
            bdrycond = self.bdrycond[color]
            # print("Boundary condition:", bdrycond)
            if bdrycond == "Dirichlet":
                for ie in edges:
                    iv0 = self.mesh.edges[ie, 0]
                    iv1 = self.mesh.edges[ie, 1]
                    if self.solexact:
                        b[iv0] = self.solexact(x[iv0], y[iv0])
                        b[iv1] = self.solexact(x[iv1], y[iv1])
                    else:
                        b[iv0] = self.dirichlet(x[iv0], y[iv0])
                        b[iv1] = self.dirichlet(x[iv1], y[iv1])
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
                    A[count + 3 * ii + jj] = self.kheat*lap[ii,jj]
            count += nlocal
        # print("A", A)
        Asp = sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes)).tocsr()
        Asp += self.massmatrix
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

