import numpy as np
import scipy.linalg as linalg
import fempy.tools.analyticalsolution
import scipy.sparse as sparse

from fempy import solvers

class Heat(solvers.newtonsolver.NewtonSolver):
    """
    """
    def __init__(self, problem):
        solvers.newtonsolver.NewtonSolver.__init__(self)
        print("problem is {}".format(problem))
        self.problem = problem
        self.dirichlet = None
        self.rhs = None
        self.solexact = None
        self.defineProblem(problem=problem)

    def defineProblem(self, problem):
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
        # self.hvert = np.ravel(np.mean(self.mesh.simpOfVert, axis=1))
        self.hvert = np.zeros(nnodes)
        for i in range(nnodes):
            # print("si[i]", type(si.data[i]))
            # print("si[i]", si.data[i])
            self.hvert[i] = np.sum(ar[si.data[si.indptr[i]:si.indptr[i+1]]])/3
        # print(self.mesh.area[self.mesh.simpOfVert])
        # print("self.hvert", self.hvert)
        # print("self.hvert", np.mean(self.hvert))

    def solve(self):
        return self.solveLinear()
    def computeRhs(self):
        dirichlet, rhs = self.dirichlet, self.rhs
        nnodes, ncells = self.mesh.nnodes, self.mesh.ncells
        x, y, simps, xc, yc = self.mesh.x, self.mesh.y, self.mesh.triangles, self.mesh.centersx, self.mesh.centersy
        ar = self.mesh.area
        b = np.zeros(nnodes)
        if self.solexact:
            assert self.rhs is None
            bcells = (self.solexact(xc, yc) -self.solexact.xx(xc, yc) - self.solexact.yy(xc, yc))* ar[:]
            # bcells = self.solexact(xc, yc)*ar[:]
        elif rhs:
            assert self.solexact is None
            bcells = self.rhs(xc, yc) *  ar[:]
        else:
            raise ValueError("nor exactsolution nor rhs given")
        for i,simp in enumerate(simps):
            b[simp] += bcells[i]/3
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
            for ii in range(3):
                for jj in range(3):
                    index[count+3*ii+jj] = simps[ic, ii]
                    jndex[count+3*ii+jj] = simps[ic, jj]
                    # A[count + 3 * ii + jj] = np.dot(normals[simps[ic, jj]], normals[simps[ic, ii]])/ar[ic]/4
                    A[count + 3 * ii + jj] += ar[ic]/9
            count += nlocal
        # print("A", A)
        Asp = sparse.coo_matrix((A, (index, jndex)), shape=(nnodes, nnodes)).tocsr()
        # print("Asp", Asp)
        return Asp
    def postProcess(self, u, timer, nit=True, vtkbeta=True):
        info = {}
        cell_data = {}
        point_data = {}
        point_data['U'] = u
        if self.solexact:
            info['error'] = self.computeError(self.solexact, u)
        info['timer'] = self.timer
        if nit: info['nit'] = nit
        return point_data, cell_data, info
    def computeError(self, solex, uh):
        x, y, simps, xc, yc = self.mesh.x, self.mesh.y, self.mesh.triangles, self.mesh.centersx, self.mesh.centersy
        e = solex(x, y) - uh
        e *= e*self.hvert
        errors = {}
        errors['L2'] = np.sqrt(np.sum(e))
        return errors

