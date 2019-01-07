import time
import numpy as np
import fempy.tools.analyticalsolution
from fempy import solvers
from fempy import fems

#=================================================================#
class Elasticity(solvers.newtonsolver.NewtonSolver):
    """
    """
    def __init__(self, **kwargs):
        solvers.newtonsolver.NewtonSolver.__init__(self)
        self.dirichlet = None
        self.neumann = None
        self.rhs = None
        self.solexact = None
        self.bdrycond = kwargs.pop('bdrycond')
        self.kheat = None
        fem = 'p1'
        if 'fem' in kwargs: fem = kwargs.pop('fem')
        if fem == 'p1':
            self.fem = fems.femp1.FemP1()
        elif fem == 'cr1':
            self.fem = fems.femcr1.FemCR1()
        else:
            raise ValueError("unknown fem '{}'".format(fem))
        if 'mu' in kwargs:
            self.mu = np.vectorize(kwargs.pop('mu'))
        else:
            self.mu = np.vectorize(lambda i: 1234.56)
        if 'lam' in kwargs:
            self.lam = np.vectorize(kwargs.pop('lam'))
        else:
            self.lam = np.vectorize(lambda i: 0.123)
        if 'problem' in kwargs:
            self.defineProblem(problem=kwargs.pop('problem'))
        if 'rhs' in kwargs:
            rhs = kwargs.pop('rhs')
            assert rhs is not None
            self.rhs = np.vectorize(rhs)
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
        dim = self.mesh.dimension
        self.solexact = {}
        self.solexact["U"] = []
        if function == 'Linear':
            self.solexact["U"].appned(fempy.tools.analyticalsolution.AnalyticalSolution('0.3 * x + 0.7 * y'))
            self.solexact["U"].append(fempy.tools.analyticalsolution.AnalyticalSolution('0.7 * x - 0.3 * y'))
        elif function == 'Linear3d':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('0.3 * x + 0.2 * y + 0.4*z')
        else:
            raise ValueError("unknown analytic solution: {}".format(function))
        if self.solexact:
            for color, bc in self.bdrycond.type.items():
                if bc == "Dirichlet":
                    self.bdrycond.fct[color] = self.solexact
                elif bc == "Neumann":
                    self.bdrycond.fct[color] = None
                else:
                    raise ValueError("unownd boundary condition {} for color {}".format(bc,color))
    def setMesh(self, mesh):
        t0 = time.time()
        self.mesh = mesh
        self.fem.setMesh(self.mesh)
        colorsdir = []
        for color, type in self.bdrycond.type.items():
            if type == "Dirichlet": colorsdir.append(color)
        self.fem.prepareBoundary(colorsdir, self.postproc)
        self.kheatcell = self.kheat(self.mesh.cell_labels)
        self.rhocpcell = self.rhocp(self.mesh.cell_labels)
        t1 = time.time()
        self.timer['setmesh'] = t1-t0
    def solvestatic(self):
        return self.solveLinear()
    def solve(self, iter, dirname):
        return self.solveLinear()
    def computeRhs(self):
        # return self.fem.computeRhs(self.rhs, self.solexact, self.kheatcell, self.bdrycond)
        if solexact or rhs:
            x, y, z = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.points[:, 2]
            if solexact:
                bnodes = -solexact.xx(x, y, z) - solexact.yy(x, y, z) - solexact.zz(x, y, z)
                bnodes *= kheatcell[0]
            else:
                bnodes = rhs(x, y, z)
            b = self.massmatrix * bnodes
        else:
            b = np.zeros(self.mesh.nnodes)
        normals = self.mesh.normals
        for color, faces in self.mesh.bdrylabels.items():
            condition = bdrycond.type[color]
            if condition == "Neumann":
                neumann = bdrycond.fct[color]
                scale = 1 / self.mesh.dimension
                normalsS = normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                xS = np.mean(self.mesh.points[self.mesh.faces[faces]], axis=1)
                kS = kheatcell[self.mesh.cellsOfFaces[faces, 0]]
                assert (dS.shape[0] == len(faces))
                assert (xS.shape[0] == len(faces))
                assert (kS.shape[0] == len(faces))
                x1, y1, z1 = xS[:, 0], xS[:, 1], xS[:, 2]
                nx, ny, nz = normalsS[:, 0] / dS, normalsS[:, 1] / dS, normalsS[:, 2] / dS
                if solexact:
                    bS = scale * dS * kS * (
                                solexact.x(x1, y1, z1) * nx + solexact.y(x1, y1, z1) * ny + solexact.z(x1, y1, z1) * nz)
                else:
                    bS = scale * neumann(x1, y1, z1, nx, ny, nz, kS) * dS
                np.add.at(b, self.mesh.faces[faces].T, bS)
        return b

    def matrix(self):
        return self.fem.matrixDiffusion(self.kheatcell)
    def boundary(self, A, b, u):
        return self.fem.boundary(A, b, u, self.bdrycond, self.method)

    def postProcess(self, u):
        info = {}
        cell_data = {}
        point_data = {}
        point_data['U'] = self.fem.tonode(u)
        if self.solexact:
            info['error'] = {}
            info['error']['L2'], e = self.fem.computeErrorL2(self.solexact, u)
            point_data['E'] = self.fem.tonode(e)
        info['timer'] = self.timer
        info['runinfo'] = self.runinfo
        info['postproc'] = {}
        for key, val in self.postproc.items():
            type,data = val.split(":")
            if type == "mean":
                info['postproc'][key] = self.fem.computeMean(u, key, data)
            elif type == "flux":
                info['postproc'][key] = self.fem.computeFlux(u, key, data)
            else:
                raise ValueError("unknown postprocess {}".format(key))
        cell_data['k'] = self.kheatcell
        return point_data, cell_data, info

#=================================================================#
if __name__ == '__main__':
    print("Pas encore de test")
