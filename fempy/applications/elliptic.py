import time
import numpy as np
import fempy.tools.analyticalsolution
from fempy import solvers
from fempy import fems

#=================================================================#
class Elliptic(solvers.newtonsolver.NewtonSolver):
    """
    """
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
            self.fem = fems.femcr1sys.FemCR1()
        else:
            raise ValueError("unknown fem '{}'".format(fem))
        self.ncomp = 1
        if 'ncomp' in kwargs: self.ncomp = kwargs.pop('ncomp')
        self.bdrycond = kwargs.pop('bdrycond')
        assert len(self.bdrycond) == self.ncomp
        if 'problemname' in kwargs:
            self.problemname = kwargs.pop('problemname')
        if 'problem' in kwargs:
            self.defineProblem(problem=kwargs.pop('problem'))
        if 'solexact' in kwargs:
            self.solexact = kwargs.pop('solexact')
        if self.solexact:
            for icomp,bdrycond in enumerate(self.bdrycond):
                for color, bc in bdrycond.type.items():
                    if bc == "Dirichlet":
                        bdrycond.fct[color] = self.solexact[icomp]
                    elif bc == "Neumann":
                        bdrycond.fct[color] = None
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
            assert len(self.postproc) == self.ncomp
        else:
            self.postproc=None
        if 'diff' in kwargs:
            self.diff = kwargs.pop('diff')
            assert len(self.diff) == self.ncomp
            for i in range(self.ncomp):
                self.diff[i] = np.vectorize(self.diff[i])
        else:
            self.diff = []
            for i in range(self.ncomp):
                self.diff.append(np.vectorize(lambda j: 0.123*(i+1)))
        if 'method' in kwargs: self.method = kwargs.pop('method')
        else: self.method="trad"
        if 'show_diff' in kwargs: self.show_diff = kwargs.pop('show_diff')
        else: self.show_diff=False
        
    def defineProblem(self, problem):
        self.problem = problem
        print("self.problem",self.problem)
        problemsplit = problem.split('_')
        if problemsplit[0] != 'Analytic':
            raise ValueError("unownd problem {}".format(problem))
        function = problemsplit[1]
        self.solexact = fempy.tools.analyticalsolution.randomAnalyticalSolution(function, self.ncomp)
        
    def setMesh(self, mesh):
        t0 = time.time()
        self.mesh = mesh
        self.fem.setMesh(self.mesh, self.ncomp)
        self.bdrydata = []
        self.diffcell = []
        for icomp,bdrycond in enumerate(self.bdrycond):
            colorsdir = []
            for color, type in bdrycond.type.items():
                if type == "Dirichlet": colorsdir.append(color)
            self.bdrydata.append(self.fem.prepareBoundary(colorsdir, self.postproc[icomp]))
            self.diffcell.append(self.diff[icomp](self.mesh.cell_labels))
        t1 = time.time()
        self.timer['setmesh'] = t1-t0
        
    def solvestatic(self):
        return self.solveLinear()
        
    def solve(self, iter, dirname):
        return self.solveLinear()
        
    def computeRhs(self):
        return self.fem.computeRhs(self.rhs, self.solexact, self.diffcell, self.bdrycond)
        
    def matrix(self):
        return self.fem.matrixDiffusion(self.diffcell)
        
    def boundary(self, A, b, u):
        return self.fem.boundary(A, b, u, self.bdrycond, self.bdrydata, self.method)

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
