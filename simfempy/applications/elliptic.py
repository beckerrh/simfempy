import numpy as np
from simfempy import solvers
from simfempy import fems

#=================================================================#
class Elliptic(solvers.solver.Application):
    """
    """
    def defineRhsAnalyticalSolution(self, solexact):
        def _fctu0(x, y, z, diff):
            rhs = np.zeros(x.shape[0])
            for i in range(self.mesh.dimension):
                rhs -= diff * solexact[0].dd(i, i, x, y, z)
            return rhs
        def _fctu1(x, y, z, diff):
            rhs = np.zeros(x.shape[0])
            for i in range(self.mesh.dimension):
                rhs -= diff * solexact[1].dd(i, i, x, y, z)
            return rhs
        return [_fctu0, _fctu1]

    def defineNeumannAnalyticalSolution_0(self, solexact):
        def _fctneumann(x, y, z, nx, ny, nz, diff):
            rhs = np.zeros(x.shape[0])
            normals = nx, ny, nz
            for i in range(self.mesh.dimension):
                rhs += diff * solexact[0].d(i, x, y, z) * normals[i]
            return rhs
        return _fctneumann

    def defineNeumannAnalyticalSolution_1(self, solexact):
        def _fctneumann(x, y, z, nx, ny, nz, diff):
            rhs = np.zeros(x.shape[0])
            normals = nx, ny, nz
            for i in range(self.mesh.dimension):
                rhs += diff * solexact[1].d(i, x, y, z) * normals[i]
            return rhs
        return _fctneumann

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'geometry' in kwargs:
            return
        self.linearsolver = 'pyamg'
        if 'fem' in kwargs: fem = kwargs.pop('fem')
        else: fem='p1'
        if fem == 'p1':
            self.fem = fems.femp1sys.FemP1()
        elif fem == 'cr1':
            self.fem = fems.femcr1sys.FemCR1()
        else:
            raise ValueError("unknown fem '{}'".format(fem))
        if self.problemdata.postproc:
            print("self.problemdata.postproc", self.problemdata.postproc, len(self.problemdata.postproc), self.ncomp)
            assert len(self.problemdata.postproc) == self.ncomp
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

    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.fem.setMesh(self.mesh, self.ncomp)
        self.diffcell = []
        for icomp in range(self.ncomp):
            self.diffcell.append(self.diff[icomp](self.mesh.cell_labels))
        self.bdrydata = self.fem.prepareBoundary(self.problemdata.bdrycond, self.problemdata.postproc)

    def solvestatic(self):
        return self.solveLinear()
        
    def solve(self, iter, dirname):
        return self.solveLinear()

    def computeRhs(self, u=None):
        b, u, self.bdrydata = self.fem.computeRhs(u, self.problemdata.rhs, self.diffcell, self.problemdata.bdrycond, self.method, self.bdrydata)
        return b,u

    def matrix(self):
        A, self.bdrydata = self.fem.matrixDiffusion(self.diffcell, self.problemdata.bdrycond, self.method, self.bdrydata)
        return A

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
        print("self.problemdata.postproc", self.problemdata.postproc)
        for icomp, postproc in enumerate(self.problemdata.postproc):
            for key, val in postproc.items():
                type,data = val.split(":")
                if type == "bdrymean":
                    info['postproc']["{}_{:02d}".format(key,icomp)] = self.fem.computeBdryMean(u, data, icomp)
                elif type == "bdrydn":
                    info['postproc']["{}_{:02d}".format(key,icomp)] = self.fem.computeBdryDn(u, data, self.bdrydata, self.problemdata.bdrycond, icomp)
                else:
                    raise ValueError("unknown postprocess {}".format(key))
            if self.show_diff: cell_data['diff_{:02d}'.format(icomp)] = self.diffcell[icomp]
        return point_data, cell_data, info

#=================================================================#
if __name__ == '__main__':
    print("Pas encore de test")
