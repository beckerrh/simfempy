import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from simfempy import fems
from simfempy.applications.application import Application

#=================================================================#
class Elasticity(Application):
    """
    -div( lam*div(u) + mu*D(u)) = f
    """
    YoungPoisson = {}
    YoungPoisson["Acier"] = (210, 0.285)
    YoungPoisson["Aluminium"] = (71, 0.34)
    YoungPoisson["Verre"] = (60, 0.25)
    YoungPoisson["Beton"] = (10, 0.15)
    YoungPoisson["Caoutchouc"] = (0.2, 0.49)
    YoungPoisson["Bois"] = (7, 0.2)
    YoungPoisson["Marbre"] = (26, 0.3)

    def toLame(self, E, nu):
        return 0.5*E/(1+nu), nu*E/(1+nu)/(1-2*nu)
    def material2Lame(self, material):
        E, nu = self.YoungPoisson[material]
        return self.toLame(E, nu)
    # def setParameters(self, mu, lam):
    #     ra
    #     self.mu, self.lam = mu, lam
    #     self.mufct = np.vectorize(lambda j: mu)
    #     self.lamfct = np.vectorize(lambda j: lam)
    #     if hasattr(self,'mesh'):
    #         self.mucell = self.mufct(self.mesh.cell_labels)
    #         self.lamcell = self.lamfct(self.mesh.cell_labels)
    def __init__(self, **kwargs):
        fem = kwargs.pop('fem', 'p1')
        ncomp = kwargs['problemdata'].ncomp
        self.dirichletmethod = kwargs.pop('dirichletmethod', 'strong')
        if fem == 'p1':
            self.fem = fems.p1sys.P1sys(ncomp=ncomp)
        elif fem == 'cr1':
            self.fem = fems.cr1sys.CR1sys(ncomp=ncomp)
            self.innersides=True
        else:
            raise ValueError("unknown fem '{}'".format(fem))
        # material = kwargs.pop('material', "Acier")
        # self.setParameters(*self.material2Lame(material))
        super().__init__(**kwargs)
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.fem.setMesh(self.mesh)
        if hasattr(self,'innersides'): self.mesh.constructInnerFaces()
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
        self.bdrydata = self.fem.prepareBoundary(colorsdirichlet, colorsflux)
        # print(f"{self.bdrydata=}")
        self.setMaterialParameters(self.problemdata.params)
    def setMaterialParameters(self, params):
        name = 'material'
        if name in params.scal_cells:
            self.mucell = np.empty(self.mesh.ncells)
            self.lamcell = np.empty(self.mesh.ncells)
            for color in params.scal_cells[name]:
                material = params.scal_cells[name][color]
                mu, lam = self.material2Lame(material)
                self.mucell[self.mesh.cellsoflabel[color]] = mu
                self.lamcell[self.mesh.cellsoflabel[color]] = lam
        else:
            material = params.scal_glob.pop(name,'Acier')
            mu, lam = self.material2Lame(material)
            self.mu, self.lam = mu, lam
            self.mucell = np.full(self.mesh.ncells, mu)
            self.lamcell = np.full(self.mesh.ncells, lam)
    def defineRhsAnalyticalSolution(self, solexact):
        def _fctu(x, y, z):
            rhs = np.zeros(shape=(self.ncomp, *x.shape))
            mu, lam = self.mu, self.lam
            # print(f"{solexact[0](x,y,z)=}")
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhs[i] -= (lam+mu) * solexact[j].dd(i, j, x, y, z)
                    rhs[i] -= mu * solexact[i].dd(j, j, x, y, z)
            return rhs
        return _fctu
    def defineNeumannAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        def _fctneumann(x, y, z, nx, ny, nz):
            rhs = np.zeros(shape=(self.ncomp, *x.shape))
            normals = nx, ny, nz
            mu, lam = self.mu, self.lam
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhs[i] += lam * solexact[j].d(j, x, y, z) * normals[i]
                    rhs[i] += mu  * solexact[i].d(j, x, y, z) * normals[j]
                    rhs[i] += mu  * solexact[j].d(i, x, y, z) * normals[j]
            return rhs
        return _fctneumann
    def computeRhs(self, b=None, coeffmass=None):
        b = np.zeros(self.fem.nunknowns() * self.ncomp)
        rhs = self.problemdata.params.fct_glob.get('rhs', None)
        if rhs: self.fem.computeRhsCells(b, rhs)
        colorsneu = self.problemdata.bdrycond.colorsOfType("Neumann")
        bdrycond, bdrydata = self.problemdata.bdrycond, self.bdrydata
        self.fem.computeRhsBoundary(b, colorsneu, bdrycond.fct)
        b = self.fem.vectorBoundaryStrong(b, bdrycond.fct, bdrydata, self.dirichletmethod)
        return b
    def computeMatrix(self):
        A = self.fem.computeMatrixElasticity(self.mucell, self.lamcell)
        A = self.fem.matrixBoundaryStrong(A, self.bdrydata, self.dirichletmethod)
        return A
    def postProcess(self, u):
        data = {'point':{}, 'cell':{}, 'global':{}}
        for icomp in range(self.ncomp):
            data['point']['U_{:02d}'.format(icomp)] = self.fem.fem.tonode(u[icomp::self.ncomp])
        if self.problemdata.solexact:
            err, e = self.fem.computeErrorL2(self.problemdata.solexact, u)
            data['global']['error_L2'] = np.sum(err)
            for icomp in range(self.ncomp):
                data['point']['E_{:02d}'.format(icomp)] = self.fem.fem.tonode(e[icomp])
        if self.problemdata.postproc:
            types = ["bdry_mean", "bdry_nflux", "pointvalues", "meanvalues"]
            for name, type in self.problemdata.postproc.type.items():
                colors = self.problemdata.postproc.colors(name)
                if type == types[0]:
                    data['global'][name] = self.fem.computeBdryMean(u, colors)
                elif type == types[1]:
                    data['global'][name] = self.fem.computeBdryNormalFlux(u, colors, self.bdrydata)
                elif type == types[2]:
                    data['global'][name] = self.fem.computePointValues(u, colors)
                else:
                    raise ValueError(f"unknown postprocess type '{type}' for key '{name}'\nknown types={types=}")
        return data

    def pyamg_solver_args(self, maxiter):
        return {'cycle': 'V', 'maxiter': maxiter, 'tol': 1e-12, 'accel': 'bicgstab'}
    def build_pyamg(self,A):
        import pyamg
        B = pyamg.solver_configuration(A, verb=False)['B']
        symmetry = 'nonsymmetric'
        smoother = 'gauss_seidel_nr'
        smooth = ('energy', {'krylov': 'gmres'})
        improve_candidates = [('gauss_seidel_nr', {'sweep': 'symmetric', 'iterations': 4}), None]
        symmetry = 'symmetric'
        smooth = ('energy', {'krylov': 'gmres'})
        smoother = 'gauss_seidel'
        improve_candidates = None
        SA_build_args = {
            'max_levels': 10, 'max_coarse': 25,
            'coarse_solver': 'pinv',
            'symmetry': symmetry
        }
        strength = [('evolution', {'k': 2, 'epsilon': 10.0})]
        presmoother = (smoother, {'sweep': 'symmetric', 'iterations': 3})
        postsmoother = (smoother, {'sweep': 'symmetric', 'iterations': 3})
        return pyamg.smoothed_aggregation_solver(A, B, smooth=smooth, strength=strength, presmoother=presmoother,
                                                 postsmoother=postsmoother, improve_candidates=improve_candidates,
                                                **SA_build_args)
        return pyamg.smoothed_aggregation_solver(A, B=B, smooth='energy')
        # ml = pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='jacobi')
        # return pyamg.rootnode_solver(A, B=config['B'], smooth='energy')

#=================================================================#
if __name__ == '__main__':
    raise NotImplementedError("Pas encore de test")
