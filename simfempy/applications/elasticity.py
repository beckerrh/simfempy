import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from simfempy import fems
from simfempy.applications.application import Application

#=================================================================#
class Elasticity(Application):
    """
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        fem = kwargs.pop('fem', 'p1')
        if fem == 'p1':
            # self.fem = fems.femp1sys.FemP1()
            self.fem = fems.p1sys.P1sys(self.ncomp)
        elif fem == 'cr1':
            self.fem = fems.cr1sys.CR1sys(self.ncomp)
        else:
            raise ValueError("unknown fem '{}'".format(fem))
        material = kwargs.pop('material', "Acier")
        self.setParameters(*self.material2Lame(material))
        self.dirichlet = kwargs.pop('dirichlet', "trad")
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.fem.setMesh(self.mesh)
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
        # self.bdrydata = self.fem.prepareBoundary(self.problemdata.bdrycond, colorsflux)
        self.bdrydata = self.fem.prepareBoundary(colorsdirichlet, colorsflux)
        # print(f"{self.bdrydata=}")
        self.mucell = np.full(self.mesh.ncells, self.mu)
        self.lamcell = np.full(self.mesh.ncells, self.lam)
        # xc, yc, zc = self.mesh.pointsc.T
        # self.mucell = self.mufct(self.mesh.cell_labels, xc, yc, zc)
        # self.lamcell = self.lamfct(self.mesh.cell_labels, xc, yc, zc)
    def defineRhsAnalyticalSolution(self, solexact):
        def _fctu(x, y, z):
            rhs = np.zeros(shape=(self.ncomp, x.shape[0]))
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
            rhs = np.zeros(shape=(self.ncomp, x.shape[0]))
            normals = nx, ny, nz
            mu, lam = self.mu, self.lam
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhs[i] += lam * solexact[j].d(j, x, y, z) * normals[i]
                    rhs[i] += mu  * solexact[i].d(j, x, y, z) * normals[j]
                    rhs[i] += mu  * solexact[j].d(i, x, y, z) * normals[j]
            return rhs
        return _fctneumann
    def setParameters(self, mu, lam):
        self.mu, self.lam = mu, lam
        self.mufct = np.vectorize(lambda j: mu)
        self.lamfct = np.vectorize(lambda j: lam)
        if hasattr(self,'mesh'):
            self.mucell = self.mufct(self.mesh.cell_labels)
            self.lamcell = self.lamfct(self.mesh.cell_labels)
    def solve(self, iter, dirname): return self.static(iter, dirname)
    def computeRhs(self, b=None, u=None, coeff=1, coeffmass=None):
        b = self.fem.computeRhs(self.problemdata)
        return self.fem.vectorDirichlet(self.problemdata, self.dirichlet, b, u)
    def computeMatrix(self):
        A = self.fem.computeMatrixElasticity(self.mucell, self.lamcell)
        return self.fem.matrixDirichlet(self.dirichlet,A).tobsr()
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
                    data['global'][name] = self.fem.computeBdryNormalFlux(u, colors)
                elif type == types[2]:
                    data['global'][name] = self.fem.computePointValues(u, colors)
                else:
                    raise ValueError(f"unknown postprocess type '{type}' for key '{name}'\nknown types={types=}")
        return data

    def build_pyamg(self,A):
        import pyamg
        B = np.ones((A.shape[0], 1))
        config = pyamg.solver_configuration(A, verb=False)
        # ml = pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='energy')
        # ml = pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='jacobi')
        return pyamg.rootnode_solver(A, B=config['B'], smooth='energy')

    # def linearSolver(self, A, b, u=None, solver = 'umf', verbose=0):
    #     if not sparse.isspmatrix_bsr(A): raise ValueError("no bsr matrix")
    #     if solver == 'umf':
    #         return splinalg.spsolve(A, b, permc_spec='COLAMD'), 1
    #     elif solver in ['gmres','lgmres','bicgstab','cg']:
    #         M2 = splinalg.spilu(A, drop_tol=0.2, fill_factor=2)
    #         M_x = lambda x: M2.solve(x)
    #         M = splinalg.LinearOperator(A.shape, M_x)
    #         counter = tools.iterationcounter.IterationCounter(name=solver)
    #         args=""
    #         if solver == 'lgmres': args = ', inner_m=20, outer_k=4'
    #         cmd = "u = splinalg.{}(A, b, M=M, callback=counter {})".format(solver,args)
    #         exec(cmd)
    #         return u, counter.niter
    #     elif solver == 'pyamg':
    #         import pyamg
    #         config = pyamg.solver_configuration(A, verb=False)
    #         # ml = pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='energy')
    #         # ml = pyamg.smoothed_aggregation_solver(A, B=config['B'], smooth='jacobi')
    #         ml = pyamg.rootnode_solver(A, B=config['B'], smooth='energy')
    #         # print("ml", ml)
    #         res=[]
    #         # if u is not None: print("u norm", np.linalg.norm(u))
    #         u = ml.solve(b, x0=u, tol=1e-12, residuals=res, accel='gmres')
    #         if verbose: print("pyamg {:3d} ({:7.1e})".format(len(res),res[-1]/res[0]))
    #         return u, len(res)
    #     else:
    #         raise ValueError("unknown solve '{}'".format(solver))
    #
    #     # ml = pyamg.ruge_stuben_solver(A)
    #     # B = np.ones((A.shape[0], 1))
    #     # ml = pyamg.smoothed_aggregation_solver(A, B, max_coarse=10)
    #     # res = []
    #     # # u = ml.solve(b, tol=1e-10, residuals=res)
    #     # u = pyamg.solve(A, b, tol=1e-10, residuals=res, verb=False,accel='cg')
    #     # for i, r in enumerate(res):
    #     #     print("{:2d} {:8.2e}".format(i,r))
    #     # lu = umfpack.splu(A)
    #     # u = umfpack.spsolve(A, b)

#=================================================================#
if __name__ == '__main__':
    raise NotImplementedError("Pas encore de test")
