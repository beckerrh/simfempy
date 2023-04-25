import numpy as np
from simfempy.models.stokes import Stokes
from simfempy import fems, meshes, solvers
from simfempy.solvers import linalg

class NavierStokes(Stokes):
    def __format__(self, spec):
        if spec=='-':
            repr = super().__format__(spec)
            repr += f"\tconvmethod={self.convmethod}"
            return repr
        return self.__repr__()
    def __init__(self, **kwargs):
        self.linearsolver_def = {'method': 'scipy_lgmres', 'maxiter': 10, 'prec': 'Chorin', 'disp':0, 'rtol':1e-3}
        self.mode='nonlinear'
        self.convdata = fems.data.ConvectionData()
        # self.convmethod = kwargs.pop('convmethod', 'lps')
        self.convmethod = kwargs.get('convmethod', 'lps')
        self.lpsparam = kwargs.pop('lpsparam', 0.01)
        self.newtontol = kwargs.pop('newtontol', 1e-8)
        if not 'linearsolver' in kwargs: kwargs['linearsolver'] = self.linearsolver_def
        super().__init__(**kwargs)
        self.newmatrix = 0
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.Astokes = super().computeMatrix()
    def solve(self):
        sdata = solvers.newtondata.StoppingParamaters(maxiter=200, steptype='bt', nbase=1, rtol=self.newtontol)
        return self.static(mode='newton',sdata=sdata)
    def computeForm(self, u):
        # if not hasattr(self,'Astokes'): self.Astokes = super().computeMatrix()
        # d = super().matrixVector(self.Astokes,u)
        d = self.Astokes.matvec(u)
        # d = super().computeForm(u)
        v = self._split(u)[0]
        dv = self._split(d)[0]
        self.computeFormConvection(dv, v)
        # self.femv.computeFormHdivPenaly(dv, v, self.hdivpenalty)
        self.timer.add('form')
        return d
    def rhs_dynamic(self, rhs, u, Aimp, time, dt, theta):
        # print(f"{u.shape=} {rhs.shape=} {type(Aconst)=}")
        # rhs += 1 / (theta * theta * dt) * self.Mass.dot(u)
        self.Mass.dot(rhs, 1 / (theta * theta * dt), u)
        rhs += (theta - 1) / theta * Aimp.dot(u)
        # print(f"@1@{np.min(u)=} {np.max(u)=} {np.min(rhs)=} {np.max(rhs)=}")
        rhs2 = self.computeRhs()
        rhs += (1 / theta) * rhs2
        # v = self._split(u)[0]
        # dv = self._split(rhs)[0]
        # self.computeFormConvection(dv, v)
    def defect_dynamic(self, f, u):
        # y = self.computeForm(u, coeffmass=1 / (self.theta * self.dt))-f
        y = super().computeForm(u)-f
        self.Mass.dot(y, 1 / (self.theta * self.dt), u)
        v = self._split(u)[0]
        vold = self._split(self.uold)[0]
        dv = self._split(y)[0]
        self.computeFormConvection(dv, 0.5*(v+vold))
        self.timer.add('defect_dynamic')
        return y
        # return self.computeForm(u)-self.rhs + self.Mass.dot(u)/(self.theta * self.dt)
    # def dx_dynamic(self, Aconst, b, u, info):
    #     if info.bad_convergence or not hasattr(self, 'A'):
    #         # print(f"*** new Matrix")
    #         self.A = self.computeMatrix(u=u)
    #         self.newmatrix += 1
    #     u, niter = self.solvelinear(self.A, b=b, u=u)
    #     self.timer.add('dx_dynamic')
    #     return u, niter
    def computeMatrixConstant(self, coeffmass, coeffmassold=0):
        self.Astokes.A  =  self.Mass.addToStokes(coeffmass-coeffmassold, self.Astokes.A)
        return self.Astokes
        return super().computeMatrix(u, coeffmass)
    def computeMatrix(self, u=None, coeffmass=None):
        X = self.Astokes.copy()
        v = self._split(u)[0]
        theta = 1
        if hasattr(self,'uold'): theta = 0.5
        X.A += theta*self.computeMatrixConvection(v)
        # X[0] += self.femv.computeMatrixHdivPenaly(self.hdivpenalty)
        self.timer.add('matrix')
        return X
    def _compute_conv_data(self, v):
        rt = fems.rt0.RT0(mesh=self.mesh)
        self.convdata.betart = rt.interpolateCR1(v)
        self.convdata.beta = rt.toCell(self.convdata.betart)
        if self.convmethod=='supg' or self.convmethod=='lps':
            if not hasattr(self.mesh,'innerfaces'): self.mesh.constructInnerFaces()
        if self.convmethod=='supg':
            self.convdata.md = meshes.move.move_midpoints(self.mesh, self.convdata.beta, bound=1/dim)

    def computeFormConvection(self, dv, v):
        dim = self.mesh.dimension
        self._compute_conv_data(v)
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        vdir = self.femv.interpolateBoundary(colorsdirichlet, self.problemdata.bdrycond.fct).ravel()
        self.femv.massDotBoundary(dv, vdir, colors=colorsdirichlet, ncomp=self.ncomp, coeff=np.minimum(self.convdata.betart, 0))
        for icomp in range(dim):
            # self.femv.fem.computeFormConvection(dv[icomp::dim], v[icomp::dim], self.convdata, method=self.convmethod, lpsparam=self.lpsparam)
            self.femv.fem.computeFormTransportCellWise(dv[icomp::dim], v[icomp::dim], self.convdata, type='centered')
            self.femv.fem.computeFormJump(dv[icomp::dim], v[icomp::dim], self.convdata.betart)
    def computeMatrixConvection(self, v):
        # A = self.femv.fem.computeMatrixConvection(self.convdata, method=self.convmethod, lpsparam=self.lpsparam)
        if not hasattr(self.convdata,'beta'):
            self._compute_conv_data(v)
        A = self.femv.fem.computeMatrixTransportCellWise(self.convdata, type='centered')
        A += self.femv.fem.computeMatrixJump(self.convdata.betart)
        # boundary also done by self.femv.fem.computeMatrixTransportCellWise()
        if self.singleA:
            return A
        return linalg.matrix2systemdiagonal(A, self.ncomp).tocsr()
    def computeBdryNormalFluxNitsche(self, v, p, colors):
        flux = super().computeBdryNormalFluxNitsche(v,p,colors)
        if self.convdata.betart is None : return flux
        ncomp, bdryfct = self.ncomp, self.problemdata.bdrycond.fct
        vdir = self.femv.interpolateBoundary(colors, bdryfct).ravel()
        for icomp in range(ncomp):
            for i,color in enumerate(colors):
                flux[icomp,i] -= self.femv.fem.massDotBoundary(b=None, f=v[icomp::ncomp]-vdir[icomp::ncomp], colors=[color], coeff=np.minimum(self.convdata.betart, 0))
        return flux
