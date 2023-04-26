import copy

import numpy as np
import scipy.sparse as sparse
from simfempy import fems
from simfempy.models.model import Model
from simfempy.tools.analyticalfunction import analyticalSolution
from simfempy.solvers import linalg, saddle_point
from functools import partial

#=================================================================#
class Stokes(Model):
    """
    """
    def __format__(self, spec):
        if spec=='-':
            repr = f"{self.femv=} {self.femp=} {self.problemdata=}"
            if self.linearsolver=='spsolve': return repr + f" {self.linearsolver=}"
            ls = '@'.join([str(v) for v in self.linearsolver.values()])
            repr += f"\tlinearsolver={ls}"
            return repr
        return self.__repr__()
    def __init__(self, **kwargs):
        if not hasattr(self,'linearsolver_def'):
            self.linearsolver_def = {'method': 'scipy_lgmres', 'maxiter': 100, 'prec': 'Chorin', 'disp':0, 'rtol':1e-6}
        if not hasattr(self,'linearsolver'):
            self.linearsolver = kwargs.pop('linearsolver', self.linearsolver_def)
        self.singleA = kwargs.pop('singleA', True)
        super().__init__(**kwargs)
        # print(f"{self=}")
    def createFem(self, femparams):
        # femparams = kwargs.pop('femparams', {})
        self.hdivpenalty = femparams.pop('hdivpenalty', 0)
        self.divdivparam = femparams.pop('divdivparam', 0)
        self.dirichletmethod = femparams.pop('dirichletmethod', 'nitsche')
        if self.dirichletmethod == 'strong' and self.singleA:
            raise ValueError(f"strong Dirichlet needs system matrix {self.singleA=}")
        if self.dirichletmethod=='nitsche':
            # correct value ?!
            femparams['dirichletmethod'] = 'nitsche'
            femparams['nitscheparam'] = femparams.pop('nitscheparam', 10)
        self.femv = fems.cr1sys.CR1sys(self.ncomp, femparams)
        self.femp = fems.d0.D0()
    def new_params(self):
        self.mucell = self.compute_cell_vector_from_params('mu', self.problemdata.params)
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self._checkProblemData()
        if not self.ncomp==self.mesh.dimension: raise ValueError(f"{self.mesh.dimension=} {self.ncomp=}")
        self.femv.setMesh(self.mesh)
        self.femp.setMesh(self.mesh)
        self.pmean = not ('Neumann' in self.problemdata.bdrycond.type.values() or 'Pressure' in self.problemdata.bdrycond.type.values())
        if self.dirichletmethod=='strong':
            assert 'Navier' not in self.problemdata.bdrycond.type.values()
            colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
            colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
            self.bdrydata = self.femv.prepareBoundary(colorsdirichlet, colorsflux)
        self.new_params()
    def _zeros(self):
        nv = self.mesh.dimension*self.mesh.nfaces
        n = nv+self.mesh.ncells
        if self.pmean: n += 1
        return np.zeros(n)
    def _split(self, x):
        nv = self.mesh.dimension*self.mesh.nfaces
        ind = [nv]
        if self.pmean: ind.append(nv+self.mesh.ncells)
        # print(f"{ind=} {np.split(x, ind)=}")
        return np.split(x, ind)
    def _checkProblemData(self):
        # TODO checkProblemData() incomplete
        for col, fct in self.problemdata.bdrycond.fct.items():
            type = self.problemdata.bdrycond.type[col]
            if type == "Dirichlet":
                if len(fct) != self.mesh.dimension: raise ValueError(f"*** {type=} {len(fct)=} {self.mesh.dimension=}")
    def defineAnalyticalSolution(self, exactsolution, random=True):
        dim = self.mesh.dimension
        # print(f"defineAnalyticalSolution: {dim=} {self.ncomp=}")
        if exactsolution=="Linear":
            exactsolution = ["Linear", "Constant"]
        elif exactsolution=="Quadratic":
            exactsolution = ["Quadratic", "Linear"]
        v = analyticalSolution(exactsolution[0], dim, dim, random)
        p = analyticalSolution(exactsolution[1], dim, 1, random)
        return v,p
    def dirichletfct(self):
        solexact = self.problemdata.solexact
        v,p = solexact
        # def _solexactdirv(x, y, z):
        #     return [v[icomp](x, y, z) for icomp in range(self.ncomp)]
        def _solexactdirp(x, y, z, nx, ny, nz):
            return p(x, y, z)
        from functools import partial
        def _solexactdirv(x, y, z, icomp):
            # print(f"{icomp=}")
            return v[icomp](x, y, z)
        return [partial(_solexactdirv, icomp=icomp) for icomp in range(self.ncomp)]
        # return _solexactdirv
    def defineRhsAnalyticalSolution(self, solexact):
        v,p = solexact
        mu = self.problemdata.params.scal_glob['mu']
        def _fctrhsv(x, y, z):
            rhsv = np.zeros(shape=(self.ncomp, *x.shape))
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhsv[i] -= mu * v[i].dd(j, j, x, y, z)
                rhsv[i] += p.d(i, x, y, z)
            # print(f"{rhsv=}")
            return rhsv
        def _fctrhsp(x, y, z):
            rhsp = np.zeros(x.shape)
            for i in range(self.ncomp):
                rhsp += v[i].d(i, x, y, z)
            return rhsp
        return _fctrhsv, _fctrhsp
    def defineNeumannAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        mu = self.problemdata.params.scal_glob['mu']
        def _fctneumannv(x, y, z, nx, ny, nz, icomp):
            v, p = solexact
            rhsv = np.zeros(shape=x.shape)
            normals = nx, ny, nz
            # for i in range(self.ncomp):
            for j in range(self.ncomp):
                rhsv += mu  * v[icomp].d(j, x, y, z) * normals[j]
            rhsv -= p(x, y, z) * normals[icomp]
            return rhsv
        return [partial(_fctneumannv, icomp=icomp) for icomp in range(self.ncomp)]
    def defineNavierAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        mu = self.problemdata.params.scal_glob['mu']
        lambdaR = self.problemdata.params.scal_glob['navier']
        def _fctnaviervn(x, y, z, nx, ny, nz):
            v, p = solexact
            rhs = np.zeros(shape=x.shape)
            normals = nx, ny, nz
            # print(f"{x.shape=} {nx.shape=} {normals[0].shape=}")
            for i in range(self.ncomp):
                rhs += v[i](x, y, z) * normals[i]
            return rhs
        def _fctnaviertangent(x, y, z, nx, ny, nz, icomp):
            v, p = solexact
            rhs = np.zeros(shape=x.shape)
            # h = np.zeros(shape=(self.ncomp, x.shape[0]))
            normals = nx, ny, nz
            rhs = lambdaR*v[icomp](x, y, z)
            for j in range(self.ncomp):
                rhs += mu*v[icomp].d(j, x, y, z) * normals[j]
            return rhs
        return {'vn':_fctnaviervn, 'g':[partial(_fctnaviertangent, icomp=icomp) for icomp in range(self.ncomp)]}
    def definePressureAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        mu = self.problemdata.params.scal_glob['mu']
        lambdaR = self.problemdata.params.scal_glob['navier']
        def _fctpressure(x, y, z, nx, ny, nz):
            v, p = solexact
            # rhs = np.zeros(shape=x.shape)
            normals = nx, ny, nz
            # print(f"{x.shape=} {nx.shape=} {normals[0].shape=}")
            rhs = 1.0*p(x,y,z)
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhs -= mu*v[j].d(i, x, y, z) * normals[i]* normals[j]
            return rhs
        def _fctpressurevtang(x, y, z, nx, ny, nz, icomp):
            v, p = solexact
            return v[icomp](x,y,z)
        return {'p':_fctpressure, 'v':[partial(_fctpressurevtang, icomp=icomp) for icomp in range(self.ncomp)]}
    def sol_to_data(self, u, single_vector=True):
        if self.pmean: v, p, lam = self._split(u)
        else: v, p = self._split(u)
        data = {'point':{}, 'cell':{}, 'global':{}}
        if single_vector:
            vnode = np.empty(self.mesh.nnodes*self.ncomp)
            for icomp in range(self.ncomp):
                vnode[icomp::self.ncomp] = self.femv.fem.tonode(v[icomp::self.ncomp])
            data['point']['V'] = vnode
        else:
            for icomp in range(self.ncomp):
                data['point'][f'V_{icomp:1d}'] = self.femv.fem.tonode(v[icomp::self.ncomp])
        data['cell']['P'] = p
        return data
    def postProcess(self, u):
        if self.pmean: v, p, lam = self._split(u)
        else: v, p = self._split(u)
        data = {'scalar':{}}
        if self.problemdata.solexact:
            err, e = self.femv.computeErrorL2(self.problemdata.solexact[0], v)
            data['scalar']['error_V_L2'] = np.sum(err)
            err, e = self.femp.computeErrorL2(self.problemdata.solexact[1], p)
            data['scalar']['error_P_L2'] = err
        if self.problemdata.postproc:
            types = ["bdry_pmean", "bdry_vmean", "bdry_nflux"]
            for name, type in self.problemdata.postproc.type.items():
                colors = self.problemdata.postproc.colors(name)
                if type == types[0]:
                    # data['scalar'][name] = self.femp.computeBdryMean(p, colors)
                    pp = self.femp.computeBdryMean(p, colors)
                elif type == types[1]:
                    # data['scalar'][name] = self.femv.computeBdryMean(v, colors)
                    pp = self.femv.computeBdryMean(v, colors)
                elif type == types[2]:
                    if self.dirichletmethod=='strong':
                        # data['scalar'][name] = self.computeBdryNormalFluxStrong(v, p, colors)
                        pp = self.computeBdryNormalFluxStrong(v, p, colors)
                    else:
                        # data['scalar'][name] = self.computeBdryNormalFluxNitsche(v, p, colors)
                        pp = self.computeBdryNormalFluxNitsche(v, p, colors)
                else:
                    raise ValueError(f"unknown postprocess type '{type}' for key '{name}'\nknown types={types=}")
                if pp.ndim==1:
                    for i, color in enumerate(colors):
                        data['scalar'][name + "_" + f"{color}"] = pp[i]
                else:
                    assert pp.ndim==2
                    for i,color in enumerate(colors):
                        data['scalar'][name+"_"+f"{color}"] = pp[:,i]
                # print(f"{name=} {data['scalar'][name].shape=}")
        return data
    def computelinearSolver(self, A):
        if self.linearsolver == 'spsolve':
            args = {'method':'spsolve'}
        else:
            args = copy.deepcopy(self.linearsolver)
        if args['method'] != 'spsolve':
            if self.scale_ls and hasattr(A,'scale_A'):
                A.scale_A()
                args['scale'] = self.scale_ls
            # args['counter'] = 'sys'
            args['matvec'] = A.matvec
            args['n'] = A.nall
            prec = args.pop('prec', 'full')
            solver_v = args.pop('solver_v', None)
            solver_p = args.pop('solver_p', None)
            if prec == 'BS':
                alpha = args.pop('alpha', 10)
                P = saddle_point.BraessSarazin(A, alpha=alpha)
            elif prec == 'Chorin':
                P = saddle_point.Chorin(A, solver_v=solver_v, solver_p=solver_p)
            else:
                # if solver_p['type']=='scale':
                # solver_p['coeff'] = self.mesh.dV/self.mucell
                P = saddle_point.SaddlePointPreconditioner(A, solver_v=solver_v, solver_p=solver_p, method=prec)
            args['matvecprec'] = P.matvecprec
        return linalg.getLinearSolver(args=args)
    def computeRhs(self, b=None, u=None, coeffmass=None):
        b = self._zeros()
        bs  = self._split(b)
        bv,bp = bs[0], bs[1]
        if 'rhs' in self.problemdata.params.fct_glob:
            rhsv, rhsp = self.problemdata.params.fct_glob['rhs']
            if rhsv: self.femv.computeRhsCells(bv, rhsv)
            if rhsp: self.femp.computeRhsCells(bp, rhsp)
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsneu = self.problemdata.bdrycond.colorsOfType("Neumann")
        colorsnav = self.problemdata.bdrycond.colorsOfType("Navier")
        colorsp = self.problemdata.bdrycond.colorsOfType("Pressure")
        self.femv.computeRhsBoundary(bv, colorsneu, self.problemdata.bdrycond.fct)
        if not self.dirichletmethod == 'strong':
            vdir = self.femv.interpolateBoundary(colorsdir, self.problemdata.bdrycond.fct)
            self.computeRhsBdryNitscheDirichlet((bv,bp), colorsdir, vdir, self.mucell)
            bdryfct = self.problemdata.bdrycond.fct
            # Navier condition
            colors = set(bdryfct.keys()).intersection(colorsnav)
            if len(colors):
                if not isinstance(bdryfct[next(iter(colors))],dict):
                    msg = """
                    For Navier b.c. please give a dictionary {vn:fct_scal, g:fvt_vec} with fct_scal scalar and fvt_vec a list of dim functions
                    """
                    raise ValueError(msg+f"\ngiven: {bdryfct[next(iter(colors))]=}")
                vnfct, gfct = {}, {}
                for col in colors:
                    if 'vn' in bdryfct[col].keys() : 
                        if not callable(bdryfct[col]['vn']):
                            raise ValueError(f"'vn' must be a function. Given:{bdryfct[col]['vn']=}")
                        vnfct[col] = bdryfct[col]['vn']
                    if 'g' in bdryfct[col].keys() : 
                        if not isinstance(bdryfct[col]['g'], list) or len(bdryfct[col]['g'])!=self.ncomp:
                            raise ValueError(f"'g' must be a list of functions with {self.ncomp} elements. Given:{bdryfct[col]['g']=}")
                        gfct[col] = bdryfct[col]['g']
                if len(vnfct): 
                    vn = self.femv.fem.interpolateBoundary(colorsnav, vnfct, lumped=False)
                    self.computeRhsBdryNitscheNavierNormal((bv,bp), colorsnav, self.mucell, vn)
                if len(gfct): 
                    gt = self.femv.interpolateBoundary(colorsnav, gfct)
                    self.computeRhsBdryNitscheNavierTangent((bv,bp), colorsnav, self.mucell, gt)
            # Pressure condition
            colors = set(bdryfct.keys()).intersection(colorsp)
            if len(colors):
                if not isinstance(bdryfct[next(iter(colors))],dict):
                    msg = """
                    For Pressure b.c. please give a dictionary {p:fct_scal, v:fvt_vec} with fct_scal scalar and fvt_vec a list of dim functions
                    """
                    raise ValueError(msg+f"\ngiven: {bdryfct[next(iter(colors))]=}")
                pfct, vfct = {}, {}
                for col in colors:
                    if 'p' in bdryfct[col].keys() : 
                        if not callable(bdryfct[col]['p']):
                            raise ValueError(f"'vn' must be a function. Given:{bdryfct[col]['p']=}")
                        pfct[col] = bdryfct[col]['p']
                    if 'v' in bdryfct[col].keys() : 
                        if not isinstance(bdryfct[col]['v'], list) or len(bdryfct[col]['v'])!=self.ncomp:
                            raise ValueError(f"'v' must be a list of functions with {self.ncomp} elements. Given:{bdryfct[col]['v']=}")
                        vfct[col] = bdryfct[col]['v']
                if len(pfct):
                    p = self.femv.fem.interpolateBoundary(colorsp, pfct, lumped=False)
                    self.computeRhsBdryNitschePressureNormal((bv,bp), colorsp, self.mucell, p)
                if len(vfct): 
                    v = self.femv.interpolateBoundary(colorsp, vfct)
                    self.computeRhsBdryNitschePressureTangent((bv,bp), colorsp, self.mucell, v)
        if not self.pmean: return b
        if self.problemdata.solexact is not None:
            p = self.problemdata.solexact[1]
            bmean = self.femp.computeMean(p)
        else: bmean=0
        b[-1] = bmean
        return b
    def computeForm(self, u, coeffmass=None):
        d = np.zeros_like(u)
        if self.pmean: 
            v, p, lam = self._split(u)
            dv, dp, dlam = self._split(d)
        else: 
            v, p = self._split(u)
            dv, dp = self._split(d)
        # d2 = self.matrixVector(self.A, u)
        self.femv.computeFormLaplace(self.mucell, dv, v)
        self.femv.computeFormDivGrad(dv, dp, v, p)
        if coeffmass: self.femv.computeFormMass(dv, v, coeffmass)
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsnav = self.problemdata.bdrycond.colorsOfType("Navier")
        if self.dirichletmethod == 'strong':
            self.femv.vectorBoundaryStrongEqual(dv, v, self.bdrydata)
            # self.femv.formBoundary(dv, self.bdrydata, self.dirichletmethod)
        else:
            self.computeFormBdryNitscheDirichlet(dv, dp, v, p, colorsdir, self.mucell)
            self.computeFormBdryNitscheNavier(dv, dp, v, p, colorsnav, self.mucell)
        if self.pmean:
            self.computeFormMeanPressure(dp, dlam, p, lam)
        # if not np.allclose(d,d2):
        #     raise ValueError(f"{d=}\n{d2=}")
        return d
    def computeMassMatrix(self):
        class MassMatrixIncompressible:
            def __init__(self, app, M):
                self.app = app
                self.M = M
            def addToStokes(self, d, A):
                if self.app.singleA:
                    A += d*self.M
                else:
                    A += linalg.matrix2systemdiagonal(d*self.M, self.app.ncomp)
                return A
            def dot(self, du, d, u):
                # print(f"{d=}")
                n = self.M.shape[0]
                v, p = self.app._split(u)
                dv, dp = self.app._split(du)
                for i in range(self.app.ncomp):
                    dv[i::self.app.ncomp] += d*self.M.dot(v[i::self.app.ncomp])
                return du
        return MassMatrixIncompressible(self, self.femv.fem.computeMassMatrix())
    def rhs_dynamic(self, rhs, u, Aimp, time, dt, theta):
        # print(f"{u.shape=} {rhs.shape=} {type(Aconst)=}")
        # rhs += 1 / (theta * theta * dt) * self.Mass.dot(u)
        self.Mass.dot(rhs, 1 / (theta * theta * dt), u)
        rhs += (theta - 1) / theta * Aimp.dot(u)
        # print(f"@1@{np.min(u)=} {np.max(u)=} {np.min(rhs)=} {np.max(rhs)=}")
        rhs2 = self.computeRhs()
        rhs += (1 / theta) * rhs2
    def defect_dynamic(self, f, u):
        y = self.computeForm(u)-f
        self.Mass.dot(y, 1 / (self.theta * self.dt), u)
        return y
   #     return u, niter
    def computeMatrix(self, u=None, coeffmass=None):
        if coeffmass is None and 'alpha' in self.problemdata.params.scal_glob.keys():
            coeffmass = self.problemdata.params.scal_glob['alpha']
        # print(f"computeMatrix {coeffmass=}")
        if self.singleA:
            A = self.femv.fem.computeMatrixDiffusion(self.mucell, coeffmass)
        else:
            A = self.femv.computeMatrixLaplace(self.mucell, coeffmass)
        B = self.femv.computeMatrixDivergence()
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsnav = self.problemdata.bdrycond.colorsOfType("Navier")
        colorsp = self.problemdata.bdrycond.colorsOfType("Pressure")
        if len(colorsp):
            raise NotImplementedError(f"Pressure boundary consition wrong (in newton)")
        if self.dirichletmethod == 'strong':
            A, B = self.matrixBoundaryStrong(A, B, self.bdrydata, self.dirichletmethod)
            self.vectorBoundaryStrong(self.b, self.problemdata.bdrycond.fct, self.bdrydata, self.dirichletmethod)
        else:
            #TODO eviter le retour de A,B
            # print(f"{id(A)=} {id(B)=}")
            A, B = self.computeMatrixBdryNitscheDirichlet(A, B, colorsdir, self.mucell, self.singleA)
            # print(f"{id(A)=} {id(B)=}")
            if len(colorsnav):
                lam = self.problemdata.params.scal_glob.get('navier', 0)
                A, B = self.computeMatrixBdryNitscheNavier(A, B, colorsnav, self.mucell, lam)
            if len(colorsp): A, B = self.computeMatrixBdryNitschePressure(A, B, colorsp, self.mucell)
            # print(f"{id(A)=} {id(B)=}")
        # if self.hdivpenalty:
        #     if not hasattr(self.mesh,'innerfaces'): self.mesh.constructInnerFaces()
        #     A += self.femv.computeMatrixHdivPenaly(self.hdivpenalty)
        # if self.divdivparam:
        #     A += self.femv.computeMatrixDivDiv(self.divdivparam)
        # if coeffmass:
        #     print(f"{coeffmass=}")
        #     self.Mass.addToStokes(coeffmass, A)
        # print(f"{self.linearsolver=}")
        if not self.pmean:
            return linalg.SaddlePointSystem(A, B, singleA=self.singleA, ncomp=self.ncomp)
        ncells = self.mesh.ncells
        rows = np.zeros(ncells, dtype=int)
        cols = np.arange(0, ncells)
        C = sparse.coo_matrix((self.mesh.dV, (rows, cols)), shape=(1, ncells)).tocsr()
        return linalg.SaddlePointSystem(A, B, singleA=self.singleA, ncomp=self.ncomp)
    def computeFormMeanPressure(self,dp, dlam, p, lam):
        dlam += self.mesh.dV.dot(p)
        dp += lam*self.mesh.dV
    def computeBdryNormalFluxNitsche(self, v, p, colors):
        ncomp, bdryfct = self.ncomp, self.problemdata.bdrycond.fct
        flux = np.zeros(shape=(ncomp,len(colors)))
        vdir = self.femv.interpolateBoundary(colors, bdryfct).ravel()
        for icomp in range(ncomp):
            flux[icomp] = self.femv.fem.computeBdryNormalFluxNitsche(v[icomp::ncomp], colors, vdir[icomp::ncomp], self.mucell)
            for i,color in enumerate(colors):
                faces = self.mesh.bdrylabels[color]
                cells = self.mesh.cellsOfFaces[faces,0]
                normalsS = self.mesh.normals[faces][:,:ncomp]
                dS = np.linalg.norm(normalsS, axis=1)
                flux[icomp,i] -= p[cells].dot(normalsS[:,icomp])
        return flux
    def computeRhsBdryNitscheDirichlet(self, b, colors, vdir, mucell, coeff=1):
        bv, bp = b
        ncomp  = self.ncomp
        faces = self.mesh.bdryFaces(colors)
        cells = self.mesh.cellsOfFaces[faces,0]
        normalsS = self.mesh.normals[faces][:,:ncomp]
        np.add.at(bp, cells, -np.einsum('nk,nk->n', coeff*vdir[faces], normalsS))
        self.femv.computeRhsNitscheDiffusion(bv, mucell, colors, vdir, ncomp)
    def computeRhsBdryNitscheNavierNormal(self, b, colors, mucell, vn):
        bv, bp = b
        ncomp, dim  = self.ncomp, self.mesh.dimension
        faces = self.mesh.bdryFaces(colors)
        cells = self.mesh.cellsOfFaces[faces,0]
        normalsS = self.mesh.normals[faces][:,:ncomp]
        dS = np.linalg.norm(normalsS, axis=1)
        # normals = normalsS/dS[:,np.newaxis]
        # foc = self.mesh.facesOfCells[cells]
        np.add.at(bp, cells, -dS*vn[faces])
        self.femv.computeRhsNitscheDiffusionNormal(bv, mucell, colors, vn, ncomp)
    def computeRhsBdryNitscheNavierTangent(self, b, colors, mucell, gt):
        bv, bp = b
        ncomp, dim  = self.ncomp, self.mesh.dimension
        self.femv.massDotBoundary(bv, gt.ravel(), colors=colors, ncomp=ncomp, coeff=1)
        self.femv.massDotBoundaryNormal(bv, -gt.ravel(), colors=colors, ncomp=ncomp, coeff=1)
    def computeRhsBdryNitschePressureNormal(self, b, colors, mucell, p):
        bv, bp = b
        self.femv.massDotBoundaryNormal(bv, -p, colors=colors, ncomp=self.ncomp, coeff=1)
    def computeRhsBdryNitschePressureTangent(self, b, colors, mucell, v):
        bv, bp = b
        ncomp, dim  = self.ncomp, self.mesh.dimension
        self.femv.computeRhsNitscheDiffusion(bv, mucell, colors, v, ncomp)
        self.femv.computeRhsNitscheDiffusionNormal(bv, mucell, colors, -v.ravel(), ncomp)
    def computeFormBdryNitscheDirichlet(self, dv, dp, v, p, colorsdir, mu):
        ncomp, dim  = self.femv.ncomp, self.mesh.dimension
        self.femv.computeFormNitscheDiffusion(dv, v, mu, colorsdir, ncomp)
        faces = self.mesh.bdryFaces(colorsdir)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :self.ncomp]
        for icomp in range(ncomp):
            r = np.einsum('f,f->f', p[cells], normalsS[:,icomp])
            np.add.at(dv[icomp::ncomp], faces, r)
            r = np.einsum('f,f->f', normalsS[:,icomp], v[icomp::ncomp][faces])
            np.add.at(dp, cells, -r)
    def computeFormBdryNitscheNavier(self, dv, dp, v, p, colors, mu):
        if not len(colors): return
        raise NotImplementedError()
    def computeMatrixBdryNitscheDirichlet(self, A, B, colors, mucell, singleA):
        nfaces, ncells, ncomp, dim  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp, self.mesh.dimension
        if singleA:
            A += self.femv.fem.computeMatrixNitscheDiffusion(mucell, colors)
        else:
            A += self.femv.computeMatrixNitscheDiffusion(mucell, colors, ncomp)
        #grad-div
        faces = self.mesh.bdryFaces(colors)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :self.ncomp]
        indfaces = np.repeat(ncomp * faces, ncomp)
        for icomp in range(ncomp): indfaces[icomp::ncomp] += icomp
        cols = indfaces.ravel()
        rows = cells.repeat(ncomp).ravel()
        mat = normalsS.ravel()
        B -= sparse.coo_matrix((mat, (rows, cols)), shape=(ncells, ncomp*nfaces))
        return A,B
    def computeMatrixBdryNitscheNavier(self, A, B, colors, mucell, lambdaR):
        nfaces, ncells, ncomp, dim  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp, self.mesh.dimension
        faces = self.mesh.bdryFaces(colors)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :dim]
        #grad-div
        indfaces = np.repeat(ncomp * faces, ncomp)
        for icomp in range(ncomp): indfaces[icomp::ncomp] += icomp
        cols = indfaces.ravel()
        rows = cells.repeat(ncomp).ravel()
        B -= sparse.coo_matrix((normalsS.ravel(), (rows, cols)), shape=(ncells, ncomp*nfaces))
        #vitesses
        A += self.femv.computeMatrixNitscheDiffusionNormal(mucell, colors, ncomp)
        A += self.femv.computeMassMatrixBoundary(colors, ncomp, coeff=lambdaR)-self.femv.computeMassMatrixBoundaryNormal(colors, ncomp, coeff=lambdaR)
        return A,B
    def computeMatrixBdryNitschePressure(self, A, B, colors, mucell):
        #vitesses
        A += self.femv.computeMatrixNitscheDiffusion(mucell, colors, self.ncomp)
        A -= self.femv.computeMatrixNitscheDiffusionNormal(mucell, colors, self.ncomp)
        return A,B
    def vectorBoundaryStrong(self, b, bdryfctv, bdrydata, method):
        # bv, bp = b
        bs = self._split(b)
        bv, bp = bs[0], bs[1]
        bv = self.femv.vectorBoundaryStrong(bv, bdryfctv, bdrydata, method)
        facesdirall, facesinner, colorsdir, facesdirflux = bdrydata.facesdirall, bdrydata.facesinner, bdrydata.colorsdir, bdrydata.facesdirflux
        nfaces, ncells, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp
        bdrydata.bsaved = {}
        for key, faces in facesdirflux.items():
            indfaces = np.repeat(ncomp * faces, ncomp)
            for icomp in range(ncomp): indfaces[icomp::ncomp] += icomp
            bdrydata.bsaved[key] = bv[indfaces]
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        #suppose strong-trad
        bp -= bdrydata.B_inner_dir * bv[inddir]
        return (bv,bp)
    def matrixBoundaryStrong(self, A, B, bdrydata, method):
        A = self.femv.matrixBoundaryStrong(A, bdrydata, method)
        facesdirall, facesinner, colorsdir, facesdirflux = bdrydata.facesdirall, bdrydata.facesinner, bdrydata.colorsdir, bdrydata.facesdirflux
        nfaces, ncells, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.femv.ncomp
        bdrydata.Bsaved = {}
        for key, faces in facesdirflux.items():
            nb = faces.shape[0]
            helpB = sparse.dok_matrix((ncomp*nfaces, ncomp*nb))
            for icomp in range(ncomp):
                for i in range(nb): helpB[icomp + ncomp*faces[i], icomp + ncomp*i] = 1
            bdrydata.Bsaved[key] = B.dot(helpB)
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        bdrydata.B_inner_dir = B[:,:][:,inddir]
        help = np.ones((ncomp * nfaces))
        help[inddir] = 0
        help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
        B = B.dot(help)
        return A,B
    def computeBdryNormalFluxStrong(self, v, p, colors):
        nfaces, ncells, ncomp, bdrydata  = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.bdrydata
        flux, omega = np.zeros(shape=(ncomp,len(colors))), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            As = bdrydata.Asaved[color]
            Bs = bdrydata.Bsaved[color]
            res = bdrydata.bsaved[color] - As * v + Bs.T * p
            for icomp in range(ncomp):
                flux[icomp, i] = np.sum(res[icomp::ncomp])
            # print(f"{flux=}")
            #TODO flux Stokes Dirichlet strong wrong
        return flux
    def _plot2d(self, data, fig, gs, title='', alpha=0.5):
        import matplotlib.gridspec as gridspec
        inner = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs, wspace=0.1, hspace=0.1)
        v = data['point']['V']
        p = data['cell']['P']
        ax = fig.add_subplot(inner[0])
        self.plot_p(ax, p)
        ax = fig.add_subplot(inner[1])
        self.plot_v(ax, v)
    def _plot3d(self, mesh, data, fig, gs, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        inner = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs, wspace=0.01, hspace=0.01)
        import pyvista
        alpha = kwargs.pop('alpha', 0.6)
        p = data['cell']['P']
        v = data['point']['V']
        vr = v.reshape(self.mesh.nnodes, self.ncomp)
        vnorm = np.linalg.norm(vr, axis=1)
        mesh["V"] = vr
        mesh["vn"] = vnorm
        mesh.cell_data['P'] = p
        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen',True))
        plotter.renderer.SetBackground(255, 255, 255)
        plotter.add_mesh(mesh, opacity=alpha, color='gray', show_edges=True)
        plotter.show(title=kwargs.pop('title', self.__class__.__name__))
        ax = fig.add_subplot(inner[0])
        ax.imshow(plotter.image)
        ax.set_xticks([])
        ax.set_yticks([])


        scalar_bar_args = {'title': 'p', 'color':'black'}
        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen',True))
        plotter.renderer.SetBackground(255, 255, 255)
        plotter.add_mesh(mesh, opacity=alpha, color='gray', scalars='P', scalar_bar_args=scalar_bar_args)
        plotter.show(title=kwargs.pop('title', self.__class__.__name__))
        ax = fig.add_subplot(inner[1])
        ax.imshow(plotter.image)
        ax.set_xticks([])
        ax.set_yticks([])

        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen',True))
        plotter.renderer.SetBackground(255, 255, 255)
        glyphs = mesh.glyph(orient="V", scale="vn", factor=10)
        # mesh.set_active_vectors("vectors")
        plotter.add_mesh(glyphs, show_scalar_bar=False, lighting=False, cmap='coolwarm')
        # plotter.show(title=kwargs.pop('title', self.__class__.__name__))
        # mesh.arrows.plot(off_screen=kwargs.pop('off_screen',True))
        plotter.show(title=kwargs.pop('title', self.__class__.__name__))
        ax = fig.add_subplot(inner[2])
        ax.imshow(plotter.image)
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_p(self, ax, p=None, uname='p', title='', alpha=0.5):
        import matplotlib.pyplot as plt
        if self.mesh.dimension==2:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            x, y, tris = self.mesh.points[:,0], self.mesh.points[:,1], self.mesh.simplices
            if title: ax.set_title(title)
            ax.triplot(x, y, tris, color='gray', lw=1, alpha=alpha)
            assert len(p)==self.mesh.ncells
            cnt = ax.tripcolor(x, y, tris, facecolors=p, edgecolors='k', cmap='jet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = plt.colorbar(cnt, cax=cax, orientation='vertical')
            clb.ax.set_title(uname)
        else:
            from simfempy.meshes.plotmesh3d import plotmesh
            plotmesh(self.mesh)
    def plot_v(self, ax, v=None, uname='v', title='', alpha=0.5):
        import matplotlib.pyplot as plt
        if self.mesh.dimension==2:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            x, y, tris = self.mesh.points[:,0], self.mesh.points[:,1], self.mesh.simplices
            if title: ax.set_title(title)
            ax.triplot(x, y, tris, color='gray', lw=1, alpha=alpha)
            vr = v.reshape(self.mesh.nnodes,self.ncomp)
            vnorm = np.linalg.norm(vr, axis=1)
            cnt = ax.tricontourf(x, y, tris, vnorm, levels=16, cmap='jet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = plt.colorbar(cnt, cax=cax, orientation='vertical')
            clb.ax.set_title(uname)
            ax.quiver(x, y, vr[:,0], vr[:,1], units='xy')
        else:
            print(f"3D not written")

#=================================================================#
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from simfempy.meshes import plotmesh, animdata
    from simfempy.examples.incompflow import schaeferTurek2d
    mesh, data = schaeferTurek2d(h=0.4, mu=1)
    stokes = Stokes(mesh=mesh, problemdata=data, femparams={'dirichletmethod':'strong'}, linearsolver='spsolve')
    print(f"{stokes=}")
    # results = stokes.static(mode="linear")
    results = stokes.static(mode="newton")
    print(f"{results.data['scalar']=}")
    fig = plt.figure(1)
    gs = fig.add_gridspec(2, 1)
    plotmesh.meshWithBoundaries(stokes.mesh, gs=gs[0,0], fig=fig)
    plotmesh.meshWithDataNew(stokes.mesh, data=results.data, alpha=0.5, gs=gs[1,0], fig=fig)
    plt.show()
