import numpy as np
import scipy.linalg as linalg
import scipy.sparse
import scipy.sparse.linalg as splinalg
from simfempy import solvers
import simfempy.tools.iterationcounter


#=================================================================#
class LaplaceMixed(solvers.solver.Application):
    """
    """
    def defineRhsAnalyticalSolution(self, solexact):
        # constant diffusion
        def _fctu(x, y, z):
            rhs = np.zeros(x.shape[0])
            if self.beta is not None:
                beta = self.beta(x,y,z)
                for i in range(self.mesh.dimension):
                    rhs += beta[i] * solexact.d(i, x, y, z)
            for i in range(self.mesh.dimension):
                rhs -= self.diff(0)*solexact.dd(i, i, x, y, z)
            return rhs
        return _fctu

    def defineNeumannAnalyticalSolution(self, solexact, color):
        def _fctneumann(x, y, z, nx, ny, nz):
            rhs = np.zeros(x.shape[0])
            normals = nx, ny, nz
            for i in range(self.mesh.dimension):
                rhs += self.diff(0)*solexact.d(i, x, y, z) * normals[i]
            return rhs
        return _fctneumann

    def defineRobinAnalyticalSolution(self, solexact, color):
        return solexact

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linearsolver = "gmres"
        if 'linearsolver' in kwargs:
            self.linearsolver = kwargs.pop('linearsolver')
        if 'massproj' in kwargs: massproj = kwargs.pop('massproj')
        else: massproj=None
        if 'fem' in kwargs:
            fem = kwargs.pop('fem')
            if fem =='bv0':
                self.femv = simfempy.fems.fembv0.FemBV0()
            else:
                self.femv = simfempy.fems.femrt0.FemRT0(massproj=massproj)
        else:
            self.femv = simfempy.fems.femrt0.FemRT0(massproj=massproj)
        if 'diff' in kwargs:
            self.diff = np.vectorize(kwargs.pop('diff'))
        else:
            self.diff = np.vectorize(lambda i: 1)
            # self.diff = np.vectorize(lambda i: 0.123)
        if 'beta' in kwargs:
            self.beta = np.vectorize(kwargs.pop('beta'))
        else:
            self.beta = None
        if 'method' in kwargs:
            self.method = kwargs.pop('method')
        else:
            self.method="trad"
        if 'plotdiff' in kwargs:
            self.plotdiff = kwargs.pop('plotdiff')
        else:
            self.plotdiff = False
        if 'mesh' in kwargs:
            self.setMesh(kwargs.pop('mesh'))

    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.femv.setMesh(mesh)
        self.diffcell = self.diff(self.mesh.cell_labels)
        self.diffcellinv = 1/self.diffcell

    def postProcess(self, u):
        nfaces, dim =  self.mesh.nfaces, self.mesh.dimension
        info = {}
        cell_data = {'p': u[nfaces:]}
        vc = self.femv.toCell(u[:nfaces])
        pn = self.femv.reconstruct(u[nfaces:], vc, self.diffcellinv)
        for i in range(dim):
            cell_data['v{:1d}'.format(i)] = vc[i::dim]
        point_data = {}
        point_data['p_1'] = pn
        if self.problemdata.solexact:
            err, pe, vexx = self.computeError(self.problemdata.solexact, u[nfaces:], vc, pn)
            # print("vexx", vexx)
            # print("u[:nfaces]", u[:nfaces])
            # print("vc[i::dim]", vc[i::dim])
            cell_data['perr'] = np.abs(pe - u[nfaces:])
            for i in range(dim):
                cell_data['verrx{:1d}'.format(i)] = np.abs(vexx[i] - vc[i::dim])
            info['error'] = err
        info['postproc'] = {}
        if self.problemdata.postproc:
            for name, type in self.problemdata.postproc.type.items():
                colors = self.problemdata.postproc.colors(name)
                if type == "bdrymean":
                    info['postproc'][name] = self.computeBdryMean(pn, colors)
                elif type == "bdryfct":
                    info['postproc'][name] = self.computeBdryFct(u, type, colors)
                elif type == "bdrydn":
                    info['postproc'][name] = self.computeBdryDn(u, colors)
                elif type == "pointvalues":
                    info['postproc'][name] = self.computePointValues(u, type, colors)
                else:
                    raise ValueError("unknown postprocess '{}' for key '{}'".format(type, colors))
        if self.plotdiff: cell_data['diff'] = self.diffcell
        return point_data, cell_data, info

    def computeBdryDn(self, u, data):
        colors = [int(x) for x in data.split(',')]
        flux, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            flux[i] = np.sum(dS*u[faces])
        return flux
        return flux/omega

    def computeBdryMean(self, pn, data):
        colors = [int(x) for x in data.split(',')]
        mean, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            mean[i] = np.sum(dS*np.mean(pn[self.mesh.faces[faces]],axis=1))
        return mean/omega

    def computeError(self, solexact, p, vc, pn):
        nfaces, dim =  self.mesh.nfaces, self.mesh.dimension
        errors = {}
        xc, yc, zc = self.mesh.pointsc.T
        pex = solexact(xc, yc, zc)
        errp = np.sqrt(np.sum((pex-p)**2* self.mesh.dV))
        errv = 0
        vexx=[]
        for i in range(dim):
            solxi = self.diffcell*solexact.d(i, xc, yc, zc)
            errv += np.sum( (solxi-vc[i::dim])**2* self.mesh.dV)
            vexx.append(solxi)
        errv = np.sqrt(errv)
        errors['pcL2'] = errp
        errors['vcL2'] = errv

        x, y, z = self.mesh.points.T
        epn = solexact(x, y, z) - pn
        epn = epn**2
        epn= np.mean(epn[self.mesh.simplices], axis=1)
        errors['pnL2'] = np.sqrt(np.sum(epn* self.mesh.dV))
        return errors, pex, vexx

    def computeRhs(self, u=None):
        xf, yf, zf = self.mesh.pointsf.T
        xc, yc, zc = self.mesh.pointsc.T
        normals =  self.mesh.normals
        bsides = np.zeros(self.mesh.nfaces)
        bcells = np.zeros(self.mesh.ncells)
        if self.problemdata.rhs:
            bcells = -self.problemdata.rhs(xc, yc, zc) * self.mesh.dV
        bdrycond = self.problemdata.bdrycond
        for color, faces in self.mesh.bdrylabels.items():
            if bdrycond.type[color] not in ["Dirichlet","Robin"]: continue
            # nx, ny, nz = np.mean(normals[faces], axis=0)
            # ud = bdrycond.fct[color](xf[faces], yf[faces], zf[faces], nx, ny, nz)
            ud = bdrycond.fct[color](xf[faces], yf[faces], zf[faces])
            # bsides[faces] = linalg.norm(self.mesh.normals[faces],axis=1) * ud
            bsides[faces] = self.femv.rhsDirichlet(faces, ud)
            if self.beta is not None:
                faces = self.mesh.bdrylabels[color]
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                normalsS = normalsS / dS[:, np.newaxis]
                xf, yf, zf = self.mesh.pointsf[faces].T
                beta = np.array(self.beta(xf, yf, zf))
                # print("beta", beta)
                # print("normalsS", normalsS.T)
                bn = np.einsum("ij,ij->j", beta, normalsS.T)
                # print("bn", bn)
                bn[bn<=0] = 0
                cells = self.mesh.cellsOfFaces[faces,0]
                bcells[cells] += bn*ud*dS

        help = np.zeros(self.mesh.nfaces)
        if hasattr(self.problemdata.bdrycond,'fctexact'):
            for color in self.bdrydata.colorsneum:
                faces = self.mesh.bdrylabels[color]
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                normalsS = normalsS / dS[:, np.newaxis]
                xf, yf, zf = self.mesh.pointsf[faces].T
                nx, ny, nz = normalsS.T
                help[faces] += self.problemdata.bdrycond.fctexact["Neumann"](xf, yf, zf, nx, ny, nz)
            bsides[self.bdrydata.facesinner] -= self.bdrydata.A_inner_neum*help[self.bdrydata.facesneumann]
            bsides[self.bdrydata.facesneumann] += self.bdrydata.A_neum_neum*help[self.bdrydata.facesneumann]
            bcells -= self.bdrydata.B_inner_neum*help[self.bdrydata.facesneumann]

        # for robin-exactsolution
        if self.problemdata.bdrycond.hasExactSolution():
            for color, faces in self.mesh.bdrylabels.items():
                if self.problemdata.bdrycond.type[color] != "Robin": continue
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS,axis=1)
                normalsS = normalsS/dS[:,np.newaxis]
                xf, yf, zf = self.mesh.pointsf[faces].T
                nx, ny, nz = normalsS.T
                bsides[faces] += self.problemdata.bdrycond.fctexact["Neumann"](xf, yf, zf, nx, ny, nz) * dS/self.problemdata.bdrycond.param[color]
        b = np.concatenate((bsides, bcells))
        if u is None: u = np.zeros_like(b)
        else: assert u.shape == b.shape
        return b,u

    def matrix(self):
        A = self.femv.constructMass(self.diffcellinv)
        A += self.femv.constructRobin(self.problemdata.bdrycond, "Robin")
        B = self.femv.constructDiv()
        self.bdrydata, A,B = self.femv.matrixNeumann(A, B, self.problemdata.bdrycond)
        return A,B

    def _to_single_matrix(self, Ain):
        A, B = Ain
        ncells = self.mesh.ncells
        help = np.zeros((ncells))
        help = scipy.sparse.dia_matrix((help, 0), shape=(ncells, ncells))
        A1 = scipy.sparse.hstack([A, B.T])
        A2 = scipy.sparse.hstack([B, help])
        Aall = scipy.sparse.vstack([A1, A2])
        return Aall.tocsr()

    def linearSolver(self, Ain, bin, u=None, solver = None, verbose=0):
        if solver is None: solver = self.linearsolver
        if solver == 'umf':
            # print("bin", bin)
            Aall = self._to_single_matrix(Ain)
            u =  splinalg.spsolve(Aall, bin, permc_spec='COLAMD'), 1
            # print("u", u)
            return u
        elif solver == 'gmres':
            nfaces, ncells = self.mesh.nfaces, self.mesh.ncells
            counter = simfempy.tools.iterationcounter.IterationCounter(name=solver, verbose=verbose)
            # Aall = self._to_single_matrix(Ain)
            # M2 = splinalg.spilu(Aall, drop_tol=0.2, fill_factor=2)
            # M_x = lambda x: M2.solve(x)
            # M = splinalg.LinearOperator(Aall.shape, M_x)
            A, B = Ain
            A, B = A.tocsr(), B.tocsr()
            D = scipy.sparse.diags(1/A.diagonal(), offsets=(0), shape=(nfaces,nfaces))
            S = -B*D*B.T
            # print("S", S)


            # Dex = scipy.sparse.linalg.inv(A)
            # print("Sex", -B*Dex*B.T)


            import pyamg
            config = pyamg.solver_configuration(S, verb=False)
            ml = pyamg.rootnode_solver(S, B=config['B'], smooth='energy')
            # Ailu = splinalg.spilu(A, drop_tol=0.2, fill_factor=2)
            def amult(x):
                v,p = x[:nfaces],x[nfaces:]
                return np.hstack( [A.dot(v) + B.T.dot(p), B.dot(v)])
            Amult = splinalg.LinearOperator(shape=(nfaces+ncells,nfaces+ncells), matvec=amult)
            def pmult(x):
                v,p = x[:nfaces],x[nfaces:]
                w = D.dot(v)
                # w = Ailu.solve(v)
                q = ml.solve(p - B.dot(w), maxiter=1, tol=1e-16)
                w = w - D.dot(B.T.dot(q))
                # w = w - Ailu.solve(B.T.dot(q))
                return np.hstack( [w, q] )
            P = splinalg.LinearOperator(shape=(nfaces+ncells,nfaces+ncells), matvec=pmult)
            u,info = splinalg.lgmres(Amult, bin, M=P, callback=counter, atol=1e-12, tol=1e-12, inner_m=10, outer_k=4)
            if info: raise ValueError("no convergence info={}".format(info))
            # print("u", u)
            return u, counter.niter
        else:
            raise NotImplementedError("solver '{}' ".format(solver))

#=================================================================#
if __name__ == '__main__':
    import simfempy
    from simfempy.meshes.simplexmesh import SimplexMesh
    from simfempy.meshes import plotmesh
    import pygmsh
    import matplotlib.pyplot as plt
    geometry = pygmsh.built_in.Geometry()
    a = 1.0
    h = 2
    p = geometry.add_rectangle(xmin=-a, xmax=a, ymin=-a, ymax=a, z=0, lcar=h)
    geometry.add_physical(p.surface, label=100)
    for i in range(4): geometry.add_physical(p.line_loop.lines[i], label=1000 + i)
    mesh = pygmsh.generate_mesh(geometry, verbose=False)
    mesh = SimplexMesh(mesh=mesh, hmean=h)

    data = simfempy.applications.problemdata.ProblemData()
    bdrycond =  data.bdrycond
    postproc = data.postproc
    bdrycond.set("Dirichlet", [1000, 1001, 1002, 1003])
    data.params.scal_glob['kheat'] = 0.123
    laplace = LaplaceMixed(mesh=mesh, problemdata=data, showmesh=False)
    exactsolution = "Linear"
    problemdata = laplace.generatePoblemDataForAnalyticalSolution(exactsolution=exactsolution, problemdata=data, random=False)
    laplace = LaplaceMixed(problemdata=problemdata, mesh=mesh)
    # print("A=",laplace.A)
    # print("B=",laplace.B)
    point_data, cell_data, info = laplace.solve(dirname="Results")
    print("point_data", point_data)
    print("cell_data", cell_data)
    plotmesh.meshWithData(mesh, point_data=point_data, cell_data=cell_data, title="Test", suptitle="LaplaceMixed")
    plt.show()