import numpy as np
import scipy.linalg as linalg
import scipy.sparse
import scipy.sparse.linalg as splinalg
import simfempy
from simfempy import solvers
import simfempy.tools.iterationcounter


#=================================================================#
class LaplaceMixed(solvers.solver.Solver):
    """
    """
    def defineRhsAnalyticalSolution(self, solexact):
        def _fctu(x, y, z):
            rhs = np.zeros(x.shape[0])
            for i in range(self.mesh.dimension):
                rhs -= self.diff(0)*solexact.dd(i, i, x, y, z)
            return rhs
        return _fctu

    def defineNeumannAnalyticalSolution(self, solexact):
        def _fctneumann(x, y, z, nx, ny, nz):
            rhs = np.zeros(x.shape[0])
            normals = nx, ny, nz
            for i in range(self.mesh.dimension):
                rhs += self.diff(0)*solexact.d(i, x, y, z) * normals[i]
            return rhs
        return _fctneumann

    def defineRobinAnalyticalSolution(self, solexact):
        return solexact

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linearsolver = "gmres"
        self.femv = simfempy.fems.femrt0.FemRT0()
        if hasattr(self,'problemdata') and hasattr(self.problemdata,'diffcoeff'):
            self.diff = np.vectorize(self.problemdata.diffcoeff)
        else:
            self.diff = np.vectorize(lambda i: 0.123)
        if 'method' in kwargs:
            self.method = kwargs.pop('method')
        else:
            self.method="trad"

    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.femv.setMesh(mesh)
        self.diffcell = self.diff(self.mesh.cell_labels)
        self.diffcellinv = 1/self.diffcell

    def solve(self, iter, dirname):
        return self.solveLinear()

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
            cell_data['perr'] = np.abs(pe - u[nfaces:])
            for i in range(dim):
                cell_data['verrx{:1d}'.format(i)] = np.abs(vexx[i] - vc[i::dim])
            info['error'] = err
        info['postproc'] = {}
        if self.problemdata.postproc:
            for key, val in self.problemdata.postproc.items():
                type,data = val.split(":")
                if type == "bdrymean":
                    info['postproc'][key] = self.computeBdryMean(pn, key, data)
                elif type == "bdryfct":
                    info['postproc'][key] = self.computeBdryFct(u, key, data)
                elif type == "bdrydn":
                    info['postproc'][key] = self.computeBdryDn(u, key, data)
                elif type == "pointvalues":
                    info['postproc'][key] = self.computePointValues(u, key, data)
                else:
                    raise ValueError("unknown postprocess '{}' for key '{}'".format(type, key))
        return point_data, cell_data, info

    def computeBdryDn(self, u, key, data):
        colors = [int(x) for x in data.split(',')]
        mean, omega = 0, 0
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega += np.sum(dS)
            mean += np.sum(dS*u[faces])
        return mean
        return mean/omega

    def computeBdryMean(self, pn, key, data):
        colors = [int(x) for x in data.split(',')]
        mean, omega = 0, 0
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega += np.sum(dS)
            mean += np.sum(dS*np.mean(pn[self.mesh.faces[faces]],axis=1))
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
        bcells = -self.problemdata.rhs(xc, yc, zc) * self.mesh.dV
        bsides = np.zeros(self.mesh.nfaces)

        for color, faces in self.mesh.bdrylabels.items():
            if self.problemdata.bdrycond.type[color] not in ["Dirichlet","Robin"]: continue
            ud = self.problemdata.bdrycond.fct[color](xf[faces], yf[faces], zf[faces])
            bsides[faces] = linalg.norm(self.mesh.normals[faces],axis=1) * ud

        help = np.zeros(self.mesh.nfaces)
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

    def linearSolver(self, Ain, bin, u=None, solver = 'umf'):
        if solver == 'umf':
            Aall = self._to_single_matrix(Ain)
            return splinalg.spsolve(Aall, bin, permc_spec='COLAMD'), 1
        elif solver == 'gmres':
            nfaces, ncells = self.mesh.nfaces, self.mesh.ncells
            counter = simfempy.tools.iterationcounter.IterationCounter(name=solver)
            # Aall = self._to_single_matrix(Ain)
            # M2 = splinalg.spilu(Aall, drop_tol=0.2, fill_factor=2)
            # M_x = lambda x: M2.solve(x)
            # M = splinalg.LinearOperator(Aall.shape, M_x)
            A, B = Ain
            A, B = A.tocsr(), B.tocsr()
            D = scipy.sparse.diags(1/A.diagonal(), offsets=(0), shape=(nfaces,nfaces))
            S = -B*D*B.T
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
            return u, counter.niter
        else:
            raise NotImplementedError("solver '{}' ".format(solver))

#=================================================================#
if __name__ == '__main__':
    print("Pas encore de test")
