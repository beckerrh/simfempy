import numpy as np
import scipy.linalg as linalg
import fempy.tools.analyticalsolution
import scipy.sparse as sparse
from fempy import solvers
from fempy import fems

#=================================================================#
class Heat(solvers.newtonsolver.NewtonSolver):
    """
    """
    def __init__(self, **kwargs):
        solvers.newtonsolver.NewtonSolver.__init__(self)
        self.fem = fems.femp1.FemP1()
        self.dirichlet = None
        self.neumann = None
        self.rhs = None
        self.solexact = None
        self.bdrycond = kwargs.pop('bdrycond')
        self.kheat = None
        if 'rhocp' in kwargs:
            self.rhocp = np.vectorize(kwargs.pop('rhocp'))
        else:
            self.rhocp = np.vectorize(lambda i: 1234.56)
        if 'kheat' in kwargs:
            self.kheat = np.vectorize(kwargs.pop('kheat'))
        else:
            self.kheat = np.vectorize(lambda i: 0.123)
        if 'problem' in kwargs:
            self.defineProblem(problem=kwargs.pop('problem'))
        else:
            self.rhs = np.vectorize(kwargs.pop('rhs'))
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
        if function == 'Linear':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('0.3 * x + 0.7 * y')
        elif function == 'Linear3d':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('0.3 * x + 0.2 * y + 0.4*z')
        elif function == 'Quadratic':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('x*x+2*y*y')
        elif function == 'Quadratic3d':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('x*x+2*y*y+3*z*z')
        elif function == 'Hubbel':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('(1-x*x)*(1-y*y)')
        elif function == 'Exponential':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('exp(x-0.7*y)')
        elif function == 'Sinus':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('sin(x+0.2*y*y)')
        elif function == 'Sinus3d':
            self.solexact = fempy.tools.analyticalsolution.AnalyticalSolution('sin(x+0.2*y*y+0.5*z)')
        else:
            raise ValueError("unknown analytic solution: {}".format(function))
        class NeummannExact():
            def __init__(self, ex):
                self.ex = ex
            def __call__(self, x, y, z, nx, ny, nz, k):
                return k*(self.ex.x(x, y, z)*nx + self.ex.y(x, y, z)*ny + self.ex.z(x, y, z)*nz)
        class RhsExact():
            def __init__(self, ex, k):
                self.ex = ex
                self.k = k
            def __call__(self, x, y, z):
                return -self.k*(self.ex.xx(x, y, z) + self.ex.yy(x, y, z) + self.ex.zz(x, y, z))
        neumannex = np.vectorize(NeummannExact(self.solexact).__call__)
        self.rhs = np.vectorize(RhsExact(self.solexact, self.kheat).__call__)
        for color, bc in self.bdrycond.type.items():
            if bc == "Dirichlet":
                self.bdrycond.fct[color] = self.solexact
            elif bc == "Neumann":
                self.bdrycond.fct[color] = neumannex
            else:
                raise ValueError("unownd boundary condition {} for color {}".format(bc,color))
    def setMesh(self, mesh):
        self.mesh = mesh
        self.mesh.computeSimpOfVert(test=False)
        nnodes, ncells = self.mesh.nnodes, self.mesh.ncells
        xc, yc, zc = self.mesh.pointsc[:,0], self.mesh.pointsc[:,1], self.mesh.pointsc[:,2]
        self.fem.setMesh(self.mesh)
        self.massmatrix = self.fem.massMatrix()
        colorsdir = []
        self.nodedirall = np.empty(shape=(0), dtype=int)
        for color, type in self.bdrycond.type.items():
            if type == "Dirichlet": colorsdir.append(color)
        self.nodesdir={}
        for color in colorsdir:
            edgesdir = self.mesh.bdrylabels[color]
            self.nodesdir[color] = np.unique(self.mesh.faces[edgesdir].flat[:])
            self.nodedirall = np.unique(np.union1d(self.nodedirall, self.nodesdir[color]))
        # print("colorsdir", colorsdir)
        # print("nodesdir", self.nodesdir)
        self.bsaved={}
        self.Asaved={}
        self.nodesdirflux={}
        for key, val in self.postproc.items():
            type,data = val.split(":")
            if type != "flux": continue
            colors = [int(x) for x in data.split(',')]
            self.nodesdirflux[key] = np.empty(shape=(0), dtype=int)
            for color in colors:
                edgesdir = self.mesh.bdrylabels[color]
                self.nodesdirflux[key] = np.unique(np.union1d(self.nodesdirflux[key], np.unique(self.mesh.faces[edgesdir].flatten())))


        self.kheatcell = np.zeros(ncells)
        self.kheatcell = self.kheat(self.mesh.cell_labels)
        self.rhocpcell = np.zeros(ncells)
        self.rhocpcell = self.rhocp(self.mesh.cell_labels)
        # print("self.kheatcell", self.kheatcell)
    def solvestatic(self):
        return self.solveLinear()
    def solve(self, iter, dirname):
        return self.solveLinear()
    def computeRhs(self):
        import time
        x, y, z = self.mesh.points[:,0], self.mesh.points[:,1], self.mesh.points[:,2]
        t1 = time.time()
        if self.solexact:
            bnodes = -self.solexact.xx(x, y, z) - self.solexact.yy(x, y, z)- self.solexact.zz(x, y, z)
            bnodes *= self.kheat(0)
        else:
            bnodes = self.rhs(x, y, z)
        b = self.massmatrix*bnodes
        t2 = time.time()
        normals =  self.mesh.normals
        for color, edges in self.mesh.bdrylabels.items():
            bdrycond = self.bdrycond.type[color]
            # print("Boundary condition:", bdrycond)
            if bdrycond == "Neumann":
                neumann = self.bdrycond.fct[color]
                scale = 1/self.mesh.dimension
                normalsS = normals[edges]
                dS = linalg.norm(normalsS,axis=1)
                xS = np.mean(self.mesh.points[self.mesh.faces[edges]], axis=1)
                kS = self.kheatcell[self.mesh.cellsOfFaces[edges,0]]
                assert(dS.shape[0] == len(edges))
                assert(xS.shape[0] == len(edges))
                assert(kS.shape[0] == len(edges))
                normalsS[:, 0] /= dS
                normalsS[:, 1] /= dS
                normalsS[:, 2] /= dS
                x1, y1, z1 = xS[:,0], xS[:,1], xS[:,2]
                nx, ny, nz = normalsS[:,0], normalsS[:,1], normalsS[:,2]
                bS =  scale * neumann(x1, y1, z1, nx, ny, nz, kS)*dS
                # print("b[self.mesh.faces[edges].T].shape", b[self.mesh.faces[edges].T].shape, "bS.shape", bS.shape)
                # b[self.mesh.faces[edges]] += bS
                np.add.at(b, self.mesh.faces[edges].T, bS)
                # for ii,ie in enumerate(edges):
                #     b[self.mesh.faces[ie]] += bS[ii]
                # btest = np.zeros((len(edges)))
                # for ii,ie in enumerate(edges):
                #     normal = normals[ie]
                #     ic = self.mesh.cellsOfFaces[ie,0]
                #     pe = np.mean(self.mesh.points[self.mesh.faces[ie]], axis=0)
                #     d = linalg.norm(normal)
                #     assert np.allclose(d, dS[ii])
                #     assert np.allclose(self.kheatcell[ic], kS[ii])
                #     assert np.allclose(pe, xS[ii])
                #     assert np.allclose(normal/d, normalsS[ii])
                #     bn = neumann(pe[0], pe[1], pe[2], normal[0]/d, normal[1]/d, normal[2]/d, self.kheatcell[ic]) * d
                #     btest[ii] = scale * bn
                #     # b[self.mesh.faces[ie]] += scale * bn
                # if not np.allclose(bS, btest):
                #     print("bS", bS)
                #     print("btest", btest)
                #     assert None
        t3 = time.time()
        for key, nodes in self.nodesdirflux.items():
            self.bsaved[key] = b[nodes]
        for color, nodes in self.nodesdir.items():
            dirichlet = self.bdrycond.fct[color]
            b[nodes] = dirichlet(x[nodes], y[nodes], z[nodes])
        t4 = time.time()
        self.timer['rhs_fct'] = t2-t1
        self.timer['rhs_neum'] = t3-t2
        self.timer['rhs_dir'] = t4-t3
        return b
    def matrix(self):
        import time
        nnodes, ncells, normals = self.mesh.nnodes, self.mesh.ncells, self.mesh.normals
        t1 = time.time()
        A = self.fem.assemble(self.kheatcell)
        t2 = time.time()
        Asp = sparse.coo_matrix((A, (self.fem.rows, self.fem.cols)), shape=(nnodes, nnodes)).tocsr()
        t3 = time.time()
        for key, nodes in self.nodesdirflux.items():
            nb = nodes.shape[0]
            # help = np.zeros((nnodes))
            # help[nodes] = 1
            # help = sparse.dia_matrix((help,0), shape=(nnodes,nnodes))
            help = sparse.dok_matrix((nb,nnodes))
            for i in range(nb): help[i, nodes[i]] = 1
            self.Asaved[key] = help.dot(Asp)
        # ndirs = self.nodedirall.shape[0]
        # help = sparse.dia_matrix((np.ones(ndirs), 0), shape=(ndirs, nnodes))
        # self.Asavedall = help.dot(Asp)
        help = np.ones((nnodes))
        help[self.nodedirall] = 0
        help = sparse.dia_matrix((help, 0), shape=(nnodes, nnodes))
        Asp = help.dot(Asp)
        help = np.zeros((nnodes))
        help[self.nodedirall] = 1.0
        help = sparse.dia_matrix((help, 0), shape=(nnodes, nnodes))
        Asp += help
        t4 = time.time()
        self.timer['mat_cells'] = t2-t1
        self.timer['mat_coo'] = t3-t2
        self.timer['mat_bdry'] = t4-t3
        return Asp
    def computeMean(self, u, key, data):
        colors = [int(x) for x in data.split(',')]
        mean, omega = 0, 0
        for color in colors:
            edges = self.mesh.bdrylabels[color]
            for ie in edges:
                normal = self.mesh.normals[ie]
                d = linalg.norm(normal)
                omega += d
                mean += d*np.mean(u[self.mesh.faces[ie]])
        return mean
    def computeFlux(self, u, key, data):
        # colors = [int(x) for x in data.split(',')]
        # omega = 0
        # for color in colors:
        #     omega += np.sum(linalg.norm(self.mesh.normals[self.mesh.bdrylabels[color]],axis=1))
        flux = np.sum(self.bsaved[key] - self.Asaved[key]*u )
        return flux
    def postProcess(self, u):
        info = {}
        cell_data = {}
        point_data = {}
        point_data['U'] = u
        if self.solexact:
            info['error'], point_data['E'] = self.computeError(self.solexact, u)
        info['timer'] = self.timer
        info['runinfo'] = self.runinfo
        info['postproc'] = {}
        for key, val in self.postproc.items():
            type,data = val.split(":")
            if type == "mean":
                info['postproc'][key] = self.computeMean(u, key, data)
            elif type == "flux":
                info['postproc'][key] = self.computeFlux(u, key, data)
            else:
                raise ValueError("unknown postprocess {}".format(key))
        cell_data['k'] = self.kheatcell
        return point_data, cell_data, info
    def computeError(self, solex, uh):
        x, y, z = self.mesh.points[:,0], self.mesh.points[:,1], self.mesh.points[:,2]
        e = solex(x, y, z) - uh
        errors = {}
        errors['L2'] = np.sqrt( np.dot(e, self.massmatrix*e) )
        return errors, e

#=================================================================#
if __name__ == '__main__':
    print("Pas encore de test")
