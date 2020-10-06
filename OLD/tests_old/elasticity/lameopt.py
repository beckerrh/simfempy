assert __name__ == '__main__'
from os import sys, path
import numpy as np
import copy
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

import matplotlib.pyplot as plt
import pygmsh
from OLD.simfempy.meshes import pygmshext


#================================================================#
def mesh_traction(h, dim=3, nmeasure=4):
    geometry = pygmsh.built_in.Geometry()
    postproc = {}
    bdrycond =  OLD.simfempy.applications.problemdata.BoundaryConditions()
    if dim==2:
        ncomp = 2
        bdrycond.type[1000] = "Neumann"
        bdrycond.type[1001] = "Neumann"
        bdrycond.type[1002] = "Neumann"
        bdrycond.type[1003] = "Dirichlet"
        bdrycond.fct[1001] = lambda x, y, z, nx, ny, nz: np.array([10,0])
        geometry = OLD.simfempy.meshes.geomdefs.unitsquare.Unitsquare(x, y)
    elif dim==3:
        ncomp = 3
        bdrycond.type[100] = "Neumann"
        bdrycond.type[101] = "Neumann"
        bdrycond.type[102] = "Neumann"

        x, y, z = [-1, 1], [0, 1], [-1, 1]
        bdrycond.fct[102] = lambda x, y, z, nx, ny, nz: np.array([30,0,0])

        x, y, z = [-1, 1], [-0.5, 0.5], [-0.5, 0.5]
        bdrycond.fct[102] = lambda x, y, z, nx, ny, nz: np.array([0,1,0])

        def torsion(x,y,z, nx, ny, nz):
            scale = 0.01
            r = 10*np.sqrt(y**2+z**2)
            theta = np.arctan2(y,z)
            return [0, scale*r*np.cos(theta), -scale*r*np.sin(theta) ]
        bdrycond.fct[102] = torsion

        bdrycond.type[103] = "Neumann"
        bdrycond.type[104] = "Dirichlet"
        bdrycond.type[105] = "Neumann"

        p = geometry.add_rectangle(xmin=x[0], xmax=x[1], ymin=y[0], ymax=y[1], z=z[0], lcar=h)
        geometry.add_physical(p.surface, label=100)
        axis = [0, 0, z[1]-z[0]]
        top, vol, ext = geometry.extrude(p.surface, axis)
        # print ('vol', vars(vol))
        # print ('top', vars(top))
        # print ('top.id', top.id)
        # print ('ext[0]', vars(ext[0]))
        geometry.add_physical(top, label=105)
        geometry.add_physical(ext[0], label=101)
        geometry.add_physical(ext[1], label=102)
        geometry.add_physical(ext[2], label=103)
        geometry.add_physical(ext[3], label=104)
        geometry.add_physical_volume(vol, label=10)
        nmeasurey = int(np.sqrt(nmeasure))
        nmeasurez = int(nmeasure / nmeasurey)
        if nmeasurey==1: py = [0.5]
        else: py = np.linspace(0.2,0.8, nmeasurey, endpoint=True)
        if nmeasurez==1: pz = [0.5]
        else: pz = np.linspace(0.2,0.8, nmeasurez, endpoint=True)
        # print("py", py, "pz", pz)
        hpoint = 0.05*h
        hpoint = h
        pointlabels = []
        for iy in range(nmeasurey):
            for iz in range(nmeasurez):
                X = (x[1], py[iy]*y[0]+(1-py[iy])*y[1], pz[iz]*z[0]+(1-pz[iz])*z[1])
                label = 10000+iy+iz*nmeasurey
                pointlabels.append(label)
                # print("label", label)
                pygmshext.add_point_in_surface(geometry, surf=ext[1], X=X, lcar=hpoint, label=label)
        postproc['measured'] = "pointvalues:{}".format(','.join( [str(l) for l in pointlabels]))
    else:
        raise ValueError("unknown dim={}".format(dim))
    mesh = pygmsh.generate_mesh(geometry, verbose=False)
    mesh = OLD.simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)
    bdrycond.check(mesh.bdrylabels.keys())
    # mesh.plot(title='Mesh with measures')
    # mesh.plotWithBoundaries()

    problemdata = OLD.simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc, ncomp=ncomp)
    return mesh, problemdata

#----------------------------------------------------------------#
class Elasticity(OLD.simfempy.applications.elasticity.Elasticity):
    def __init__(self, **kwargs):
        kwargs['fem'] = 'p1'
        kwargs['plotk'] = True
        super().__init__(**kwargs)
        self.linearsolver = "pyamg"
        self.linearsolver = "umf"
        self.nparam = 2

    def plot(self, **kwargs):
        if not 'point_data' in kwargs: point_data = self.point_data
        else: point_data = kwargs.pop('point_data')
        if not 'cell_data' in kwargs: cell_data = self.cell_data
        else: cell_data = kwargs.pop('cell_data')
        self.mesh.plotWithData(point_data=point_data, cell_data=cell_data)
        plt.show()

    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.Ais = [np.empty(shape=(0,0)) for i in range(self.nparam)]
        self.computeAis()

    def computeAis(self):
        bdrycond_bu = copy.deepcopy(self.problemdata.bdrycond)
        for color in self.problemdata.bdrycond.fct:
            self.problemdata.bdrycond.fct[color] = None
            self.problemdata.bdrycond.param[color] = 0
        for i in range(self.nparam):
            if i==0:
                self.setParameters(mu=1, lam=0)
            elif i==1:
                self.setParameters(mu=0, lam=1)
            else: raise ValueError("too many parameters")
            self.Ais[i] = self.matrix()
        self.problemdata.bdrycond = bdrycond_bu

    def computeRes(self, param, u=None):
        self.setParameters(*param)
        # print("self.mu", self.mu, "self.lam", self.lam)
        A = self.matrix()
        b, u = self.computeRhs(u)
        u, niter = self.linearSolver(A, b, u, solver=self.linearsolver)
        self.point_data, self.cell_data, info = self.postProcess(u)
        # self.mesh.plotWithData(point_data=point_data, translate_point_data=1)
        data = info['postproc']['measured'].reshape(-1)
        self.nmeasure = data.shape[0]
        # if not hasattr(self,'data0'): self.data0 = np.zeros_like(data)
        # self.mesh.plotWithData(point_data=point_data, translate_point_data=1)
        # print("self.data0", self.data0, "data", data)
        # return data - self.data0, u
        return data, u

    def computeDRes(self, param, u, du):
        nparam = param.shape[0]
        nmeasure = self.nmeasure
        jac = np.zeros(shape=(nmeasure, nparam))
        bdrycond_bu = copy.deepcopy(self.problemdata.bdrycond)
        for color in self.problemdata.bdrycond.fct:
            self.problemdata.bdrycond.fct[color] = None
        if du is None: du = nparam*[np.empty(0)]
        for i in range(nparam):
            # if i==0:
            #     self.setParameters(mu=1, lam=0)
            # elif i==1:
            #     self.setParameters(mu=0, lam=1)
            # else: raise ValueError("too many parameters")
            # Bi = self.matrix()
            # b = -Bi.dot(u)
            b = -self.Ais[i].dot(u)
            du[i] = np.zeros_like(b)
            self.setParameters(*param)
            A = self.matrix()
            b,du[i] = self.vectorDirichlet(b, du[i])
            du[i], iter = self.linearSolver(A, b, du[i], solver=self.linearsolver)
            point_data, cell_data, info = self.postProcess(du[i])
            # print("info['postproc'].shape",self.getData(info['postproc']).shape)
            # print("jac.shape",jac.shape)
            # self.plot(point_data, cell_data, info)
            jac[:,i] = info['postproc']['measured'].reshape(-1)
        self.problemdata.bdrycond = bdrycond_bu
        return jac, du

    def computeAdj(self, param, r, u, z):
        self.param = param
        problemdata_bu = copy.deepcopy(self.problemdata)
        self.problemdata.clear()
        if z is None: z = np.zeros(3*self.mesh.nnodes)
        assert r.shape[0] == self.nmeasures
        for i, label in enumerate(self.measure_labels):
            self.problemdata.bdrycond.fct[label] = OLD.simfempy.solvers.optimize.RhsParam(r[i])
        self.diffcellinv = self.diffinv(self.mesh.cell_labels)
        self.diffcell = 1/self.diffcellinv
        b, z = self.computeRhs(z)
        z, iter = self.linearSolver(self.A, b, z, solver=self.linearsolver, verbose=0)
        # point_data, cell_data, info = self.postProcess(z)
        # self.plotter.plot(point_data, cell_data)
        self.problemdata = problemdata_bu
        return z

    def computeM2(self, param, du, z):
        self.param = param
        M = np.zeros(shape=(self.nparam,self.nparam))
        assert z is not None
        for i in range(self.nparam):
            for j in range(self.nparam):
                M[j, i] -= self.Ais[i].dot(du[j][:self.mesh.nfaces]).dot(z[:self.mesh.nfaces])
                M[i, j] -= self.Ais[i].dot(du[j][:self.mesh.nfaces]).dot(z[:self.mesh.nfaces])
        # print("M", np.array2string(M, precision=2, floatmode='fixed'))
        return M


#================================================================#
def test_plot():
    hmean = 0.1
    # hmean = 0.8
    nmeasure = 1
    mesh, problemdata = mesh_traction(hmean, nmeasure=nmeasure)

    elasticity = Elasticity(problemdata=problemdata)
    elasticity.setMesh(mesh)

    optimizer = OLD.simfempy.solvers.optimize.Optimizer(elasticity, nparam=2, nmeasure=3 * nmeasure)

    refparam = elasticity.material2Lame("Acier")
    # refparam = elasticity.material2Lame("Caoutchouc")
    initialparam = elasticity.material2Lame("Bois")
    print("refparam", refparam)
    print("initialparam",initialparam)
    percrandom = 0.0
    optimizer.create_data(refparam=refparam, percrandom=percrandom, plot=True)
    # print("refdata", refdata)
    # print("perturbeddata", perturbeddata)


    # optimizer.gradtest = True
    bounds = False
    bounds = True
    if bounds:
        from scipy.optimize import Bounds
        bounds =  Bounds([0.001, 0.001],[np.inf, np.inf])
        methods = optimizer.boundmethods
    else:
        bounds = None
        methods = optimizer.methods
    # methods = []
    # methods = ['lm']
    methods.remove('TNC')
    methods.remove('L-BFGS-B')
    values, valformat, xall = optimizer.testmethods(x0=initialparam, methods=methods, bounds=bounds, plot=True, verbose=0)
    latex = OLD.simfempy.tools.latexwriter.LatexWriter(filename="mincompare")
    latex.append(n=methods, nname='method', nformat="20s", values=values, valformat=valformat)
    latex.write()
    latex.compile()

#================================================================#

test_plot()