assert __name__ == '__main__'
from os import sys, path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

import simfempy.applications
import pygmsh
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from simfempy.tools import npext
from simfempy.meshes import pygmshext
import copy

# ----------------------------------------------------------------#
def createMesh2d(h=0.1, hhole=0.05, hmeas=0.02, nmeasurepoints=2, nholes=2):
    geometry = pygmsh.built_in.Geometry()
    nholesy = int(np.sqrt(nholes))
    nholesx = int(nholes/nholesy)
    holes, hole_labels = pygmshext.add_holesnew(geometry, h=h, hhole=hhole, x0=-0.8, x1=0.8, y0=-0.8, y1=0.5, nholesx=nholesx,nholesy=nholesy)
    xmeas = np.linspace(1,-1,nmeasurepoints+2)[1:-1]
    outer = []
    outer.append([[-1, -1, 0], 1000, h])
    outer.append([[1, -1, 0], 1001, h])
    outer.append([[1, 1, 0], 1002, h])
    for xm in xmeas:
        outer.append([[xm, 1, 0], 1002, hmeas])
    outer.append([[-1, 1, 0], 1003, h])
    xouter = [out[0] for out in outer]
    labels = [out[1] for out in outer]
    lcars = [out[2] for out in outer]
    p1 = pygmshext.add_polygon(geometry, xouter, lcar=lcars, holes=holes)
    vals, inds = npext.unique_all(labels)
    for val, ind in zip(vals, inds):
        geometry.add_physical_line([p1.line_loop.lines[i] for i in ind], label=int(val))
    mnum = range(2, 2+nmeasurepoints)
    mpoints = [p1.line_loop.lines[i].points[1] for i in mnum]
    pointlabels = []
    for m, mpoint in zip(mnum, mpoints):
        label = 9998+m
        geometry.add_physical_point(mpoint, label=label)
        pointlabels.append(label)
    geometry.add_physical_surface(p1.surface, label=100)
    # print("code", geometry.get_code())
    data = pygmsh.generate_mesh(geometry, verbose=False)
    mesh = simfempy.meshes.simplexmesh.SimplexMesh(data=data)
    bdrycond = simfempy.applications.problemdata.BoundaryConditions()
    bdrycond.type[1002] = "Neumann"
    bdrycond.type[1000] = "Dirichlet"
    bdrycond.type[1001] = "Dirichlet"
    bdrycond.type[1003] = "Dirichlet"
    bdrycond.fct[1002] = lambda x, y, z, nx, ny, nz: 0
    bdrycond.fct[1000] = lambda x, y, z: 333
    bdrycond.fct[1001] = bdrycond.fct[1003] = lambda x, y, z: 293
    bdrycond.check(mesh.bdrylabels.keys())
    postproc = {}
    postproc['measured'] = "pointvalues:{}".format(','.join( [str(l) for l in pointlabels]))
    postproc['bdryfct'] = "bdryfct:1002"
    # postproc['meanout'] = "bdrymean:1002"
    # postproc['flux1'] = "bdrydn:1001"
    # postproc['flux2'] = "bdrydn:1003"
    # fluxes = ['flux1', 'flux2', 'meanout']
    # fluxes = []
    return mesh, bdrycond, postproc, hole_labels

#----------------------------------------------------------------#
class Plotter:
    def __init__(self, heat):
        self.heat = heat
        self.addplots = [self.plotmeas]
    def plotmeas(self, ax):
        p, ub = self.info['postproc']['bdryfct']
        x = p[:, 0]
        fct = scipy.interpolate.interp1d(x, ub)
        xn = np.linspace(x.min(), x.max(), 50)
        ax.plot(xn, fct(xn), label=r'$u$')
        # ax.plot(x, fct(x), 'r.')
        pointsmeas = self.heat.mesh.vertices
        # print("self.heat.mesh.vertices", self.heat.mesh.vertices)
        # print("self.heat.mesh.points[pointsmeas,0]", self.heat.mesh.points[pointsmeas,0])
        # print("info['postproc']['measured']", info['postproc']['measured'])
        assert len(pointsmeas) == len(self.info['postproc']['measured'])
        ax.plot(self.heat.mesh.points[pointsmeas,0], self.info['postproc']['measured'], 'Dm', label=r'$C(u)$')
        ax.plot(self.heat.mesh.points[pointsmeas,0], self.heat.data0+293, 'vy', label=r'$C_0$')
        ax.legend()
    def plot(self, point_data=None, cell_data=None, info=None):
        if point_data is None:
            point_data, cell_data = self.heat.point_data, self.heat.cell_data
        if info is None:
            addplots = None
        else:
            self.info = info
            addplots = [self.plotmeas]
        fig, axs = simfempy.meshes.plotmesh.meshWithData(self.heat.mesh, point_data, cell_data, addplots=addplots)
        plt.show()

#----------------------------------------------------------------#
class Heat(simfempy.applications.heat.Heat):
    def __init__(self, **kwargs):
        kwargs['fem'] = 'p1'
        kwargs['plotk'] = True
        super().__init__(**kwargs)
        self.linearsolver = "pyamg"
        # self.linearsolver = "umf"
        self.kheat = np.vectorize(self.kparam)
        self.dkheat = np.vectorize(self.dkparam)
        self.diffglobal = kwargs.pop('diffglobal')
        self.hole_labels = kwargs.pop('hole_labels')
        self.hole_labels_inv = {}
        for i in range(len(self.hole_labels)):
            self.hole_labels_inv[int(self.hole_labels[i])] = i
        # self.fluxes = kwargs.pop('fluxes')
        self.nparam = len(self.hole_labels)
        self.param = np.ones(self.nparam)
        self.plotter = Plotter(self)
        pp = self.problemdata.postproc['measured'].split(":")[1]
        self.nmeasures = len(pp.split(","))
        self.data0 = np.zeros(self.nmeasures)

    def kparam(self, label):
        if label==100: return self.diffglobal
        # return self.param[label-200]
        return self.param[self.hole_labels_inv[label]]
    def dkparam(self, label):
        if label==self.dlabel: return 1.0
        return 0.0

    def getData(self, infopp):
        return infopp['measured'] - 293
        # return np.concatenate([np.array([infopp[f] for f in self.fluxes]), infopp['measured']], axis=0)

    def computeRes(self, param, u=None):
        # print("#")
        self.param = param
        self.kheatcell = self.kheat(self.mesh.cell_labels)
        A = self.matrix()
        b,u= self.computeRhs(u)
        self.A = A
        u, iter = self.linearSolver(A, b, u, solver=self.linearsolver, verbose=0)
        self.point_data, self.cell_data, self.info = self.postProcess(u)
        data = self.getData(self.info['postproc'])
        # self.plotter.plot()
        # print("self.data0", self.data0, "data", data)
        return data - self.data0, u

    def computeDRes(self, param, u, du):
        assert self.data0.shape[0] == self.nmeasures
        jac = np.zeros(shape=(self.nmeasures, self.nparam))
        bdrycond_bu = copy.deepcopy(self.problemdata.bdrycond)
        for color in self.problemdata.bdrycond.fct:
            self.problemdata.bdrycond.fct[color] = None
        if du is None: du = self.nparam*[np.empty(0)]
        for i in range(self.nparam):
            self.dlabel = self.hole_labels[i]
            self.kheatcell = self.dkheat(self.mesh.cell_labels)
            Bi = self.matrix()
            b = -Bi.dot(u)
            du[i] = np.zeros_like(b)
            self.kheatcell = self.kheat(self.mesh.cell_labels)
            b,du[i] = self.boundaryvec(b, du[i])
            du[i], iter = self.linearSolver(self.A, b, du[i], solver=self.linearsolver, verbose=0)
            point_data, cell_data, info = self.postProcess(du[i])
            # print("info['postproc'].shape",self.getData(info['postproc']).shape)
            # print("jac.shape",jac.shape)
            # self.plot(point_data, cell_data, info)
            jac[:self.nmeasures,i] = self.getData(info['postproc'])+293
        self.problemdata.bdrycond = bdrycond_bu
        return jac, du

    def computeDResAdjW(self, param, u):
        assert self.data0.shape[0] == self.nmeasures
        jac = np.zeros(shape=(self.nmeasures, self.nparam))
        pdsplit = self.problemdata.postproc['measured'].split(':')
        assert pdsplit[0] == 'pointvalues'
        pointids = [int(l) for l in pdsplit[1].split(',')]
        problemdata_bu = copy.deepcopy(self.problemdata)
        self.problemdata.clear()
        if not hasattr(self,'w'): self.w = self.nmeasures*[np.empty(0)]
        self.problemdata.rhspoint = {}
        for j in range(self.nmeasures):
            for k in range(self.nmeasures):
                if k==j: self.problemdata.rhspoint[pointids[k]] = simfempy.solvers.optimize.RhsParam(1)
                else: self.problemdata.rhspoint[pointids[k]] = None
            self.kheatcell = self.kheat(self.mesh.cell_labels)
            if self.w[j].shape[0]==0:
                self.w[j] = np.zeros(self.mesh.nnodes)
            b, self.w[j] = self.computeRhs(self.w[j])
            self.w[j], iter = self.linearSolver(self.A, b, self.w[j], solver=self.linearsolver, verbose=0)
            # point_data, cell_data, info = self.postProcess(self.w[j])
            # self.plotter.plot(point_data, cell_data)
            for i in range(self.nparam):
                self.dlabel = self.hole_labels[i]
                self.kheatcell = self.dkheat(self.mesh.cell_labels)
                Bi = self.matrix()
                jac[j, i] = -Bi.dot(u).dot(self.w[j])
        self.problemdata = problemdata_bu
        return jac

    def computeAdj(self, param, r, u, z):
        pdsplit = self.problemdata.postproc['measured'].split(':')
        assert pdsplit[0] == 'pointvalues'
        pointids = [int(l) for l in pdsplit[1].split(',')]
        problemdata_bu = copy.deepcopy(self.problemdata)
        self.problemdata.clear()
        if z is None:
            z = np.zeros(self.mesh.nnodes)
        self.problemdata.rhspoint = {}
        for j in range(self.nmeasures):
            self.problemdata.rhspoint[pointids[j]] = simfempy.solvers.optimize.RhsParam(r[j])
        self.kheatcell = self.kheat(self.mesh.cell_labels)
        b, z = self.computeRhs(z)
        z, iter = self.linearSolver(self.A, b, z, solver=self.linearsolver, verbose=0)
        # point_data, cell_data, info = self.postProcess(self.z)
        # self.plotter.plot(point_data, cell_data)
        self.problemdata = problemdata_bu
        return z

    def computeDResAdj(self, param, r, u, z):
        z = self.computeAdj(param, r, u, z)
        grad = np.zeros(shape=(self.nparam))
        for i in range(self.nparam):
            self.dlabel = self.hole_labels[i]
            self.kheatcell = self.dkheat(self.mesh.cell_labels)
            Bi = self.matrix()
            grad[i] = -Bi.dot(u).dot(z)
        return grad, z

    def computeM(self, param, du, z):
        M = np.zeros(shape=(self.nparam,self.nparam))
        assert z is not None
        for i in range(self.nparam):
            self.dlabel = self.hole_labels[i]
            self.kheatcell = self.dkheat(self.mesh.cell_labels)
            Bi = self.matrix()
            for j in range(self.nparam):
                M[i,j] = -Bi.dot(du[j]).dot(z)
        # print("M", np.array2string(M, precision=2, floatmode='fixed'))
        return M

#----------------------------------------------------------------#
def compute_j(diffglobal):
    nmeasurepoints = 6
    nholes = 2
    h = 0.2
    hhole, hmeas = 0.5*h, 0.1*h
    mesh, bdrycond, postproc, hole_labels = createMesh2d(h=h, hhole=hhole, hmeas=hmeas,nmeasurepoints=nmeasurepoints, nholes=nholes)
    problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc)
    heat = Heat(problemdata=problemdata, diffglobal=diffglobal, hole_labels=hole_labels, method="new")
    heat.setMesh(mesh)
    heat.data0 = np.zeros(nmeasurepoints)

    param = np.ones(2, dtype=float)
    n = 100
    c = np.empty(shape=(n,nmeasurepoints))
    ps = np.linspace(diffglobal, 100*diffglobal, n)
    for i in range(n):
        param[::2] = 2*ps[i]
        param[1::2] = ps[i]
        data, u = heat.computeRes(param)
        # print("{} --> {}".format(p, heat.data0))
        c[i,:] = data
    fig, axs = plt.subplots(1, 3, figsize=(12,4), squeeze=False)
    for j in range(nmeasurepoints):
        axs[0,0].plot(ps/diffglobal, c[:,j], label ="meas_{}".format(j))
    axs[0,0].legend()
    res = c-np.mean(c,axis=0)
    Jhat = np.einsum('ij,ij->i', res, res)
    axs[0,1].plot(ps / diffglobal, Jhat, label="J".format(j))
    axs[0,1].legend()
    axs[0,1].set_title("LS with respect to mean")
    simfempy.meshes.plotmesh.plotmesh(mesh, ax=axs[0,2], title='Mesh with measures')
    plt.suptitle("Even holes have double diffusion")
    plt.show()

#----------------------------------------------------------------#
def compute_j2d(diffglobal):
    nmeasurepoints = 2
    nholes = 2
    h = 0.2
    hhole, hmeas = 0.5*h, 0.1*h
    mesh, bdrycond, postproc, hole_labels = createMesh2d(h=h, hhole=hhole, hmeas=hmeas,nmeasurepoints=nmeasurepoints, nholes=nholes)
    problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc)
    heat = Heat(problemdata=problemdata, diffglobal=diffglobal, hole_labels=hole_labels, method="new")
    heat.setMesh(mesh)
    heat.data0 = np.zeros(nmeasurepoints)

    param = np.ones(2, dtype=float)
    n = 20
    c = np.empty(shape=(n,n,nmeasurepoints))
    ps = np.linspace(diffglobal, 100*diffglobal, n)
    for i in range(n):
        for j in range(n):
            param[0] = ps[i]
            param[1] = ps[j]
            data, u = heat.computeRes(param)
            # print("data", data)
            # print("param", param, "data",data)
            c[i,j] = data
    ps /= diffglobal
    xx, yy = np.meshgrid(ps, ps)
    fig, axs = plt.subplots(1, 3,figsize=(12,4), squeeze=False)
    for i in range(2):
        ax = axs[0,i]
        cnt = ax.contourf(xx, yy, c[:,:,i], 16, cmap='jet')
        ax.set_aspect(1)
        clb = plt.colorbar(cnt, ax=ax)
        ax.set_title(r"$c_{}(u)$".format(i))
    res = c-np.mean(c,axis=(0,1))
    print("np.mean(c,axis=(1,2))", np.mean(c,axis=(1,2)).shape)
    print("res", res.shape)
    Jhat = np.einsum('ijk,ijk->ij', res, res)
    CS = axs[0, 2].contour(ps, ps, Jhat, levels=np.linspace(0.,1.,8))
    axs[0, 2].clabel(CS, inline=1, fontsize=10)
    axs[0, 2].set_title('LS with respect to mean')
    plt.suptitle("Two-dim parameters")
    plt.show()

#----------------------------------------------------------------#
def testholes(diffglobal, nholes):
    nmeasurepoints = 6
    h = 0.1
    hhole, hmeas = 0.5*h, 0.1*h
    mesh, bdrycond, postproc, hole_labels = createMesh2d(h=h, hhole=hhole, hmeas=hmeas,nmeasurepoints=nmeasurepoints, nholes=nholes)
    simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
    plt.show()
    regularize = 0.000
    problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc)
    heat = Heat(problemdata=problemdata, diffglobal=diffglobal, hole_labels=hole_labels, method="new")
    heat.setMesh(mesh)

    optimizer = simfempy.solvers.optimize.Optimizer(heat, nparam=nholes, nmeasure=nmeasurepoints, regularize=regularize, param0=diffglobal*np.ones(nholes))

    # heat.data0 = np.zeros(nmeasurepoints)
    refparam = diffglobal*np.ones(nholes, dtype=float)
    refparam[::2] *= 200
    refparam[1::2] *= 100
    print("refparam",refparam)
    percrandom = 0.
    refdata, perturbeddata = optimizer.create_data(refparam=refparam, percrandom=percrandom)
    # perturbeddata[::2] *= 1.2
    # perturbeddata[1::2] *= 0.8

    # heat.data0 =  perturbeddata

    initialparam = diffglobal*np.ones(nholes)
    print("initialparam",initialparam)

    # optimizer.gradtest = True
    # for method in optimizer.methods:
    for method in optimizer.minmethods:
        optimizer.minimize(x0=initialparam, method=method)
        # heat.plotter.plot(info=heat.info)

#================================================================#

diffglobal = 1e-3
# compute_j(diffglobal)
# compute_j2d(diffglobal)
testholes(diffglobal, nholes=4)
