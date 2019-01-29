assert __name__ == '__main__'
from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import simfempy.applications
import pygmsh
import numpy as np
import scipy.interpolate
import scipy.optimize
import matplotlib.pyplot as plt
from simfempy.tools import npext
from simfempy.meshes import pygmshext



# ----------------------------------------------------------------#
def createMesh2d(h=0.1, hhole=0.05, hmeas=0.02, nmeasurepoints=2):
    geometry = pygmsh.built_in.Geometry()
    xholes = []
    xa, xb = 0.8, 0.1
    ya, yb = -0.8, 0.5
    xholes.append([[-xa, ya, 0], [-xb, ya, 0], [-xb, yb, 0], [-xa, yb, 0]])
    xholes.append([[xa, ya, 0], [xb, ya, 0], [xb, yb, 0], [xa, yb, 0]])
    holes = []
    hole_labels = np.arange(200, 200 + len(xholes), dtype=int)
    for xhole, hole_label in zip(xholes, hole_labels):
        xarrm = np.mean(np.array(xhole), axis=0)
        holes.append(geometry.add_polygon(X=xhole, lcar=hhole))
        pygmshext.add_point_in_surface(geometry, holes[-1].surface, xarrm, lcar=h)
        geometry.add_physical_surface(holes[-1].surface, label=int(hole_label))
    # outer poygon
    # nmeasurepoints = 2
    xmeas = np.linspace(1,-1,nmeasurepoints+2)[1:-1]
    # print("xmeas", xmeas)
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
    bdrycond = simfempy.applications.boundaryconditions.BoundaryConditions(mesh.bdrylabels.keys())
    bdrycond.type[1002] = "Neumann"
    bdrycond.type[1000] = "Dirichlet"
    bdrycond.type[1001] = "Dirichlet"
    bdrycond.type[1003] = "Dirichlet"
    bdrycond.fct[1002] = lambda x, y, z, nx, ny, nz, k: 0
    bdrycond.fct[1000] = lambda x, y, z: 333
    bdrycond.fct[1001] = bdrycond.fct[1003] = lambda x, y, z: 293
    postproc = {}
    postproc['measured'] = "pointvalues:{}".format(','.join( [str(l) for l in pointlabels]))
    print("postproc['measured']",postproc['measured'])
    postproc['bdryfct'] = "bdryfct:1002"
    postproc['meanout'] = "bdrymean:1002"
    postproc['flux1'] = "bdrydn:1001"
    postproc['flux2'] = "bdrydn:1003"
    fluxes = ['flux1', 'flux2', 'meanout']
    fluxes = []
    return mesh, bdrycond, postproc, hole_labels, fluxes



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
        # print("pointsmeas", pointsmeas)
        # print("info['postproc']['measured']", info['postproc']['measured'])
        assert len(pointsmeas) == len(self.info['postproc']['measured'])
        ax.plot(self.heat.mesh.points[pointsmeas,0], self.info['postproc']['measured'], 'Dm', label=r'$C(u)$')
        ax.plot(self.heat.mesh.points[pointsmeas,0], self.heat.data0[len(self.heat.fluxes):], 'vy', label=r'$C_0$')
        ax.legend()
    def plot(self, point_data=None, cell_data=None, info=None):
        if info is None:
            self.point_data, self.cell_data, self.info = self.heat.point_data, self.heat.cell_data, self.heat.info
        else:
            self.point_data, self.cell_data, self.info = point_data, cell_data, info
        fig, axs = simfempy.meshes.plotmesh.meshWithData(self.heat.mesh, self.point_data, self.cell_data, addplots=self.addplots)
        plt.show()


#----------------------------------------------------------------#
class Heat(simfempy.applications.heat.Heat):
    def __init__(self, **kwargs):
        kwargs['fem'] = 'p1'
        kwargs['plotk'] = True
        super().__init__(**kwargs)
        self.linearsolver = "pyamg"
        self.kheat = np.vectorize(self.kparam)
        self.dkheat = np.vectorize(self.dkparam)
        self.diffglobal = kwargs.pop('diffglobal')
        self.hole_labels = kwargs.pop('hole_labels')
        self.hole_labels_inv = {}
        for i in range(len(self.hole_labels)):
            self.hole_labels_inv[int(self.hole_labels[i])] = i
        self.fluxes = kwargs.pop('fluxes')
        self.param = np.ones(len(self.hole_labels))
        self.plotter = Plotter(self)

    def kparam(self, label):
        if label==100: return self.diffglobal
        # return self.param[label-200]
        return self.param[self.hole_labels_inv[label]]
    def dkparam(self, label):
        if label==self.dlabel: return 1.0
        return 0.0
    def matrix(self):
        A = self.fem.matrixDiffusion(self.kheatcell)
        return A
    def getData(self, infopp):
        # dn20, dn40, up = infopp['flux20'], infopp['flux40'], infopp['measured']
        # return np.concatenate( [np.array([dn20,dn40]), up], axis=0)
        return np.concatenate([np.array([infopp[f] for f in self.fluxes]), infopp['measured']], axis=0)
    # def setinitial(self, param):
    #     self.param = param
    #     self.kheatcell = self.kheat(self.mesh.cell_labels)
    #     # self.setMesh(mesh)
    #     point_data, cell_data, info = self.solve()
    #     self.data0 = self.getData(info['postproc'])
    #     # self.plot(point_data, cell_data, info)
    #     # print("self.data0", self.data0)
    def solvestate(self, param):
        # print("#")
        assert param.shape == self.param.shape
        self.param = param
        # print("self.param", self.param)
        self.kheatcell = self.kheat(self.mesh.cell_labels)
        b = self.computeRhs()
        if not hasattr(self, 'ustate'):
            self.ustate = np.zeros_like(b)
        A = self.matrix()
        A,b,self.ustate = self.boundary(A, b, self.ustate)
        self.A = A
        self.ustate = self.linearSolver(A, b, self.ustate, solver=self.linearsolver, verbose=0)
        self.point_data, self.cell_data, self.info = self.postProcess(self.ustate)
        data = self.getData(self.info['postproc'])
        # self.plotter.plot()
        # print("self.data0", self.data0, "data", data)
        return data - self.data0
    def solveDstate(self, param):
        # print("@")
        assert param.shape == self.param.shape
        nparam = param.shape[0]
        jac = np.empty(shape=(self.data0.shape[0],nparam))
        import copy
        bdrycond_bu = copy.deepcopy(self.bdrycond)
        for color in self.bdrycond.fct:
            self.bdrycond.fct[color] = None
        for i in range(nparam):
            self.dlabel = 200 + i
            self.kheatcell = self.dkheat(self.mesh.cell_labels)
            Bi = self.matrix()
            b = -Bi.dot(self.ustate)
            du = np.zeros_like(b)
            self.kheatcell = self.kheat(self.mesh.cell_labels)
            b,du = self.boundaryvec(b, du)
            du = self.linearSolver(self.A, b, du, solver=self.linearsolver, verbose=0)
            point_data, cell_data, info = self.postProcess(du)
            # self.plot(point_data, cell_data, info)
            jac[:,i] = self.getData(info['postproc'])
        self.bdrycond = bdrycond_bu
        # print("jac", jac)
        return jac



#----------------------------------------------------------------#
def compute_j(diffglobal):
    h = 0.2
    nmeasurepoints = 2
    mesh, bdrycond, postproc, hole_labels, fluxes = createMesh2d(h=h, hhole=0.5*h, hmeas=0.2*h)
    simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
    plt.show()
    heat = Heat(bdrycond=bdrycond, postproc=postproc, diffglobal=diffglobal, hole_labels=hole_labels, fluxes=fluxes, method="new")
    heat.setMesh(mesh)
    heat.data0 = np.zeros(nmeasurepoints)

    param = np.ones(2, dtype=float)
    n = 100
    j = np.empty(shape=(2,n))
    ps = np.linspace(0.1*diffglobal, 100*diffglobal, n)
    for i in range(n):
        param[:] = ps[i]
        data = heat.solvestate(param)
        # print("{} --> {}".format(p, heat.data0))
        j[0, i] = data[0]
        j[1, i] = data[1]
    plt.plot(ps/diffglobal, j[0])
    plt.plot(ps/diffglobal, j[1])
    plt.show()
    print("min/max", np.min(j[0]), np.max(j[1]))


#----------------------------------------------------------------#
def compute_j2d(diffglobal):
    h = 0.05
    nmeasurepoints = 2
    mesh, bdrycond, postproc, hole_labels, fluxes = createMesh2d(h=h, hhole=0.5*h, hmeas=0.2*h)
    simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
    plt.show()
    heat = Heat(bdrycond=bdrycond, postproc=postproc, diffglobal=diffglobal, hole_labels=hole_labels, fluxes=fluxes, method="new")
    heat.setMesh(mesh)
    heat.data0 = np.zeros(nmeasurepoints)

    param = np.ones(2, dtype=float)
    n = 40
    cost = np.empty(shape=(2,n,n))
    ps = np.linspace(diffglobal, 100*diffglobal, n)
    for i in range(n):
        for j in range(n):
            param[0] = ps[i]
            param[1] = ps[j]
            data = heat.solvestate(param)
            print("data", data)
            # print("param", param, "data",data)
            cost[0, i, j] = data[0]
            cost[1, i, j] = data[1]
    ps /= diffglobal
    xx, yy = np.meshgrid(ps, ps)
    # for i in range(n):
    #     for j in range(n):
    #         print("x", xx[i,j], "y", yy[i,j], "cost", cost[0, i, j], cost[1, i, j])
    cost = np.round(cost, 4)
    fig, axs = plt.subplots(1, 2,figsize=(10,5), squeeze=False)
    ax = axs[0,0]
    cnt = ax.contourf(xx, yy, cost[0], 16, cmap='jet')
    ax.set_aspect(1)
    clb = plt.colorbar(cnt, ax=ax)
    ax.set_title(r"$C_1(u)$")
    ax = axs[0,1]
    cnt = ax.contourf(xx, yy, cost[1], 16, cmap='jet')
    ax.set_aspect(1)
    clb = plt.colorbar(cnt, ax=ax)
    # clb.ax.set_title(r"$C_2(u)$")
    ax.set_title(r"$C_2(u)$")
    plt.show()
    print("min/max", np.min(cost[0]), np.max(cost[0]))
    print("min/max", np.min(cost[1]), np.max(cost[1]))

#----------------------------------------------------------------#
def test(diffglobal):
    nmeasurepoints = 6
    mesh, bdrycond, postproc, hole_labels, fluxes = createMesh2d(nmeasurepoints=nmeasurepoints)
    # simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
    # plt.show()
    heat = Heat(bdrycond=bdrycond, postproc=postproc, diffglobal=diffglobal, hole_labels=hole_labels, fluxes=fluxes, method="new")
    heat.setMesh(mesh)

    heat.data0 = np.zeros(nmeasurepoints)
    param = np.zeros(len(hole_labels), dtype=float)
    param[0] = 101*diffglobal
    param[1] = diffglobal
    data = heat.solvestate(param)
    heat.data0[:] =  data[:]*(1+0.001* ( 2*np.random.rand()-1))

    methods = ['trf','lm']
    import time
    for method in methods:
        param[:] = diffglobal
        t0 = time.time()
        info = scipy.optimize.least_squares(heat.solvestate, jac=heat.solveDstate, x0=param, method=method, verbose=0)
        dt = time.time()-t0
        print("{:^10s} x = {} J={:10.2e} nf={:4d} nj={:4d} {:10.2f} s".format(method, info.x, info.cost, info.nfev, info.njev, dt))
        heat.plotter.plot()


#================================================================#

diffglobal = 1e-3
# compute_j(diffglobal)
# compute_j2d(diffglobal)
test(diffglobal)
