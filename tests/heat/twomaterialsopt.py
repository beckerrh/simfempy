assert __name__ == '__main__'
from os import sys, path
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import fempy.applications
import pygmsh
import numpy as np
import scipy.interpolate
import scipy.optimize
import matplotlib.pyplot as plt
from fempy.tools import npext


#----------------------------------------------------------------#
def add_polygon(geom, X, lcar=None, holes=None, make_surface=True):
    assert len(X) == len(lcar)
    if holes is None:
        holes = []
    else:
        assert make_surface

    # Create points.
    p = [geom.add_point(x, lcar=l) for x,l in zip(X,lcar)]
    # Create lines
    lines = [geom.add_line(p[k], p[k + 1]) for k in range(len(p) - 1)]
    lines.append(geom.add_line(p[-1], p[0]))
    ll = geom.add_line_loop((lines))
    surface = geom.add_plane_surface(ll, holes) if make_surface else None

    return geom.Polygon(ll, surface)
    return geom.Polygon(ll, surface, lcar=lcar)

#----------------------------------------------------------------#
def add_point_in_surface(geom, surf, X, lcar, label=None):
    p = geom.add_point(X, lcar=lcar)
    geom.add_raw_code("Point {{{}}} In Surface {{{}}};".format(p.id, surf.id))
    if label:
        geom.add_physical_point(p, label=label)


# ----------------------------------------------------------------#
def createMesh2d(h=0.1, hhole=0.05, hmeas=0.02):
    geometry = pygmsh.built_in.Geometry()
    xholes = []
    xa, xb = 0.75, 0.1
    ya, yb = -0.8, 0.1
    xholes.append([[-xa, ya, 0], [-xb, ya, 0], [-xb, yb, 0], [-xa, yb, 0]])
    xholes.append([[xa, ya, 0], [xb, ya, 0], [xb, yb, 0], [xa, yb, 0]])
    holes = []
    hole_labels = np.arange(200, 200 + len(xholes), dtype=int)
    for xhole, hole_label in zip(xholes, hole_labels):
        xarrm = np.mean(np.array(xhole), axis=0)
        holes.append(geometry.add_polygon(X=xhole, lcar=hhole))
        add_point_in_surface(geometry, holes[-1].surface, xarrm, lcar=h)
        geometry.add_physical_surface(holes[-1].surface, label=int(hole_label))
    # outer poygon
    nmeasurepoints = 2
    xmeas = np.linspace(1,-1,nmeasurepoints+2)[1:-1]
    # print("xmeas", xmeas)
    outer = []
    outer.append([[-1, -1, 0], 10, h])
    outer.append([[1, -1, 0], 20, h])
    outer.append([[1, 1, 0], 30, h])
    for xm in xmeas:
        outer.append([[xm, 1, 0], 30, hmeas])
    # outer.append([[0.5, 1, 0], 30, hmeas])
    # outer.append([[0, 1, 0], 30, hmeas])
    # outer.append([[-0.5, 1, 0], 30, hmeas])
    outer.append([[-1, 1, 0], 40, h])
    xouter = [out[0] for out in outer]
    labels = [out[1] for out in outer]
    lcars = [out[2] for out in outer]
    p1 = add_polygon(geometry, xouter, lcar=lcars, holes=holes)
    vals, inds = npext.unique_all(labels)
    for val, ind in zip(vals, inds):
        geometry.add_physical_line([p1.line_loop.lines[i] for i in ind], label=int(val))

    mnum = range(2, 2+nmeasurepoints)
    mpoints = [p1.line_loop.lines[i].points[1] for i in mnum]
    for m, mpoint in zip(mnum, mpoints):
        geometry.add_physical_point(mpoint, label=m - 1)
    geometry.add_physical_surface(p1.surface, label=100)

    # print("code", geometry.get_code())
    data = pygmsh.generate_mesh(geometry, verbose=False)
    mesh = fempy.meshes.simplexmesh.SimplexMesh(data=data)
    bdrycond = fempy.applications.boundaryconditions.BoundaryConditions(mesh.bdrylabels.keys())
    bdrycond.type[30] = "Neumann"
    bdrycond.type[10] = "Dirichlet"
    bdrycond.type[20] = "Dirichlet"
    bdrycond.type[40] = "Dirichlet"
    bdrycond.fct[30] = lambda x, y, z, nx, ny, nz, k: 0
    bdrycond.fct[10] = lambda x, y, z: 333
    bdrycond.fct[20] = lambda x, y, z: 293
    bdrycond.fct[40] = bdrycond.fct[20]
    postproc = {}
    postproc['measured'] = "pointvalues:{}".format(','.join(f"{i}" for i in range(1,nmeasurepoints+1)))
    postproc['bdryfct'] = "bdryfct:30"
    postproc['mean30'] = "bdrymean:30"
    postproc['flux1'] = "bdrydn:20"
    postproc['flux2'] = "bdrydn:40"
    fluxes = ['flux1', 'flux2', 'mean30']
    # fluxes = []
    return mesh, bdrycond, postproc, hole_labels, fluxes


#----------------------------------------------------------------#
class Heat(fempy.applications.heat.Heat):
    def __init__(self, **kwargs):
        kwargs['fem'] = 'p1'
        kwargs['plotk'] = True
        super().__init__(**kwargs)
        # self.linearsolver = "pyamg"
        self.kheat = np.vectorize(self.kparam)
        self.dkheat = np.vectorize(self.dkparam)
        self.diffglobal = kwargs.pop('diffglobal')
        self.hole_labels = kwargs.pop('hole_labels')
        self.hole_labels_inv = {}
        for i in range(len(self.hole_labels)):
            self.hole_labels_inv[int(self.hole_labels[i])] = i
        self.fluxes = kwargs.pop('fluxes')

    def plot(self, point_data, cell_data, info):
        # print("time: {}".format(info['timer']))
        # print("postproc: {}".format(info['postproc']))
        fig, axs = fempy.meshes.plotmesh.meshWithData(self.mesh, point_data, cell_data)
        fig.show()
        # plt.show()
        p, ub = info['postproc']['bdryfct']
        x = p[:, 0]
        fct = scipy.interpolate.interp1d(x, ub)
        xn = np.linspace(x.min(), x.max(), 50)
        fig2 = plt.figure(2)
        plt.plot(xn, fct(xn))
        plt.plot(x, fct(x), 'r.')
        pointsmeas = self.mesh.vertices
        # print("pointsmeas", pointsmeas)
        # print("info['postproc']['measured']", info['postproc']['measured'])
        assert len(pointsmeas) == len(info['postproc']['measured'])
        plt.plot(self.mesh.points[pointsmeas,0], info['postproc']['measured'], 'Dm')
        plt.plot(self.mesh.points[pointsmeas,0], self.infopostproc_0['measured'], 'vy')
        fig2.show()
        plt.show()
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
    def setinitial(self, mesh, param):
        self.param = param
        self.setMesh(mesh)
        point_data, cell_data, info = self.solve()
        self.infopostproc_0 = info['postproc']
        self.data0 = self.getData(info['postproc'])
        # self.plot(point_data, cell_data, info)
        # print("self.data0", self.data0)
    def solvestate(self, param):
        # print("#")
        assert param.shape == self.param.shape
        self.param = param
        # print("self.param", self.param)
        self.kheatcell = self.kheat(self.mesh.cell_labels)
        b = self.computeRhs()
        u = np.zeros_like(b)
        A = self.matrix()
        A,b,u = self.boundary(A, b, u)
        self.A = A
        self.ustate = self.linearSolver(A, b, u, solver=self.linearsolver)
        point_data, cell_data, info = self.postProcess(self.ustate)
        data = self.getData(info['postproc'])
        # self.plot(point_data, cell_data, info)
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
            du = self.linearSolver(self.A, b, du, solver=self.linearsolver)
            point_data, cell_data, info = self.postProcess(du)
            # self.plot(point_data, cell_data, info)
            jac[:,i] = self.getData(info['postproc'])
        self.bdrycond = bdrycond_bu
        # print("jac", jac)
        return jac



#----------------------------------------------------------------#
def compute_j():
    mesh, bdrycond, postproc, hole_labels, fluxes = createMesh2d()
    fempy.meshes.plotmesh.meshWithBoundaries(mesh)
    plt.show()
    diffglobal = 1
    heat = Heat(bdrycond=bdrycond, postproc=postproc, diffglobal=diffglobal, hole_labels=hole_labels, fluxes=fluxes, method="new")

    param = np.ones(2, dtype=float)
    n = 100
    j = np.empty(shape=(2,n))
    ps = np.linspace(0.1*diffglobal, 100*diffglobal, n)
    for i in range(n):
        param[:] = ps[i]
        heat.setinitial(mesh, param)
        # print("{} --> {}".format(p, heat.data0))
        j[0, i] = heat.data0[0]
        j[1, i] = heat.data0[3]
    plt.plot(ps/diffglobal, j[0])
    plt.show()

#----------------------------------------------------------------#
def test():
    mesh, bdrycond, postproc, hole_labels, fluxes = createMesh2d()
    fempy.meshes.plotmesh.meshWithBoundaries(mesh)
    plt.show()
    diffglobal = 1e-6
    heat = Heat(bdrycond=bdrycond, postproc=postproc, diffglobal=diffglobal, hole_labels=hole_labels, fluxes=fluxes, method="new")


    param[0] = 0.3
    param[1] = diffglobal
    heat.setinitial(mesh, param)

    methods = ['trf','lm']
    import time
    for method in methods:
        param[:] = diffglobal
        t0 = time.time()
        info = scipy.optimize.least_squares(heat.solvestate, jac=heat.solveDstate, x0=param, method=method, verbose=0)
        dt = time.time()-t0
        print("{:^10s} x = {} J={:10.2e} nf={:4d} nj={:4d} {:10.2f} s".format(method, info.x, info.cost, info.nfev, info.njev, dt))


#================================================================#

# test()
compute_j()
