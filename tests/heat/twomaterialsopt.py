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
def createMesh(h=0.1, hhole=0.2, hmeas=0.02):
    # a, b = 0.75, 0.1
    # holes.append([ [-a,-a], [-b,-a], [-b, -b], [-a, -b] ])
    # holes.append([ [a,-a], [b,-a], [b, -b], [a, -b] ])
    # holes.append([ [-a,a], [-b,a], [-b, b], [-a, b] ])
    # holes.append([ [a,a], [b,a], [b, b], [a, b] ])
    # from fempy.meshes.geomdefs import unitsquareholes
    # geometry = unitsquareholes.define_geometry(h=h, holes=holes)

    geometry = pygmsh.built_in.Geometry()
    xholes = []
    a, b = 0.75, 0.1
    xholes.append([ [-a,-a,0], [-b,-a,0], [-b, a,0], [-a, a,0] ])
    xholes.append([ [a,-a,0], [b,-a,0], [b, a,0], [a, a,0] ])
    holes=[]
    for i,xhole in enumerate(xholes):
        holes.append(geometry.add_polygon(X=xhole, lcar=hhole))
        geometry.add_physical_surface(holes[i].surface, label=200+i)
    # outer poygon
    xouter, labels, lcars = [], [], []
    xouter.append([-1,-1,0])
    labels.append(10)
    lcars.append(h)
    xouter.append([1,-1,0])
    labels.append(20)
    lcars.append(h)
    xouter.append([1,1,0])
    labels.append(30)
    lcars.append(h)
    xouter.append([0.5,1,0])
    labels.append(31)
    lcars.append(hmeas)
    xouter.append([0,1,0])
    labels.append(32)
    lcars.append(hmeas)
    xouter.append([-0.5,1,0])
    labels.append(33)
    lcars.append(hmeas)
    xouter.append([-1,1,0])
    labels.append(40)
    lcars.append(h)
    # p1 = geometry.add_polygon(xouter, lcar=h, holes=holes)
    p1 = add_polygon(geometry, xouter, lcar=lcars, holes=holes)
    mnum = range(2,5)
    mpoints = [p1.line_loop.lines[i].points[1] for i in mnum]
    for m,mpoint in zip(mnum,mpoints):
        old = "0, {}}};".format(h)
        new = "0, {}}};".format(hmeas)
        mpoint.code = mpoint.code.replace(old, new)
        print("mpoint", mpoint.code, old, new)
        mpoint.lcar = 0.01
        geometry.add_physical_point(mpoint, label=m-1)
    # geometry.add_physical_point(p1.line_loop.lines[2].points[1], label=1)
    # geometry.add_physical_point(p1.line_loop.lines[3].points[1], label=2)
    # geometry.add_physical_point(p1.line_loop.lines[4].points[1], label=3)
    geometry.add_physical_surface(p1.surface, label=100)
    for i in range(len(xouter)): geometry.add_physical_line(p1.line_loop.lines[i], label=labels[i])
    print("code", geometry.get_code())
    data = pygmsh.generate_mesh(geometry, verbose=False)
    mesh = fempy.meshes.simplexmesh.SimplexMesh(data=data)
    bdrycond =  fempy.applications.boundaryconditions.BoundaryConditions(mesh.bdrylabels.keys())
    bdrycond.type[30] = "Neumann"
    bdrycond.type[10] = "Dirichlet"
    bdrycond.type[20] = "Dirichlet"
    bdrycond.type[40] = "Dirichlet"
    bdrycond.fct[30] = lambda x,y,z, nx, ny, nz, k: 0
    bdrycond.fct[10] = lambda x,y,z: 333
    bdrycond.fct[20] = lambda x,y,z: 293
    bdrycond.fct[40] = bdrycond.fct[20]
    postproc = {}
    postproc['measured'] = "pointvalues:1,2,3"
    postproc['bdryfct'] = "bdryfct:30,31,32,33"
    # postproc['mean33'] = "bdrymean:33"
    postproc['flux20'] = "bdrydn:20"
    postproc['flux40'] = "bdrydn:40"
    return mesh, bdrycond, postproc

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

    def plot(self, point_data, cell_data, info):
        print("time: {}".format(info['timer']))
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
        plt.plot(x, fct(x), 'rx')
        fig2.show()
        plt.show()
    def kparam(self, label):
        if label==100: return self.diffglobal
        return self.param[label-200]
    def dkparam(self, label):
        if label==self.dlabel: return 1.0
        return 0.0
    def matrix(self):
        A = self.fem.matrixDiffusion(self.kheatcell)
        # A += -0*self.fem.massmatrix
        return A
    def getData(self, info):
        dn20, dn40, up = info['postproc']['flux20'], info['postproc']['flux40'], info['postproc']['measured']
        # return up
        return np.concatenate( [np.array([dn20,dn40]), up], axis=0)
    def setinitial(self, mesh, param):
        self.param = param
        # print("self.param", self.param)
        # self.kheatcell = self.kheat(self.mesh.cell_labels)
        self.setMesh(mesh)
        point_data, cell_data, info = self.solve()
        self.plot(point_data, cell_data, info)
        self.data0 = self.getData(info)
        print("self.data0", self.data0)
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
        # self.plot(point_data, cell_data, info)
        data = self.getData((info))
        return data - self.data0
    def solveDstate(self, param):
        # print("@")
        assert param.shape == self.param.shape
        nparam = param.shape[0]
        jac = np.empty(shape=(5,nparam))
        import copy
        bdrycond_bu = copy.deepcopy(self.bdrycond)
        self.bdrycond.fct[10] = np.vectorize(lambda x,y,z: 0.0)
        self.bdrycond.fct[20] = np.vectorize(lambda x,y,z: 0.0)
        self.bdrycond.fct[40] = np.vectorize(lambda x,y,z: 0.0)
        kheatcell_bu = copy.deepcopy(self.kheatcell)
        for i in range(nparam):
            self.dlabel = 200 + i
            self.kheatcell = self.dkheat(self.mesh.cell_labels)
            Bi = self.matrix()
            b = -Bi.dot(self.ustate)
            du = np.zeros_like(b)
            self.kheatcell = self.kheat(self.mesh.cell_labels)
            # A = self.matrix()
            # self.A,b,du = self.boundary(self.A, b, du)
            b,du = self.boundaryvec(b, du)
            du = self.linearSolver(self.A, b, du, solver=self.linearsolver)
            point_data, cell_data, info = self.postProcess(du)
            # print("info", info)
            # self.plot(point_data, cell_data, info)
            jac[:,i] = self.getData((info))
        self.bdrycond = bdrycond_bu
        # print("jac", jac)
        return jac


#----------------------------------------------------------------#
def test():
    mesh, bdrycond, postproc = createMesh()
    # fempy.meshes.plotmesh.meshWithBoundaries(mesh)
    # plt.show()
    diffglobal = 1e-6
    heat = Heat(bdrycond=bdrycond, postproc=postproc, diffglobal=diffglobal)

    param = np.ones(2, dtype=float)
    param[0] = 0.3
    param[1] = 0.03
    heat.setinitial(mesh, param)

    methods = ['trf','lm']
    import time
    for method in methods:
        param[:] = diffglobal
        t0 = time.time()
        info = scipy.optimize.least_squares(heat.solvestate, jac=heat.solveDstate, x0=param, method=method, verbose=0, ftol=1e-14, xtol=1e-14)
        dt = time.time()-t0
        print("{:^10s} x = {} J={:10.2e} nf={:4d} nj={:4d} {:10.2f} s".format(method, info.x, info.cost, info.nfev, info.njev, dt))


#================================================================#

test()
