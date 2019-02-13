from os import sys, path
import numpy as np
fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(fempypath)

import simfempy
from simfempy.meshes import geomdefs
from simfempy.applications.heat import Heat
from simfempy.applications.laplacemixed import LaplaceMixed
import simfempy.tools.comparemethods


#----------------------------------------------------------------#
def getGeometryProblemDataInterface(dim=2, kin=1, kex=1):
    class InterfaceGeometry2d(geomdefs.geometry.Geometry):
        def __init__(self, radius):
            self.radius = radius
            super().__init__()
        def define(self, h=1):
            self.reset()
            p = self.add_circle(x0=(0,0,0), radius=self.radius, lcar=h)
            self.add_physical_surface(p.plane_surface, label=200)
            rect = [-1, 1, -1, 1]
            p1 = self.add_rectangle(rect[0], rect[1], rect[2], rect[3], 0, lcar=h, holes=[p])
            self.add_physical_surface(p1.surface, label=100)
            for i,line in enumerate(p1.line_loop.lines):
                self.add_physical_line(line, label=1000+i)
    class InterfaceGeometry3d(geomdefs.geometry.Geometry):
        def __init__(self, radius):
            self.radius = radius
            super().__init__()
        def define(self, h=1):
            self.reset()
            p = self.add_ball(x0=(0,0,0), radius=self.radius, lcar=h, with_volume=True)
            # print(dir(p))
            # self.add_physical_volume(p.volume, label=200)
            # box = [-1, 1, -1, 1, -1, 1]
            # p1 = self.add_box(*box, lcar=h, holes=[p.surface_loop])
            # print(dir(p1))
            # self.add_physical_volume(p1.volume, label=100)
            # for i,surface in enumerate(p1.surface_loop.surfaces):
            #     self.add_physical_surface(surface, label=1000+i)

    class AnalyticalSolutionInterface(object):
        def __init__(self, kin, kex, r, dim):
            self.kin, self.kex, self.r2 = kin, kex, r*r
            self.cor = self.r2 / self.kin - self.r2 / self.kex
            self.fct = np.vectorize(self._fct)
            self.fct_x = np.vectorize(self._fct_x)
            self.fct_y = np.vectorize(self._fct_y)
            self.rhs = np.vectorize(lambda x,y,z: -2*dim)

        def _fct(self, x, y, z):
            r2 = x**2 + y**2 + z**2
            if r2 < self.r2: return r2 / self.kin
            return r2 / self.kex + self.cor
        def _fct_x(self, x, y, z):
            r2 = x**2 + y**2 + z**2
            if r2 < self.r2: return 2*x / self.kin
            return 2*x / self.kex
        def _fct_y(self, x, y, z):
            r2 = x**2 + y**2 + z**2
            if r2 < self.r2: return 2*y / self.kin
            return 2*y / self.kex

        def d(self, i, x, y, z):
            if i == 2:
                return self.fct_z(x, y, z)
            elif i == 1:
                return self.fct_y(x, y, z)
            return self.fct_x(x, y, z)

        def __call__(self, x, y, z):
            return self.fct(x, y, z)

    radius = 1 / np.sqrt(2)
    if dim==2:
        geometry = InterfaceGeometry2d(radius=radius)
    else:
        geometry = InterfaceGeometry3d(radius=radius)

    mesh = simfempy.meshes.simplexmesh.SimplexMesh(geometry=geometry)
    mesh.plotWithBoundaries()
    import matplotlib.pyplot as plt
    plt.show()


    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    postproc = {}
    bdrycond.type[1000] = bdrycond.type[1001] = bdrycond.type[1002] = bdrycond.type[1003] = "Dirichlet"

    problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc)
    problemdata.solexact = AnalyticalSolutionInterface(kin=kin, kex=kex, r=radius, dim=dim)
    bdrycond.fct[1000] = bdrycond.fct[1001] = bdrycond.fct[1002] = bdrycond.fct[1003] = problemdata.solexact
    problemdata.rhs = problemdata.solexact.rhs
    def kheat(label):
        if label==200: return kin
        return kex
    problemdata.diffcoeff = kheat

    return geometry, problemdata

#----------------------------------------------------------------#
def getGeometryProblemData(geomname = "unitcube", exactsolution="Linear"):
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    postproc = {}
    if geomname == "unitsquare":
        bdrycond.type[1000] = "Neumann"
        bdrycond.type[1001] = "Dirichlet"
        bdrycond.type[1002] = "Neumann"
        bdrycond.type[1003] = "Robin"
        bdrycond.param[1003] = 11
        postproc['bdrymean'] = "bdrymean:1000,1002"
        postproc['bdrydn'] = "bdrydn:1001"
        geometry = geomdefs.unitsquare.Unitsquare()
    elif geomname == "unitcube":
        bdrycond.type[100] = "Neumann"
        bdrycond.type[105] = "Neumann"
        bdrycond.type[101] = "Dirichlet"
        bdrycond.type[102] = "Dirichlet"
        bdrycond.type[103] = "Robin"
        bdrycond.type[104] = "Robin"
        bdrycond.param[103] = 1
        bdrycond.param[104] = 10
        postproc['bdrymean'] = "bdrymean:100,105"
        postproc['bdrydn'] = "bdrydn:101,102"
        geometry = geomdefs.unitcube.Unitcube()
    heat = Heat(geometry=geometry, showmesh=False)
    problemdata = heat.generatePoblemData(exactsolution=exactsolution, bdrycond=bdrycond, postproc=postproc)
    return geometry, problemdata


#----------------------------------------------------------------#
def test_analytic(geometry, problemdata, h, verbose=2):

    methods = {}
    methods['p1'] = Heat(problemdata=problemdata, fem='p1')
    methods['cr1'] = Heat(problemdata=problemdata, fem='cr1')
    methods['rt0'] = LaplaceMixed(problemdata=problemdata)
    comp = simfempy.tools.comparemethods.CompareMethods(methods, verbose=verbose)

    comp.compare(geometry=geometry, h=h)

#================================================================#
if __name__ == '__main__':
    geometry, problemdata = getGeometryProblemDataInterface(dim=2, kin=100000, kex=1)
    # geometry, problemdata = getGeometryProblemData(geomname = "unitsquare", exactsolution= 'Sinus')
    if str(geometry) == "Unitsquare":
        h = [1, 0.5, 0.25, 0.125, 0.06, 0.03, 0.015]
    elif str(geometry) == "Unitcube":
        h = [2.0, 1.0, 0.5, 0.25, 0.12, 0.06]
    else:
        h = [1, 0.5, 0.25, 0.125, 0.06, 0.03]

    test_analytic(geometry, problemdata, h, verbose=5)
