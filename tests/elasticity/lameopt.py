assert __name__ == '__main__'
from os import sys, path
import numpy as np
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

import pygmsh
from simfempy.meshes import geomdefs
import simfempy.tools.timer
from simfempy.meshes import pygmshext

#================================================================#
def mesh_traction(h, dim=3, nmeasure=4):
    geometry = pygmsh.built_in.Geometry()
    postproc = {}
    bdrycond =  simfempy.applications.problemdata.BoundaryConditions()
    if dim==2:
        ncomp = 2
        bdrycond.type[1000] = "Neumann"
        bdrycond.type[1001] = "Neumann"
        bdrycond.type[1002] = "Neumann"
        bdrycond.type[1003] = "Dirichlet"
        bdrycond.fct[1001] = lambda x, y, z, nx, ny, nz: np.array([10,0])
        geometry = geomdefs.unitsquare.Unitsquare(x, y)
    elif dim==3:
        ncomp = 3
        bdrycond.type[100] = "Neumann"
        bdrycond.type[101] = "Neumann"
        bdrycond.type[102] = "Neumann"

        x, y, z = [-1, 1], [0, 1], [-1, 1]
        bdrycond.fct[102] = lambda x, y, z, nx, ny, nz: np.array([10,0,0])

        x, y, z = [-1, 1], [-0.5, 0.5], [-0.5, 0.5]
        bdrycond.fct[102] = lambda x, y, z, nx, ny, nz: np.array([0,1,0])
        # bdrycond.fct[102] = lambda x, y, z, nx, ny, nz: np.array([0,0,1])
        # bdrycond.fct[102] = lambda x, y, z, nx, ny, nz: np.array([0,np.cos(np.sqrt(y**2+z**2)),-np.sin(np.sqrt(y**2+z**2))])

        bdrycond.type[103] = "Neumann"
        bdrycond.type[104] = "Dirichlet"
        bdrycond.type[105] = "Neumann"

        p = geometry.add_rectangle(xmin=x[0], xmax=x[1], ymin=y[0], ymax=y[1], z=z[0], lcar=h)
        geometry.add_physical_surface(p.surface, label=100)
        axis = [0, 0, z[1]-z[0]]
        top, vol, ext = geometry.extrude(p.surface, axis)
        # print ('vol', vars(vol))
        # print ('top', vars(top))
        # print ('top.id', top.id)
        # print ('ext[0]', vars(ext[0]))
        geometry.add_physical_surface(top, label=105)
        geometry.add_physical_surface(ext[0], label=101)
        geometry.add_physical_surface(ext[1], label=102)
        geometry.add_physical_surface(ext[2], label=103)
        geometry.add_physical_surface(ext[3], label=104)
        geometry.add_physical_volume(vol, label=10)
        nmeasurey = int(np.sqrt(nmeasure))
        nmeasurez = int(nmeasure / nmeasurey)
        py = np.linspace(0.2,0.8, nmeasurey, endpoint=True)
        pz = np.linspace(0.2,0.8, nmeasurez, endpoint=True)
        print("py", py, "pz", pz)
        hpoint = 0.05*h
        for iy in range(nmeasurey):
            for iz in range(nmeasurez):
                X = (x[1], py[iy]*y[0]+(1-py[iy])*y[1], pz[iz]*z[0]+(1-pz[iz])*z[1])
                label = 10000+iy+iz*nmeasurey
                print("label", label)
                pygmshext.add_point_in_surface(geometry, surf=ext[1], X=X, lcar=hpoint, label=label)
    else:
        raise ValueError("unknown geomname={}".format(geomname))
    data = pygmsh.generate_mesh(geometry, verbose=False)
    mesh = simfempy.meshes.simplexmesh.SimplexMesh(data=data)
    bdrycond.check(mesh.bdrylabels.keys())
    # mesh.plotWithBoundaries()
    problemdata = simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc, ncomp=ncomp)
    return mesh, problemdata

#----------------------------------------------------------------#
class Elasticity(simfempy.applications.elasticity.Elasticity):
    def __init__(self, **kwargs):
        kwargs['fem'] = 'p1'
        kwargs['plotk'] = True
        super().__init__(**kwargs)
        self.linearsolver = "pyamg"


#================================================================#
def test_plot():
    hmean = 0.1
    mesh, problemdata = mesh_traction(hmean, nmeasure=1)




    elasticity = Elasticity(problemdata=problemdata)
    elasticity.setMesh(mesh)



    A = elasticity.matrix()
    b, u = elasticity.computeRhs()
    ncomp = elasticity.ncomp
    u, niter = elasticity.linearSolver(A, b, u, solver="pyamg")
    ncomp = elasticity.ncomp
    point_data={}
    for icomp in range(ncomp):
        point_data["u{:1d}".format(icomp)] = u[icomp::ncomp]
    assert ncomp==3
    elasticity.mesh.plotWithData(point_data=point_data, translate_point_data=True)

#================================================================#

test_plot()