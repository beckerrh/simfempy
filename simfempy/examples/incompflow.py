# in shell
import os, sys
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pygmsh
from simfempy.applications.stokes import Stokes
from simfempy.applications.navierstokes import NavierStokes
from simfempy.applications.problemdata import ProblemData
from simfempy.meshes.simplexmesh import SimplexMesh
from simfempy.meshes import plotmesh
from simfempy.tools import timer

# ================================================================c#
def main(**kwargs):
    modelargs = kwargs.pop('modelargs', {})
    testcases = ['drivenCavity', 'backwardFacingStep', 'poiseuille', 'schaeferTurek']
    testcase = kwargs.pop('testcase', testcases[0])
    model = kwargs.pop('model', 'NavierStokes')
    bdryplot = kwargs.pop('bdryplot', False)
    plot = kwargs.pop('plot', False)
    # linearsolver = kwargs.pop('linearsolver', 'gcrotmk_1')
    # precond_p = kwargs.pop('precond_p', 'diag')
    # create mesh and data
    t = timer.Timer("mesh")
    mesh, data = eval(testcase)(**kwargs)
    t.add('pygmsh')
    mesh = SimplexMesh(mesh)
    t.add('SimplexMesh')
    print(t)
    print(f"{mesh=}")
    if bdryplot: 
        plotmesh.meshWithBoundaries(mesh)
        plt.show()
        return
    # create application
    if model == "Stokes":
        model = Stokes(mesh=mesh, problemdata=data, **modelargs)
    else:
        # model = NavierStokes(mesh=mesh, problemdata=data, hdivpenalty=10)
        model = NavierStokes(mesh=mesh, problemdata=data, **modelargs)
    result = model.solve()
    print(f"{result.info['timer']}")
    print(f"postproc:")
    for k, v in result.data['global'].items(): print(f"{k}: {v}")
    if mesh.dimension ==2:
        fig = plt.figure(figsize=(10, 8))
        outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
        plotmesh.meshWithData(mesh, data=result.data, title="Stokes", fig=fig, outer=outer[0])
        plotmesh.meshWithData(mesh, title="Stokes", fig=fig, outer=outer[1],
                            quiver_data={"V":list(result.data['point'].values())})
        plt.show()
    else:
        filename = testcase+'.vtu'
        mesh.write(filename, data=result.data)
        if plot:
            import pyvista as pv
            mesh = pv.read(filename)
            cpos = mesh.plot()


# ================================================================ #
def poiseuille2d(h= 0.1, mu=0.1):
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_rectangle(xmin=0, xmax=4, ymin=0, ymax=1, z=0, mesh_size=h)
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
        mesh = geom.generate_mesh()
    data = ProblemData()
   # boundary conditions
    data.bdrycond.set("Dirichlet", [1002,1000,1003])
    data.bdrycond.set("Neumann", [])
    data.bdrycond.set("Navier", [])
    data.bdrycond.set("Pressure", [1001])
    data.bdrycond.fct[1003] = [lambda x, y, z:  4*y*(1-y), lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    data.params.scal_glob["navier"] = 1.01
    #TODO pass ncomp with mesh ?!
    data.ncomp = 2
    return mesh, data
# ================================================================ #
def poiseuille3d(h= 0.1, mu=0.1):
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_rectangle(xmin=0, xmax=4, ymin=0, ymax=1, z=0, mesh_size=h)
        axis = [0, 0, 1]
        top, vol, lat = geom.extrude(p.surface, axis)
        geom.add_physical([top, p.surface, lat[0], lat[2]], label="100")
        geom.add_physical(lat[1], label="101")
        geom.add_physical(lat[3], label="103")
        geom.add_physical(vol, label="10")
        mesh = geom.generate_mesh()
    data = ProblemData()
   # boundary conditions
    data.bdrycond.set("Dirichlet", [100, 103])
    data.bdrycond.set("Neumann", [101])
    data.bdrycond.fct[103] = [lambda x, y, z:  16*y*(1-y)*z*(1-z), lambda x, y, z: 0, lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    data.ncomp = 3
    return mesh, data
# ================================================================c#
def drivenCavity2d(h=0.1, mu=0.01):
    with pygmsh.geo.Geometry() as geom:
        ms = [h*v for v in [1.,1.,0.2,0.2]]
        p = geom.add_rectangle(xmin=0, xmax=1, ymin=0, ymax=1, z=0, mesh_size=ms)
        geom.add_physical(p.surface, label="100")
        geom.add_physical(p.lines[2], label="1002")
        geom.add_physical([p.lines[0], p.lines[1], p.lines[3]], label="1000")
        mesh = geom.generate_mesh()
    data = ProblemData()
    # boundary conditions
    data.bdrycond.set("Dirichlet", [1000, 1002])
    data.bdrycond.fct[1002] = [lambda x, y, z: 1, lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    data.params.scal_glob["navier"] = mu
    #TODO pass ncomp with mesh ?!
    data.ncomp = 2
    return mesh, data
# ================================================================c#
def drivenCavity3d(h=0.1, mu=0.01):
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_rectangle(xmin=0, xmax=1, ymin=0, ymax=1, z=0, mesh_size=h)
        axis = [0, 0, 1]
        top, vol, lat = geom.extrude(p.surface, axis)
        geom.add_physical(top, label="102")
        geom.add_physical([p.surface, lat[0], lat[1], lat[2], lat[3]], label="100")
        geom.add_physical(vol, label="10")
        mesh = geom.generate_mesh()
    data = ProblemData()
    # boundary conditions
    data.bdrycond.set("Dirichlet", [100, 102])
    data.bdrycond.fct[102] = [lambda x, y, z: 1, lambda x, y, z: 0, lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    data.params.scal_glob["navier"] = mu
    data.ncomp = 3
    return mesh, data
# ================================================================ #
def backwardFacingStep2d(h=0.2, mu=0.02):
    with pygmsh.geo.Geometry() as geom:
        X = []
        X.append([-1.0, 1.0])
        X.append([-1.0, 0.0])
        X.append([0.0, 0.0])
        X.append([0.0, -1.0])
        X.append([3.0, -1.0])
        X.append([3.0, 1.0])
        p = geom.add_polygon(points=np.insert(np.array(X), 2, 0, axis=1), mesh_size=h)
        geom.add_physical(p.surface, label="100")
        dirlines = [p for i,p in enumerate(p.lines) if i != 0 and i != 4]
        geom.add_physical(dirlines, "1002")
        geom.add_physical(p.lines[0], "1000")
        geom.add_physical(p.lines[4], "1004")
        mesh = geom.generate_mesh()
    data = ProblemData()
    # boundary conditions
    data.bdrycond.set("Dirichlet", [1000, 1002])
    data.bdrycond.set("Pressure", [1004])
    data.bdrycond.fct[1000] = [lambda x, y, z: y*(1-y),  lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    data.ncomp = 2
    return mesh, data
# ================================================================ #
def backwardFacingStep3d(h=0.2, mu=0.02):
    with pygmsh.geo.Geometry() as geom:
        X = []
        X.append([-1.0, 1.0])
        X.append([-1.0, 0.0])
        X.append([0.0, 0.0])
        X.append([0.0, -1.0])
        X.append([3.0, -1.0])
        X.append([3.0, 1.0])
        p = geom.add_polygon(points=np.insert(np.array(X), 2, 0, axis=1), mesh_size=h)
        axis = [0, 0, 1]
        top, vol, lat = geom.extrude(p.surface, axis)
        dirf = [lat[i] for i in range(1,6) if i!=4 ]
        dirf.extend([p.surface, top])
        geom.add_physical(dirf, label="100")
        geom.add_physical(lat[0], label="102")
        geom.add_physical(lat[4], label="104")
        # for i in range(len(lat)):
        #     geom.add_physical(lat[i], label=f"{101+i}")
        geom.add_physical(vol, label="10")
        # geom.add_physical(p.surface, label="100")
        # dirlines = [p for i,p in enumerate(p.lines) if i != 0 and i != 4]
        # geom.add_physical(dirlines, "1002")
        # geom.add_physical(p.lines[0], "1000")
        # geom.add_physical(p.lines[4], "1004")
        mesh = geom.generate_mesh()
    data = ProblemData()
    # boundary conditions
    data.bdrycond.set("Dirichlet", [100, 102])
    data.bdrycond.set("Pressure", [104])
    data.bdrycond.fct[102] = [lambda x, y, z: y*(1-y)*z*(1-z), lambda x, y, z: 0, lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    data.ncomp = 3
    return mesh, data
# ================================================================ #
def schaeferTurek2d(h= 0.5, hcircle=None):
    if hcircle is None: hcircle = 0.2*h
    with pygmsh.geo.Geometry() as geom:
        circle = geom.add_circle(x0=[2,2], radius=0.5, mesh_size=hcircle, num_sections=10, make_surface=False)
        geom.add_physical(circle.curve_loop.curves, label="3000")
        p = geom.add_rectangle(xmin=0, xmax=11, ymin=0, ymax=4.1, z=0, mesh_size=h, holes=[circle])
        geom.add_physical(p.surface, label="100")
        for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
        mesh = geom.generate_mesh()
    data = ProblemData()
   # boundary conditions
    data.bdrycond.set("Dirichlet", [1002,1000,1003,3000])
    data.bdrycond.set("Neumann", [1001])
    data.bdrycond.fct[1003] = [lambda x, y, z:  0.3*y*(4.1-y)/2.05**2, lambda x, y, z: 0]
    data.params.scal_glob["mu"] = 0.01
    data.postproc.set(name='bdrynflux', type='bdry_nflux', colors=3000)
    def changepostproc(info):
        bdrynflux = info.pop('bdrynflux')
        info['drag'] = -50*bdrynflux[0]
        info['lift'] = -50*bdrynflux[1]
        info['err_drag'] =  5.57953523384+50*bdrynflux[0]
        info['err_lift'] =  0.010618937712+50*bdrynflux[1]    
    data.postproc.changepostproc = changepostproc
    data.ncomp = 2
    return mesh, data
# ================================================================ #
def schaeferTurek3d(h= 1, hcircle=None):
    if hcircle is None: hcircle = 0.25*h
    with pygmsh.geo.Geometry() as geom:
        circle = geom.add_circle(x0=[5,2], radius=0.5, mesh_size=hcircle, num_sections=8, make_surface=False)
        p = geom.add_rectangle(xmin=0, xmax=25, ymin=0, ymax=4.1, z=0, mesh_size=h, holes=[circle])
        axis = [0, 0, 4.1]
        top, vol, lat = geom.extrude(p.surface, axis)
        geom.add_physical([top,p.surface, lat[0], lat[2]], label="100")
        geom.add_physical(lat[1], label="101")
        geom.add_physical(lat[3], label="103")
        geom.add_physical(lat[4:], label="300")
        geom.add_physical(vol, label="10")
        mesh = geom.generate_mesh()
    data = ProblemData()
   # boundary conditions
    data.bdrycond.set("Dirichlet", [100,103,300])
    data.bdrycond.set("Neumann", [101])
    data.bdrycond.fct[103] = [lambda x, y, z:  0.45*y*(4.1-y)*z*(4.1-z)/2.05**4, lambda x, y, z: 0, lambda x, y, z: 0]
    data.params.scal_glob["mu"] = 0.01
    data.postproc.set(name='bdrynflux', type='bdry_nflux', colors=300)
    data.postproc.set(name='mean', type='bdry_vmean', colors=[101,103])
    data.ncomp = 3
    return mesh, data

#================================================================#
if __name__ == '__main__':
    # main(testcase='poiseuille2d', h=0.2, mu=1e-6, modelargs={'convmethod':'lps', 'divdivparam':1., 'hdivpenalty':0.1, 'precond_v':'spsolve'})
    # main(testcase='drivenCavity2d', h=1, mu=3e-2, precond_p='schur')
    # main(testcase='backwardFacingStep2d', mu=2e-3)
    main(testcase='backwardFacingStep3d', mu=2e-2)
    # main(testcase='schaeferTurek2d')
    # main(testcase='poiseuille3d', h=0.2, mu=1e-3)
    # main(testcase='drivenCavity3d', mu=0.001, precond_p='schur')
    # main(testcase='schaeferTurek3d', h=0.5, bdryplot=False, model='Stokes', plot=False)
    # main(testcase='schaeferTurek3d', h=0.5, bdryplot=False, linearsolver='gcrotmk_1', model='Stokes', plot=False)
    # main(testcase='schaeferTurek3d', h=0.95, bdryplot=False, linearsolver='spsolve', model='Stokes', plot=False)
