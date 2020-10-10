assert __name__ == '__main__'
import os, sys
import numpy as np
simfempypath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(simfempypath)

import simfempy_pygmsh7 as simfempy
import pygmsh
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- #
def createMesh():
    with pygmsh.geo.Geometry() as geom:
        circle = geom.add_circle(
            x0=[0.5, 0.5, 0.0],
            radius=0.25,
            mesh_size=0.1,
            num_sections=4,
            make_surface=True,
        )
        p = geom.add_rectangle(
            0.0, 1.0, 0.0, 1.0, 0.0, mesh_size=0.1, holes=[circle.curve_loop]
        )
        geom.add_physical(p.surface, label="S:100")
        for i in range(4): geom.add_physical(p.lines[i], label=f"L:{1000 + i}")
        mesh = geom.generate_mesh()
        # for k,v in mesh.__dict__.items():
        #     print("mesh", k, type(v))
        # print("mesh.cell_data", mesh.cell_data)
        # print("mesh.cell_data_dict", mesh.cell_data_dict)
        # print(mesh.cell_data_dict["gmsh:physical"])
        mesh.write('heat.vtu')
    return simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)

# ---------------------------------------------------------------- #
def createData():
    data = simfempy.applications.problemdata.ProblemData()
    bdrycond =  data.bdrycond
    bdrycond.type[1000] = "Neumann"
    bdrycond.type[1001] = "Dirichlet"
    bdrycond.type[1002] = "Neumann"
    bdrycond.type[1003] = "Dirichlet"
    bdrycond.fct[1000] = lambda x,y,z, nx, ny, nz: 0
    bdrycond.fct[1002] = lambda x,y,z, nx, ny, nz: 100
    bdrycond.fct[1001] = bdrycond.fct[1003] = lambda x,y,z: 120
    postproc = data.postproc
    postproc.type['bdrymean_low'] = "bdrymean"
    postproc.color['bdrymean_low'] = [1000]
    postproc.type['bdrymean_up'] = "bdrymean"
    postproc.color['bdrymean_up'] = [1002]
    postproc.type['fluxn'] = "bdrydn"
    postproc.color['fluxn'] = [1001, 1003]
    def kheat(label):
        if label==100: return 0.0001
        return 1000.0
    data.datafct["kheat"] = kheat
    return data

# ---------------------------------------------------------------- #
def test(mesh, problemdata):
    fem = 'p1' # or fem = 'cr1
    heat = simfempy.applications.heat.Heat(problemdata=problemdata, fem=fem, plotk=True)
    heat.setMesh(mesh)
    point_data, cell_data, info = heat.solve()
    print(f"fem={fem} {info['timer']}")
    print(f"postproc: {info['postproc']}")
    simfempy.meshes.plotmesh.meshWithData(mesh, point_data=point_data, cell_data=cell_data, title=fem)
    plt.show()

# ================================================================c#

mesh = createMesh()
simfempy.meshes.plotmesh.meshWithBoundaries(mesh)
# problemdata = createData()
# print("problemdata", problemdata)
# problemdata.check(mesh)
# test(mesh, problemdata)
