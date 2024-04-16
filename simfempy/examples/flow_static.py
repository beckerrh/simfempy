import matplotlib.pyplot as plt
import pathlib, sys
simfempypath = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.insert(0,simfempypath)
import simfempy.applications.navierstokes
import simfempy.models
import simfempy.applications.application

#--------------------------------------------------------------
class FlowExample(simfempy.applications.navierstokes.Application):
    def defineGeometry(self, geom, h):
        points = [geom.add_point((-2, 0, 0), mesh_size=h),
                  geom.add_point((1, 0, 0), mesh_size=h),
                  geom.add_point((1, 1, 0), mesh_size=0.25*h),
                  geom.add_point((2, 1, 0), mesh_size=0.25*h),
                  geom.add_point((2, 0, 0), mesh_size=h),
                  geom.add_point((3, 0, 0), mesh_size=h),
                  geom.add_point((3, 2, 0), mesh_size=0.25*h),
                  geom.add_point((4, 2, 0), mesh_size=0.25*h),
                  geom.add_point((4, 0, 0), mesh_size=h),
                  geom.add_point((8, 0, 0), mesh_size=h),
                  geom.add_point((8, 3, 0), mesh_size=h),
                  geom.add_point((-2, 3, 0), mesh_size=h)]
        channel_lines = [geom.add_line(points[i], points[i + 1]) for i in range(-1,len(points)-1)]
        channel_loop = geom.add_curve_loop(channel_lines)
        plane_surface = geom.add_plane_surface(channel_loop, holes=[])
        geom.add_physical([plane_surface], "Volume")
        geom.add_physical([channel_lines[0]], "Inflow")
        geom.add_physical([channel_lines[-2]], "Outflow")
        wall_lines = channel_lines[1:-2]
        wall_lines.append(channel_lines[-1])
        geom.add_physical(wall_lines, "Walls")
    def defineProblemData(self, problemdata):
        problemdata.bdrycond.set("Dirichlet", ["Walls","Inflow"])
        problemdata.bdrycond.set("Neumann", "Outflow")
        problemdata.bdrycond.fct["Inflow"] = [lambda x, y, z: y*(3-y), lambda x, y, z: 0]
        problemdata.params.scal_glob["mu"] = 5.9
def test_mesh():
    application = FlowExample()
    mesh = application.createMesh(h=0.5)
    mesh.plot(bdry=True)
    print(f"{mesh=}")
    plt.show()

def solve_flow():
    flow_solver = simfempy.models.navierstokes.NavierStokes(application=FlowExample(), verbose=1)
    pp,u = flow_solver.static()
    data = u.tovisudata()
    flow_solver.application.plot(mesh=flow_solver.mesh, data=data)
    plt.show()
    return flow_solver.mesh, u
def solve_heat(mesh, u_flow):
    class HeatExample(simfempy.applications.application.Application):
        def defineProblemData(self, problemdata):
            problemdata.bdrycond.set("Dirichlet", ["Walls", "Inflow"])
            problemdata.bdrycond.set("Neumann", "Outflow")
            problemdata.bdrycond.fct["Inflow"] = lambda x, y, z: 290 + 30* max(0,y-0.5) * max(0,1.5 - y)
            problemdata.bdrycond.fct["Walls"] = lambda x, y, z: 290
            problemdata.params.scal_glob["kheat"] = 0.001
            problemdata.params.data["convection"] = u_flow.extract(name="v")
    heat_solver = simfempy.models.elliptic.Elliptic(mesh = mesh, application=HeatExample())
    result, u_heat = heat_solver.static(method="linear")
    data = u_heat.tovisudata()
    heat_solver.application.plot(mesh=heat_solver.mesh, data=data)
    plt.show()

# testing the mesh
# test_mesh()
# solving flow
mesh, u = solve_flow()
# solving heat
solve_heat(mesh, u)
