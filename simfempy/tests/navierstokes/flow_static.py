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
        # model = geom.__enter__()
        model = geom
        points = [model.add_point((-2, 0, 0), mesh_size=h),
                  model.add_point((1, 0, 0), mesh_size=h),
                  model.add_point((1, 1, 0), mesh_size=0.25*h),
                  model.add_point((2, 1, 0), mesh_size=0.25*h),
                  model.add_point((2, 0, 0), mesh_size=h),
                  model.add_point((3, 0, 0), mesh_size=h),
                  model.add_point((3, 2, 0), mesh_size=0.25*h),
                  model.add_point((4, 2, 0), mesh_size=0.25*h),
                  model.add_point((4, 0, 0), mesh_size=h),
                  model.add_point((8, 0, 0), mesh_size=h),
                  model.add_point((8, 3, 0), mesh_size=h),
                  model.add_point((-2, 3, 0), mesh_size=h)]
        channel_lines = [model.add_line(points[i], points[i + 1]) for i in range(-1,len(points)-1)]
        channel_loop = model.add_curve_loop(channel_lines)
        plane_surface = model.add_plane_surface(channel_loop, holes=[])
        model.synchronize()
        model.add_physical([plane_surface], "Volume")
        model.add_physical([channel_lines[0]], "Inflow")
        model.add_physical([channel_lines[-2]], "Outflow")
        wall_lines = channel_lines[1:-2]
        wall_lines.append(channel_lines[-1])
        model.add_physical(wall_lines, "Walls")

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
    # linearsolver_def = {'method': 'scipy_lgmres', 'maxiter': 100, 'prec': 'Chorin', 'disp': 0, 'rtol': 1e-6}
    linearsolver = simfempy.models.stokes.linearsolver_def
    linearsolver['disp'] = 1
    # linearsolver = 'spsolve'
    # flow_solver = simfempy.models.stokes.Stokes(application=FlowExample(), linearsolver=linearsolver, singleA=False, scale_ls=False, stack_storage=False)
    flow_solver = simfempy.models.navierstokes.NavierStokes(application=FlowExample(), linearsolver=linearsolver, scale_ls=False, singleA=True, stack_storage=False)
    # pp,u = flow_solver.static(method='linear')
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
            problemdata.bdrycond.fct["Inflow"] = lambda x, y, z: y * (3 - y)
            problemdata.params.scal_glob["kheat"] = 0.01
            problemdata.params.data["convection"] = u_flow.extract(name="v")
    heat_solver = simfempy.models.elliptic.Elliptic(mesh = mesh, application=HeatExample())
    result, u_heat = heat_solver.static(mode="newton")
    data = u_heat.tovisudata()
    heat_solver.application.plot(mesh=heat_solver.mesh, data=data)
    plt.show()

# testing the mesh
# test_mesh()
# solving flow
mesh, u = solve_flow()
# solving heat
solve_heat(mesh, u)
