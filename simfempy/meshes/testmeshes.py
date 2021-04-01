import pygmsh
import numpy as np
import simfempy

__pygmsh6__ = False
if hasattr(pygmsh, "built_in"): __pygmsh6__ = True

# ------------------------------------- #
def unitline(h):
    if __pygmsh6__:
        geom = pygmsh.built_in.Geometry()
        p0 = geom.add_point([0, 0, 0], lcar=h)
        p1 = geom.add_point([1, 0, 0], lcar=h)
        p = geom.add_line(p0, p1)
        geom.add_physical(p0, label=10000)
        geom.add_physical(p1, label=10001)
        geom.add_physical(p, label=1000)
        mesh = pygmsh.generate_mesh(geom, verbose=False)
    else:
        with pygmsh.geo.Geometry() as geom:
            p0 = geom.add_point([0, 0, 0], mesh_size=h)
            p1 = geom.add_point([1, 0, 0], mesh_size=h)
            p = geom.add_line(p0, p1)
            geom.add_physical(p0, label="10000")
            geom.add_physical(p1, label="10001")
            geom.add_physical(p, label="1000")
            mesh = geom.generate_mesh()
    return simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)

# ------------------------------------- #
def unitsquare(h):
    a=1
    if __pygmsh6__:
        geom = pygmsh.built_in.Geometry()
        p = geom.add_rectangle(xmin=-a, xmax=a, ymin=-a, ymax=a, z=0, lcar=h)
        geom.add_physical(p.surface, label=100)
        # geom.add_physical(p.points[0], label=11111)
        for i in range(4): geom.add_physical(p.line_loop.lines[i], label=1000 + i)
        mesh = pygmsh.generate_mesh(geom, verbose=False)
    else:
        with pygmsh.geo.Geometry() as geom:
            p = geom.add_rectangle(xmin=-a, xmax=a, ymin=-a, ymax=a, z=0, mesh_size=h)
            geom.add_physical(p.surface, label="100")
            # geom.add_physical(p.points[0], label="11111")
            for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
            mesh = geom.generate_mesh()
    # print(f"{mesh=}")
    # print(f"{mesh.cell_data=}")
    # print(f"{mesh.cell_sets=}")
    # print("{mesh=}")
    return simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)


# ------------------------------------- #
def unitcube(h):
    if __pygmsh6__:
        geom = pygmsh.built_in.Geometry()
        x, y, z = [-1, 1], [-1, 1], [-1, 1]
        p = geom.add_rectangle(xmin=x[0], xmax=x[1], ymin=y[0], ymax=y[1], z=z[0], lcar=h)
        geom.add_physical(p.surface, label=100)
        axis = [0, 0, z[1] - z[0]]
        top, vol, ext = geom.extrude(p.surface, axis)
        geom.add_physical(top, label=105)
        geom.add_physical(ext[0], label=101)
        geom.add_physical(ext[1], label=102)
        geom.add_physical(ext[2], label=103)
        geom.add_physical(ext[3], label=104)
        geom.add_physical(vol, label=10)
        mesh = pygmsh.generate_mesh(geom, verbose=False)
    else:
        with pygmsh.geo.Geometry() as geom:
            x, y, z = [-1, 1], [-1, 1], [-1, 1]
            p = geom.add_rectangle(xmin=x[0], xmax=x[1], ymin=y[0], ymax=y[1], z=z[0], mesh_size=h)
            geom.add_physical(p.surface, label="100")
            axis = [0, 0, z[1] - z[0]]
            top, vol, ext = geom.extrude(p.surface, axis)
            geom.add_physical(top, label="105")
            geom.add_physical(ext[0], label="101")
            geom.add_physical(ext[1], label="102")
            geom.add_physical(ext[2], label="103")
            geom.add_physical(ext[3], label="104")
            geom.add_physical(vol, label="10")
            mesh = geom.generate_mesh()
    return simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)


# ------------------------------------- #
def backwardfacingstep(h=1.):
    if __pygmsh6__:
        geom = pygmsh.built_in.Geometry()
        X = []
        X.append([-1.0,  1.0])
        X.append([-1.0,  0.0])
        X.append([ 0.0,  0.0])
        X.append([ 0.0, -1.0])
        X.append([ 3.0, -1.0])
        X.append([ 3.0,  1.0])
        p = geom.add_polygon(X=np.insert(np.array(X), 2, 0, axis=1), lcar=h)
        geom.add_physical(p.surface, label=100)
        ll = p.line_loop
        for i in range(len(ll.lines)): geom.add_physical(ll.lines[i], label=1000+i)
        mesh = pygmsh.generate_mesh(geom, verbose=False)
    else:
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
            for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000 + i}")
            mesh = geom.generate_mesh()
        return simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)

# ------------------------------------- #
def backwardfacingstep3d(h):
    X = []
    X.append([-1.0, 1.0])
    X.append([-1.0, 0.0])
    X.append([0.0, 0.0])
    X.append([0.0, -1.0])
    X.append([3.0, -1.0])
    X.append([3.0, 1.0])
    if __pygmsh6__:
        geom = pygmsh.built_in.Geometry()
        p = geom.add_polygon(X=np.insert(np.array(X), 2, -1.0, axis=1), lcar=h)
        geom.add_physical(p.surface, label=100)
        axis = [0, 0, 2]
        top, vol, ext = geom.extrude(p.surface, axis)
        next = len(ext)
        geom.add_physical(top, label=101+next)
        for i in range(next):
            geom.add_physical(ext[i], label=101+i)
        geom.add_physical(vol, label=10)
        return simfempy.meshes.simplexmesh.SimplexMesh(mesh=pygmsh.generate_mesh(geom, verbose=False))
    else:
        with pygmsh.geo.Geometry() as geom:
            p = geom.add_polygon(points=np.insert(np.array(X), 2, -1.0, axis=1), mesh_size=h)
            geom.add_physical(p.surface, label="100")
            axis = [0, 0, 2]
            top, vol, ext = geom.extrude(p.surface, axis)
            next = len(ext)
            geom.add_physical(top, label=f"{101 + next}")
            for i in range(next):
                geom.add_physical(ext[i], label=f"{101 + i}")
            geom.add_physical(vol, label="10")
            mesh = geom.generate_mesh()
        return simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)


# ------------------------------------- #
def equilateral(h):
    geom = pygmsh.built_in.Geometry()
    a = 1.0
    X = []
    X.append([-0.5*a, 0, 0])
    X.append([0, -0.5*np.sqrt(3)*a, 0])
    X.append([0.5*a, 0, 0])
    X.append([0, 0.5*np.sqrt(3)*a, 0])
    p = geom.add_polygon(X=X, lcar = h)
    geom.add_physical(p.surface, label=100)
    for i in range(4): geom.add_physical(p.line_loop.lines[i], label=1000 + i)
    return simfempy.meshes.simplexmesh.SimplexMesh(mesh=pygmsh.generate_mesh(geom, verbose=False))
