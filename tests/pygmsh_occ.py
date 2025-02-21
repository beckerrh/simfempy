# example from https://pypi.org/project/pygmsh/
import pygmsh, meshio
import matplotlib.pyplot as plt

with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_min = 0.05
    geom.characteristic_length_max = 0.1
    rectangle = geom.add_rectangle([-1.0, -1.0, 0.0], 2.0, 2.0)
    disk1 = geom.add_disk([-1.2, 0.0, 0.0], 0.5)
    disk2 = geom.add_disk([+1.2, 0.0, 0.0], 0.5)
    disk3 = geom.add_disk([0.0, -0.9, 0.0], 0.5)
    disk4 = geom.add_disk([0.0, +0.9, 0.0], 0.5)
    flat = geom.boolean_difference(
        geom.boolean_union([rectangle, disk1, disk2]),
        geom.boolean_union([disk3, disk4]),
    )
    # geom.extrude(flat, [0, 0, 0.3])
    mesh = geom.generate_mesh()
#mesh = meshio.Mesh(mesh)
celltypes = [c.type for c in mesh.cells]
for cells in mesh.cells:
    if cells.type == "triangle":
        triangles = cells.data
x = mesh.points
plt.figure(figsize=(8, 8))
plt.triplot(x[:,0], x[:,1], triangles)
plt.gca().set_aspect('equal')
plt.show()