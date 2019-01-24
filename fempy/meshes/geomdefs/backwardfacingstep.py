# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import pygmsh

# ------------------------------------- #
def define_geometry(h=1.):
    geometry = pygmsh.built_in.Geometry()
    p0 = geometry.add_point([-1.0,  1.0, 0.0], h)
    p1 = geometry.add_point([-1.0,  0.0, 0.0], h)
    p2 = geometry.add_point([ 0.0,  0.0, 0.0], h)
    p3 = geometry.add_point([ 0.0, -1.0, 0.0], h)
    p4 = geometry.add_point([ 3.0, -1.0, 0.0], h)
    p5 = geometry.add_point([ 3.0,  1.0, 0.0], h)
    l0 = geometry.add_line(p0, p1)
    l1 = geometry.add_line(p1, p2)
    l2 = geometry.add_line(p2, p3)
    l3 = geometry.add_line(p3, p4)
    l4 = geometry.add_line(p4, p5)
    l5 = geometry.add_line(p5, p0)
    ll = geometry.add_line_loop([l0, l1, l2, l3, l4, l5])
    surf = geometry.add_plane_surface(ll)
    pl0 = geometry.add_physical_line(l0, label=1001)
    pl1 = geometry.add_physical_line( [l1,l2,l3,l5], label=1002)
    pl4 = geometry.add_physical_line(l4, label=1000)
    geometry.add_physical_surface(surf, label=100)
    return geometry


# ------------------------------------- #
if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import plotmesh, simplexmesh
    import matplotlib.pyplot as plt
    geometry = define_geometry(h=1.0)
    meshdata = pygmsh.generate_mesh(geometry)
    mesh = simplexmesh.SimplexMesh(data=meshdata)
    plotmesh.meshWithBoundaries(mesh)
    plt.show()
