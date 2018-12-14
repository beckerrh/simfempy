# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import pygmsh

# ------------------------------------- #
def define_geometry(h=1.0):
    geometry = pygmsh.built_in.Geometry()
    a = 1
    p = geometry.add_rectangle(xmin=-a, xmax=a, ymin=-a, ymax=a, z=-a, lcar=h)
    geometry.add_physical_surface(p.surface, label=11)
    axis = [0, 0, 2*a]
    top, vol, ext = geometry.extrude(p.surface, axis)
    # print ('vol', vars(vol))
    # print ('top', vars(top))
    # print ('top.id', top.id)
    # print ('ext[0]', vars(ext[0]))
    geometry.add_physical_surface(top, label=66)
    geometry.add_physical_surface(ext[0], label=22)
    geometry.add_physical_surface(ext[1], label=33)
    geometry.add_physical_surface(ext[2], label=44)
    geometry.add_physical_surface(ext[3], label=55)
    geometry.add_physical_volume(vol, label=111)
    return geometry

    p0 =  geometry.add_point([-1.0, -1.0, -1.0], h)
    p1 =  geometry.add_point([ 1.0, -1.0, -1.0], h)
    p2 =  geometry.add_point([ 1.0,  1.0, -1.0], h)
    p3 =  geometry.add_point([-1.0,  1.0, -1.0], h)
    p4 =  geometry.add_point([-1.0, -1.0,  1.0], h)
    p5 =  geometry.add_point([ 1.0, -1.0,  1.0], h)
    p6 =  geometry.add_point([ 1.0,  1.0,  1.0], h)
    p7 =  geometry.add_point([-1.0,  1.0,  1.0], h)
    l0 =  geometry.add_line(p0, p1)
    l1 =  geometry.add_line(p1, p2)
    l2 =  geometry.add_line(p2, p3)
    l3 =  geometry.add_line(p3, p0)
    l4 =  geometry.add_line(p4, p5)
    l5 =  geometry.add_line(p5, p6)
    l6 =  geometry.add_line(p6, p7)
    l7 =  geometry.add_line(p7, p4)
    l8 =  geometry.add_line(p0, p4)
    l9 =  geometry.add_line(p1, p5)
    l10 =  geometry.add_line(p2, p6)
    l11 =  geometry.add_line(p3, p7)
    ll1 =  geometry.add_line_loop([l0, l1, l2, l3])
    surf1 =  geometry.add_plane_surface(ll1)
    geometry.add_physical_surface(surf1, label=11)
    ll2 =  geometry.add_line_loop([l4, l5, l6, l7])
    surf2 =  geometry.add_plane_surface(ll2)
    geometry.add_physical_surface(surf2, label=22)
    ll3 =  geometry.add_line_loop([l0, l9, -l4, -l8])
    surf3 =  geometry.add_plane_surface(ll3)
    geometry.add_physical_surface(surf3, label=33)
    ll4 =  geometry.add_line_loop([l1, l10, -l5, -l9])
    surf4 =  geometry.add_plane_surface(ll4)
    geometry.add_physical_surface(surf4, label=44)
    ll5 =  geometry.add_line_loop([l2, l11, -l6, -l10])
    surf5 =  geometry.add_plane_surface(ll5)
    geometry.add_physical_surface(surf5, label=55)
    ll6 =  geometry.add_line_loop([l3, l8, -l7, -l11])
    surf6 =  geometry.add_plane_surface(ll6)
    geometry.add_physical_surface(surf6, label=66)
    sl = geometry.add_surface_loop([surf1, surf2, surf3, surf4, surf5, surf6])
    vol = geometry.add_volume(sl)
    geometry.add_physical_volume(vol, label=9999)
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
