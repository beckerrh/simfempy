# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


def definition(geometry, h=0.2):
    p0 =  geometry.add_point([-1.0, -1.0, 0.0], h)
    p1 =  geometry.add_point([1.0, -1.0, 0.0], h)
    p2 =  geometry.add_point([1.0, 1.0, 0.0], h)
    p3 =  geometry.add_point([-1.0, 1.0, 0.0], h)
    l0 =  geometry.add_line(p0, p1)
    l1 =  geometry.add_line(p1, p2)
    l2 =  geometry.add_line(p2, p3)
    l3 =  geometry.add_line(p3, p0)
    ll =  geometry.add_line_loop([l0, l1, l2, l3])
    surf =  geometry.add_plane_surface(ll)
    pl0 =  geometry.add_physical_line(l0, label=11)
    pl1 =  geometry.add_physical_line(l1, label=22)
    pl1 =  geometry.add_physical_line(l2, label=33)
    pl4 =  geometry.add_physical_line(l3, label=44)
    geometry.add_physical_surface(surf, label=99)
    return 'unitsquare', geometry

# ------------------------------------- #

if __name__ == '__main__':
    from . import geometry
    geom = geometry.Geometry(definition=definition)
    geom.show()