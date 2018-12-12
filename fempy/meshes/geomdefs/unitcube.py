# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


def definition(geometry, h=0.2):
    p0 =  geometry.add_point([-1.0, -1.0, -1.0], h)
    p1 =  geometry.add_point([ 1.0, -1.0, -1.0], h)
    p2 =  geometry.add_point([ 1.0,  1.0, -1.0], h)
    p3 =  geometry.add_point([-1.0,  1.0, -1.0], h)
    p4 =  geometry.add_point([-1.0, -1.0,  1.0], h)
    p5 =  geometry.add_point([ 1.0, -1.0,  1.0], h)
    p6 =  geometry.add_point([ 1.0,  1.0,  1.0], h)
    p7 =  geometry.add_point([-1.0,  1.0,  1.0], h)
    # geometry.add_physical_point(p0, label=11)
    # geometry.add_physical_point(p1, label=12)
    # geometry.add_physical_point(p2, label=13)
    # geometry.add_physical_point(p3, label=14)
    # geometry.add_physical_point(p4, label=15)
    # geometry.add_physical_point(p5, label=16)
    # geometry.add_physical_point(p6, label=17)
    # geometry.add_physical_point(p7, label=18)
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
    geometry.add_physical_line(l0, label=111)
    geometry.add_physical_line(l1, label=112)
    geometry.add_physical_line(l2, label=113)
    geometry.add_physical_line(l3, label=114)
    geometry.add_physical_line(l4, label=115)
    geometry.add_physical_line(l5, label=116)
    geometry.add_physical_line(l6, label=117)
    geometry.add_physical_line(l7, label=118)
    geometry.add_physical_line(l8, label=119)
    geometry.add_physical_line(l9, label=120)
    geometry.add_physical_line(l10, label=121)
    geometry.add_physical_line(l11, label=122)
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




# quad = geometry.add_polygon([
#   [-1.0,-1.0, 0.0],
#   [ 1.0,-1.0, 0.0],
#   [ 1.0, 1.0, 0.0],
#   [-1.0, 1.0, 0.0],
#   ],
#   h)
# # print 'quad', vars(quad)
# # print 'quad.surface', vars(quad.surface)
# # print 'quad.line_loop', vars(quad.line_loop)
# geometry.add_physical_line(quad.line_loop.lines[0], label=11)
# geometry.add_physical_line(quad.line_loop.lines[1], label=22)
# geometry.add_physical_line(quad.line_loop.lines[2], label=33)
# geometry.add_physical_line(quad.line_loop.lines[3], label=44)
# geometry.add_physical_surface(quad.surface, label=111)
# axis = [0, 0, 2]
# top, vol, ext = geometry.extrude(quad.surface, axis)
# print 'vol', vars(vol)
# print 'top', vars(top)
# print 'top.id', top.id
# print 'ext[0]', vars(ext[0])
# geometry.add_physical_surface(top, label=666)
# geometry.add_physical_surface(ext[0], label=222)
# geometry.add_physical_surface(ext[1], label=333)
# geometry.add_physical_surface(ext[2], label=444)
# geometry.add_physical_surface(ext[3], label=555)
# geometry.add_physical_volume(vol, label=9999)
