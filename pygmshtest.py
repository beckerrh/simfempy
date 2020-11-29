#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:04:29 2020

@author: becker
"""
import pygmsh

rect = [-2, 2, -2, 2]
recth = [-1, 1, -1, 1]
h=2
with pygmsh.geo.Geometry() as geom:
    hole = geom.add_rectangle(*recth, z=0, mesh_size=h, make_surface=True)
    geom.add_physical(hole.surface, label="222")
    p = geom.add_rectangle(*rect, z=0, mesh_size=h, holes=[hole])
    geom.add_physical(p.surface, label="111")
    for i in range(len(p.lines)): geom.add_physical(p.lines[i], label=f"{1000+i}")
    mesh = geom.generate_mesh()
print(mesh.cells[0][1].T)
print(mesh.cell_sets.keys())
print([ mesh.cell_sets[f"{1000+i}"][0] for i in range(4)] )

from simfempy.meshes.plotmesh import plotmeshWithNumbering
plotmeshWithNumbering(mesh)
