# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
from simfempy.meshes.simplexmesh import SimplexMesh


#=================================================================#
class Fem(object):
    def __repr__(self):
        repr = f"{self.__class__.__name__} {self.params_bool=} {self.params_str=} {self.params_float=}"
        return repr
    def __init__(self, mesh=None):
        if mesh is not None: self.setMesh(mesh)
        self.params_bool, self.params_str, self.params_float = {}, {}, {}
    def setMesh(self, mesh):
        self.mesh = mesh

# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
