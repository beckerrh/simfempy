# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import pygmsh

# ------------------------------------- #
class Geometry(pygmsh.built_in.Geometry):
    """
    Simple wrap for pygmsh.Geometry in order to define to be called several times
    Derived geometries must define "def define(h)". Other parameters should be passed through the consutructor
    """
    def __init__(self, h=None):
        super().__init__()
        self.define(h)

    def __repr__(self):
        return self.__class__.__name__

    def reset(self):
        super().__init__()
