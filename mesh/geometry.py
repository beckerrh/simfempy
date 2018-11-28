from __future__ import print_function, division
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""


if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    base =  path.dirname(path.dirname(path.abspath(__file__)))
    sys.path.insert(0, base)

import os, sys, subprocess
import pygmsh
import meshio
#import vtk
import importlib


# ------------------------------------- #

class Geometry(object):
    def __init__(self, definition=None, geomname=None, h=0.5):
        self.geometry = pygmsh.built_in.Geometry()
        if geomname is None:
            assert definition
        else:
            assert definition is None
            module = importlib.import_module('mesh.'+geomname)
            definition = module.definition
        self.name, self.geometry = definition(self.geometry, h)
        self.gmsh_executable = 'gmsh'
    def createGeoFile(self):
        if self.geometry is None:
            raise ValueError("self.geometry has to be defined in %s.py" % self.name)
        file = open(self.name + '.geo', "w")
        # file.write(self.geometry.get_code().encode())
        file.write(self.geometry.get_code())
        file.close()
    def runGmsh(self, verbose=False, newgeometry=False):
        filenamegeo = self.name + '.geo'
        if newgeometry or not os.path.isfile(filenamegeo):
            self.createGeoFile()
        filenamemsh = self.name+'.msh'
        cmd = [self.gmsh_executable, '-3', filenamegeo, '-o', filenamemsh]
        print('cmd', ' '.join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = p.communicate()
        if verbose:
            print("stderr", stderr)
            print("stdout", stdout)
        if p.returncode != 0:
            raise RuntimeError('Gmsh exited with error (return code %d).' %p.returncode)
    def showVtk(self, rungmsh=False):
        filenamemsh = self.name + '.msh'
        if rungmsh or not os.path.isfile(filenamemsh):
            self.runGmsh(newgeometry=True)
        # points, cells, pointdata, celldata, fielddata = meshio.read(filenamemsh)
        mesh = meshio.read(filename=filenamemsh)
        points, cells, celldata = mesh.points, mesh.cells, mesh.cell_data
        # print ('cells', cells)
        # print ('celldata', celldata)
        filenamevtk = self.name + '.vtk'
        meshio.write(filenamevtk, mesh)
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filenamevtk)
        reader.ReadAllScalarsOn()
        reader.Update()
        vtkdata = reader.GetOutput()
        scalar_range = vtkdata.GetScalarRange()
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(vtkdata)
        mapper.SetScalarRange(scalar_range)
        meshActor = vtk.vtkActor()
        meshActor.SetMapper(mapper)
        meshActor.GetProperty().SetRepresentationToWireframe()
        axes = vtk.vtkAxes()
        axesMapper = vtk.vtkPolyDataMapper()
        axesMapper.SetInputConnection(axes.GetOutputPort())
        axesActor = vtk.vtkActor()
        axesActor.SetMapper(axesMapper)
        ren= vtk.vtkRenderer()
        # ren.AddActor( axesActor )
        ren.SetBackground( 0.1, 0.2, 0.4 )
        ren.AddActor( meshActor )
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer( ren )
        WinSize = 600
        renWin.SetSize( WinSize, WinSize )
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        iren.Initialize()
        iren.Start()


# ------------------------------------- #

if __name__ == '__main__':
    # geom = Geometry(geomname="unitsquare")
    geom = Geometry(geomname="backwardfacingstep")
    geom.showVtk()


