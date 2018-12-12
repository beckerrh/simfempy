# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import os, sys, subprocess
import pygmsh
import importlib
try:
    import geomdefs
except ModuleNotFoundError:
    from . import geomdefs as geomdefs


#=================================================================#
class Geometry(pygmsh.built_in.Geometry):
    """
    wraps pygmsh.built_in.Geometry to control .geo and .msh files
    """
    def __init__(self, definition=None, geomname=None, h=0.5):
        pygmsh.built_in.Geometry.__init__(self)
        if geomname is None:
            assert definition
        else:
            assert definition is None
            fempypath = os.path.dirname(os.path.abspath(__file__))
            sys.path.append(fempypath)
            try:
                module = importlib.import_module('geomdefs.'+geomname)
            except:
                print("Could not import '{}'. Having:\n".format('geomdefs.'+geomname))
                for module in sys.modules.keys():
                    if 'fempy' in module or 'geomdefs' in module:
                        print(module)
                sys.exit(1)
            definition = module.definition
            self.geomname = geomname
        definition(self, h)
        self.gmsh_executable = 'gmsh'
    def writeGeoFile(self, filename):
        file = open(filename, "w")
        # file.write(self.geometry.get_code().encode())
        file.write(self.get_code())
        file.close()
    def runGmsh(self, verbose=False, newgeometry=False):
        filenamegeo = self.geomname + '.geo'
        if newgeometry or not os.path.isfile(filenamegeo):
            self.writeGeoFile(filenamegeo)
        filenamemsh = self.geomname+'.msh'
        cmd = [self.gmsh_executable, '-3', filenamegeo, '-o', filenamemsh]
        print('cmd', ' '.join(cmd))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = p.communicate()
        if verbose:
            print("stderr", stderr)
            print("stdout", stdout)
        if p.returncode != 0:
            raise RuntimeError('Gmsh exited with error (return code %d).' %p.returncode)


# ------------------------------------- #

if __name__ == '__main__':
    geom = Geometry(geomname="backwardfacingstep")
    meshdata = pygmsh.generate_mesh(geom)
    import plotmesh
    import matplotlib.pyplot as plt
    fig, axarr = plt.subplots(2, 1, sharex='col')
    plotmesh.meshWithBoundaries(meshdata, ax=axarr[0])
    plt.show()

