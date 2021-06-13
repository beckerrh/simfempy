# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016
@author: becker
"""
import numpy as np
import matplotlib.pyplot as plt
import pygmsh
from simfempy.tools.latexwriter import LatexWriter
import simfempy.meshes.pygmshext
#=================================================================#
class CompareMethods(object):
    """
    Run several times a list of methods (typically for comparison of different discretizations on a sequence of meshes)
    possible parameters:
      latex
      vtk
      plot
      plotpostprocs
      verb: in [0,5]
    """
    def __init__(self, methods, **kwargs):
        self.methods = methods
        self.dirname = "Results"
        if 'clean' in kwargs and kwargs.pop("clean")==True:
            import os, shutil
            try: shutil.rmtree(os.getcwd() + os.sep + self.dirname)
            except: pass
        self.verbose = kwargs.pop("verbose", 1)
        self.latex = kwargs.pop("latex", True)
        self.plotsolution = kwargs.pop("plotsolution", False)
        self.plotpostprocs = kwargs.pop("plotpostprocs", False)
        if self.verbose == 0: self.latex = False
        self.paramname = kwargs.pop("paramname", "ncells")
        self.plot = kwargs.pop("plot", False)
        self.createMesh = kwargs.pop("createMesh", None)
        self.geom = kwargs.pop("geom", None)
        self.mesh = kwargs.pop("mesh", None)
        self.h = kwargs.pop("h", None)
        self.parameters = []
    def _mesh_from_geom_or_fct(self, h=None):
        if h is None:
            if self.mesh is not None: return self.mesh
            if self.h is None: raise ValueError(f"I need h({self.h=})")
            h = self.h
        if self.createMesh is not None: return self.createMesh(h)
        if hasattr(pygmsh, "built_in"):
            mesh = pygmsh.generate_mesh(self.geom(h), verbose=False)
        else:
            with self.geom(h) as geom:
                mesh = geom.generate_mesh()
        return simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)
    def compare(self, h=None, params=None, niter=None):
        if self.paramname == "ncells":
            if h is None:
                mesh = self._mesh_from_geom_or_fct()
                gmshrefine = True
                if niter is None: raise KeyError("please give 'niter' ({self.paramname=}")
                params = [mesh.ncells*mesh.dimension**i for i in range(niter)]
            else:
                params = h
                gmshrefine = False
        else:
            mesh = self._mesh_from_geom_or_fct()
            # mesh = SimplexMesh(geometry=geometry, hmean=self.h)
        if self.plotsolution:
            import matplotlib.gridspec as gridspec
            fig = plt.figure(figsize=(10, 8))
            outer = gridspec.GridSpec(1, len(params)*len(self.methods), wspace=0.2, hspace=0.2)
            plotcount = 0
        for iter, param in enumerate(params):
            if self.verbose: print(f"{iter:2d} {self.paramname=} {param=}")
            if self.paramname == "ncells":
                if gmshrefine:
                    mesh = simfempy.meshes.pygmshext.gmshRefine(mesh)
                else:
                    mesh = self._mesh_from_geom_or_fct(param)
                self.parameters.append(mesh.ncells)
            else:
                self.parameters.append(param)
            for name, method in self.methods.items():
                if self.verbose: print(f"{method:-}")
                method.setMesh(mesh)
                self.dim = mesh.dimension
                if self.paramname != "ncells": method.setParameter(self.paramname, param)
                result = method.solve(self.dirname)
                if self.plotsolution:
                    from simfempy.meshes import plotmesh
                    suptitle = "{}={}".format(self.paramname, self.parameters[-1])
                    plotmesh.meshWithData(mesh, data=result.data, title=name, suptitle=suptitle, fig=fig, outer=outer[plotcount])
                    plotcount += 1
                    # plt.show()
                resdict = result.info.copy()
                resdict.update(result.data['global'])
                self.fillInfo(iter, name, resdict, len(params))
        if self.plotsolution: plt.show()
        if self.plotpostprocs:
            self.plotPostprocs(self.methods.keys(), self.paramname, self.parameters, self.infos)
        if self.latex:
            self.generateLatex(self.methods.keys(), self.paramname, self.parameters, self.infos)
        return  self.methods.keys(), self.paramname, self.parameters, self.infos
    def fillInfo(self, iter, name, info, n):
        if not hasattr(self, 'infos'):
            # first time - we have to generate some data
            self.infos = {}
            for key2, info2 in info.items():
                self.infos[key2] = {}
                if isinstance(info2, dict):
                    for key3, info3 in info2.items():
                        self.infos[key2][key3] = {}
                        for name2 in self.methods.keys():
                            self.infos[key2][key3][name2] = np.zeros(shape=(n), dtype=type(info3))
                elif isinstance(info2, simfempy.tools.timer.Timer):
                    for key3, info3 in info2.data.items():
                        self.infos[key2][key3] = {}
                        for name2 in self.methods.keys():
                            self.infos[key2][key3][name2] = np.zeros(shape=(n), dtype=type(info3))
                else:
                    for name2 in self.methods.keys():
                        self.infos[key2][name2] = np.zeros(shape=(n), dtype=type(info2))
        for key2, info2 in info.items():
            if isinstance(info2, dict):
                for key3, info3 in info2.items():
                    self.infos[key2][key3][name][iter] = np.sum(info3)
            elif isinstance(info2, simfempy.tools.timer.Timer):
                for key3, info3 in info2.data.items():
                    self.infos[key2][key3][name][iter] = np.sum(info3)
            else:
                self.infos[key2][name][iter] = np.sum(info2)

    def generateLatex(self, names, paramname, parameters, infos):
        mesh = self.createMesh.__name__
        title = f"mesh({mesh})\\\\"
        for name, method in self.methods.items():
            title += f"{name}\\\\"
        title = title[:-2]
        # print("title = ", title)
        latexwriter = LatexWriter(dirname=self.dirname, title=title, author=self.__class__.__name__)
        for key, val in infos.items():
            kwargs = {'n': parameters, 'nname': paramname}
            keysplit = key.split('_')
            if key == 'iter':
                newdict={}
                for key2, val2 in val.items():
                    for name in names:
                        newdict["{}-{}".format(key2, name)] = val2[name]
                kwargs['name'] = '{}'.format(key)
                kwargs['values'] = newdict
                kwargs['valformat'] = '3d'
                latexwriter.append(**kwargs)
            elif key == 'timer':
                for name in names:
                    newdict={}
                    for key2, val2 in val.items():
                        newdict["{}".format(key2)] = val2[name]
                    kwargs['name'] = '{}-{}'.format(key, name)
                    kwargs['values'] = newdict
                    kwargs['percentage'] = True
                    latexwriter.append(**kwargs)
            else:
                iserr = len(keysplit) >= 2 and keysplit[0] == 'err'
                kwargs['redrate'] = iserr and (paramname=="ncells")
                kwargs['diffandredrate'] = not kwargs['redrate'] and (paramname=="ncells")
                kwargs['dim'] = self.dim
                kwargs['name'] = '{}'.format(key)
                kwargs['values'] = val
                latexwriter.append(**kwargs)
        latexwriter.write()
        latexwriter.compile()
    def computeOrder(self, ncells, values, dim):
        fnd = float(ncells[-1]) / float(ncells[0])
        order = -dim * np.log(values[-1] / values[0]) / np.log(fnd)
        return np.power(ncells, -order / dim), np.round(order,2)
    def plotPostprocs(self, names, paramname, parameters, infos):
        nmethods = len(names)
        self.reds = np.outer(np.linspace(0.2,0.8,nmethods),[0,1,1])
        self.reds[:,0] = 1.0
        self.greens = np.outer(np.linspace(0.2,0.8,nmethods),[1,0,1])
        self.greens[:,1] = 1.0
        self.blues = np.outer(np.linspace(0.2,0.8,nmethods),[1,1,0])
        self.blues[:,2] = 1.0
        singleplots = ['timer', 'iter']
        nplotsc = len(infos.keys())
        nplotsr = 0
        for key, val in infos.items():
            if key in singleplots: number=1
            else: number=len(val.keys())
            nplotsr = max(nplotsr, number)
        fig, axs = plt.subplots(nplotsr, nplotsc, figsize=(nplotsc * 3, nplotsr * 3), squeeze=False)
        cc = 0
        for key, val in infos.items():
            cr = 0
            for key2, val2 in val.items():
                for name in names:
                    if key == "error":
                        axs[cr,cc].loglog(parameters, val2[name], '-x', label="{}_{}".format(key2, name))
                        if self.paramname == "ncells":
                            orders, order = self.computeOrder(parameters, val2[name], self.dim)
                            axs[cr, cc].loglog(parameters, orders, '-', label="order {}".format(order))
                    # else:
                    #     axs[cr, cc].plot(parameters, val2[name], '-x', label="{}_{}".format(key2, name))
                axs[cr, cc].legend()
                if key not in singleplots:
                    axs[cr, cc].set_title("{} {}".format(key, key2))
                    cr += 1
            if key in singleplots:
                axs[cr, cc].set_title("{}".format(key))
                cr += 1
            cc += 1
        plt.tight_layout()
        plt.show()
# ------------------------------------- #
if __name__ == '__main__':
    print("so far no test")