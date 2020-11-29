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
        self.latex = True
        self.plot = False
        self.plotpostprocs = False
        if 'clean' in kwargs and kwargs.pop("clean")==True:
            import os, shutil
            try: shutil.rmtree(os.getcwd() + os.sep + self.dirname)
            except: pass
        if 'latex' in kwargs: self.latex = kwargs.pop("latex")
        if 'plot' in kwargs: self.plot = kwargs.pop("plot")
        if 'plotpostprocs' in kwargs: self.plotpostprocs = kwargs.pop("plotpostprocs")
        if 'verbose' in kwargs:
            verbose = int(kwargs.pop("verbose"))
            if verbose > 0: self.latex = True
            if verbose > 1: self.plotpostprocs = True
            if verbose > 2: self.plot = True
        if 'h' in kwargs:
            self.h = kwargs.pop("h")
            self.paramname = kwargs.pop("paramname")
        else:
            self.paramname = "ncells"
            if 'niter' in kwargs: self.niter = kwargs.pop("niter")
            else: self.niter = -1
        self.parameters = []
    def _mesh_from_geom_or_fct(self, h):
        if hasattr(self,"createMesh"): return self.createMesh(h)
        if hasattr(pygmsh, "built_in"):
            mesh = pygmsh.generate_mesh(self.geom(h), verbose=False)
        else:
            with self.geom(h) as geom:
                mesh = geom.generate_mesh()
        return simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)

    def compare(self, createMesh=None, geom=None, h=None, params=None):
        if createMesh is None: self.geom = geom
        else: self.createMesh = createMesh
        if self.paramname == "ncells":
            if h is None:
                mesh = self._mesh_from_geom_or_fct(self.h)
                gmshrefine = True
                if self.niter ==-1: raise KeyError("please give 'niter'")
                params = [mesh.ncells*mesh.dimension**i for i in range(self.niter)]
            else:
                params = h
                gmshrefine = False
        else:
            mesh = self._mesh_from_geom_or_fct(self.h)
            # mesh = SimplexMesh(geometry=geometry, hmean=self.h)
        for iter, param in enumerate(params):
            if self.paramname == "ncells":
                if gmshrefine:
                    mesh = simfempy.meshes.pygmshext.gmshRefine(mesh)
                else:
                    mesh = self._mesh_from_geom_or_fct(param)
                self.parameters.append(mesh.ncells)
            else:
                self.parameters.append(param)
            for name, method in self.methods.items():
                method.setMesh(mesh)
                self.dim = mesh.dimension
                if self.paramname != "ncells": method.setParameter(self.paramname, param)
                result = method.solve(iter, self.dirname)
                if self.plot:
                    from ..meshes import plotmesh
                    suptitle = "{}={}".format(self.paramname, self.parameters[-1])
                    plotmesh.meshWithData(mesh, data=result.data, title=name, suptitle=suptitle)
                    plt.show()
                resdict = result.info.copy()
                resdict.update(result.data['global'])
                # print(resdict)
                self.fillInfo(iter, name, resdict, len(params))
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
                for key3, info3 in info2.items():
                    self.infos[key2][key3] = {}
                    for name2 in self.methods.keys():
                        self.infos[key2][key3][name2] = np.zeros(shape=(n), dtype=type(info3))
        for key2, info2 in info.items():
            for key3, info3 in info2.items():
                self.infos[key2][key3][name][iter] = np.sum(info3)
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
                kwargs['redrate'] = (key=="error") and (paramname=="ncells")
                kwargs['diffandredrate'] = not kwargs['redrate'] and (paramname=="ncells")
                kwargs['dim'] = self.dim
                if key=="error" and len(self.methods)==1:
                    newdict={}
                    for key2, val2 in val.items():
                        newdict["{}".format(key2)] = val2[name]
                    kwargs['name'] = '{}-{}'.format(key, name)
                    kwargs['values'] = newdict
                    latexwriter.append(**kwargs)
                else:
                    for key2, val2 in val.items():
                        print(f"{key=} {key2=}")
                        kwargs['name'] = '{}-{}'.format(key,key2)
                        kwargs['values'] = val2
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
                    else:
                        axs[cr, cc].plot(parameters, val2[name], '-x', label="{}_{}".format(key2, name))
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