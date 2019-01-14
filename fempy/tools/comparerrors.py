# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import matplotlib.pyplot as plt

from fempy.tools.latexwriter import LatexWriter
from fempy.meshes.simplexmesh import SimplexMesh


#=================================================================#
class CompareErrors(object):
    def __init__(self, methods, **kwargs):
        self.methods = methods
        # check that every method solves the same problem
        self.problemname = "none"
        for name, method in self.methods.items():
            if self.problemname =="none":
                try: self.problemname = method.problemname
                except: pass
            else:
                assert self.problemname == method.problemname
        self.dirname = "Results_" + self.problemname
        import os
        print("self.dirname=", self.dirname, "", os.getcwd())
        self.latex = True
        self.vtk = True
        self.plot = True
        self.plotpostprocs = True
        if 'clean' in kwargs and kwargs.pop("clean")==True:
            import os, shutil
            try: shutil.rmtree(os.getcwd() + os.sep + self.dirname)
            except: pass
        if 'latex' in kwargs: self.latex = kwargs.pop("latex")
        if 'vtk' in kwargs: self.vtk = kwargs.pop("vtk")
        if 'plot' in kwargs: self.plot = kwargs.pop("plot")
        if 'plotpostprocs' in kwargs: self.plotpostprocs = kwargs.pop("plotpostprocs")
        if 'hmean' in kwargs:
            self.hmean = kwargs.pop("hmean")
            self.paramname = kwargs.pop("paramname")
        else:
            self.paramname = "ncells"
        self.parameters = []
        self.infos = None
        
    def compare(self, geomname="unitsquare", h=None, params=None):
        if self.paramname == "ncells":
            params = h
        else:
            mesh = SimplexMesh(geomname=geomname, hmean=self.hmean)
        for iter, param in enumerate(params):
            if self.paramname == "ncells":
                mesh = SimplexMesh(geomname=geomname, hmean=param)
                self.parameters.append(mesh.ncells)
            else:
                self.parameters.append(param)
            for name, method in self.methods.items():
                method.setMesh(mesh)
                self.dim = mesh.dimension
                point_data, cell_data, info = method.solve(iter, self.dirname)
                if self.vtk:
                    filename = "{}_{}_{:02d}.vtk".format(self.problemname, name, iter)
                    mesh.write(filename=filename, dirname=self.dirname, point_data=point_data, cell_data=cell_data)
                if self.plot:
                    from ..meshes import plotmesh
                    suptitle = "{}={}".format(self.paramname, self.parameters[-1])
                    plotmesh.meshWithData(mesh, point_data, cell_data, title=name, suptitle=suptitle)
                self.fillInfo(iter, name, info, len(params))
        if self.plotpostprocs:
            self.plotPostprocs(self.methods.keys(), self.paramname, self.parameters, self.infos)
        if self.latex:
            self.generateLatex(self.methods.keys(), self.paramname, self.parameters, self.infos)
        return  self.methods.keys(), self.paramname, self.parameters, self.infos
        
    def fillInfo(self, iter, name, info, n):
        if not self.infos:
            # print("info.keys", info.keys())
            self.infos = {}
            for key2, info2 in info.items():
                # print("key2", key2)
                self.infos[key2] = {}
                for key3, info3 in info2.items():
                    self.infos[key2][key3] = {}
                    # print("key3", key3,"info3", info3)
                    for name2 in self.methods.keys():
                        self.infos[key2][key3][name2] = np.zeros(shape=(n), dtype=type(info3))
                    self.infos[key2][key3][name][iter] = info3
        for key2, info2 in info.items():
            for key3, info3 in info2.items():
                # for name in self.methods.keys():
                self.infos[key2][key3][name][iter] = info3
                
    def generateLatex(self, names, paramname, parameters, infos):
        latexwriter = LatexWriter(dirname=self.dirname)
        for key, val in infos.items():
            redrate = (key=="error") and (paramname=="ncells")
            if key == 'runinfo':
                newdict={}
                for key2, val2 in val.items():
                    for name in names:
                        newdict["{}:{}".format(key2, name)] = val2[name]
                latexwriter.append(n=parameters, values=newdict, name='{}'.format(key))
            elif key == 'timer':
                for name in names:
                    newdict={}
                    for key2, val2 in val.items():
                        newdict["{}".format(key2)] = val2[name]
                    latexwriter.append(n=parameters, values=newdict, name='{}_{}'.format(name, key), percentage=True)
            else:
                for key2, val2 in val.items():
                    latexwriter.append(n=parameters, values=val2, name='{}_{}'.format(key,key2), dim=self.dim, redrate=redrate, diffandredrate=not redrate)
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
        singleplots = ['timer', 'runinfo']
        nplotsc = len(infos.keys())
        nplotsr = 0
        for key, val in infos.items():
            if key in singleplots: number=1
            else: number=len(val.keys())
            nplotsr = max(nplotsr, number)
        fig, axs = plt.subplots(nplotsr, nplotsc, figsize=(nplotsc * 4, nplotsr * 4), squeeze=False)
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