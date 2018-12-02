# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import matplotlib.pyplot as plt

from fempy.tools.latexwriter import LatexWriter
from fempy.meshes.trianglemesh import TriangleMesh


#=================================================================#
class CompareErrors(object):
    def __init__(self, methods, latex=True, vtk=True, clean=True, plot=True):
        self.methods = methods
        self.latex = latex
        self.vtk = vtk
        self.plot = plot
        n = len(methods)
        self.reds = np.outer(np.linspace(0.2,0.8,n),[0,1,1])
        self.reds[:,0] = 1.0
        self.greens = np.outer(np.linspace(0.2,0.8,n),[1,0,1])
        self.greens[:,1] = 1.0
        self.blues = np.outer(np.linspace(0.2,0.8,n),[1,1,0])
        self.blues[:,2] = 1.0
        problemname = "none"
        for name, method in self.methods.items():
            if problemname =="none":
                problemname = method.problem
            else:
                assert problemname == method.problem
        self.dirname = "Results_" + problemname
        if clean:
            import os, shutil
            try:
                shutil.rmtree(os.getcwd()+os.sep+self.dirname)
            except:
                pass

    def compare(self, geomname="unitsquare", h=[1.0, 0.5, 0.25, 0.1, 0.05, 0.025], solve="stat"):
        errors = {}
        times = {}
        nliter = {}
        for name, method in self.methods.items():
            # errors[name] = {}
            nliter[name] = []
            times[name] = {}
            times[name]['rhs'] = []
            times[name]['matrix'] = []
            times[name]['solve'] = []
        ncells = []
        has_nliter=False
        has_errors=False
        for hiter, hs in enumerate(h):
            trimesh = TriangleMesh(geomname=geomname, hmean=hs)
            for name, method in self.methods.items():
                method.setMesh(trimesh)
                if solve=="stat":
                    point_data, cell_data, info = method.solve()
                else:
                    point_data, cell_data, info = method.solvedynamic(name, hiter, self.dirname)
                if hiter == 0:
                    if 'error' in info:
                        has_errors = True
                    if has_errors:
                        for errname in info['error']:
                            if errname not in errors:
                                errors[errname] = {}
                            errors[errname][name] = []
                if has_errors:
                    for errname, err in info['error'].items():
                        errors[errname][name].append(err)
                if 'nit' in info:
                    has_nliter = True
                    nliter[name].append(info['nit'])
                # print('method', name, 'nit', info['nit'])
                times[name]['rhs'].append(info['timer']['rhs'])
                times[name]['matrix'].append(info['timer']['matrix'])
                times[name]['solve'].append(info['timer']['solve'])
                if self.vtk:
                    filename = "{}_{}_{:02d}.vtk".format(method.problem, name, hiter)
                    trimesh.write(filename=filename, dirname=self.dirname, point_data=point_data, cell_data=cell_data)
                if self.plot:
                    from ..meshes import plotmesh
                    plotmesh.meshWithData(trimesh, point_data, cell_data)
                    plt.show()
            ncells.append(trimesh.ncells)
        orders = None
        latexwriter = LatexWriter(dirname=self.dirname)
        for errname, error in errors.items():
            if errname.find('shoot') == -1:
                orders = latexwriter.append(n=ncells, values=error, name='err_'+errname, redrate=True)
            else:
                latexwriter.append(n=ncells, values=error, name='err_' + errname)
        if has_nliter:
            latexwriter.append(n=ncells, values=nliter, name = 'nlit', type='int')
        if self.latex:
            latexwriter.write()
            latexwriter.compile()

        ax = plt.subplot(2,1,1)
        for errname, error in errors.items():
            for name, method in self.methods.items():
                if errname.find('shoot') == -1:
                    plt.loglog(ncells, error[name], '-x', label='err_'+errname+'_'+name)
        neworders=[]
        testrange = np.arange(0.25, 8.5, 0.25)
        for (k, order) in orders.items():
            if order < 0.1 : continue
            apporder = testrange.flat[np.abs(testrange - orders[k]).argmin()]
            neworders.append(apporder)
        neworders = np.unique(neworders)
        dim = 2.0
        for order in neworders:
            plt.loglog(ncells, np.power(ncells, -order/dim), '--', label='order_'+str(order))

        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.legend()
        ax = plt.subplot(2,1,2)
        count=0
        for name, method in self.methods.items():
            plt.loglog(ncells, times[name]['rhs'], '-x', label='rhs_'+name, color=self.blues[count])
            plt.loglog(ncells, times[name]['matrix'], '-x', label='matrix_' + name, color=self.greens[count])
            plt.loglog(ncells, times[name]['solve'], '-x', label='solve_' + name, color=self.reds[count])
            count += 1
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.legend()
        ax.set_title("times")
        plt.tight_layout()
        plt.show()


# ------------------------------------- #

if __name__ == '__main__':
    print("so far no test")