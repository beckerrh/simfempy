# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from .trimesh import TriMesh
import copy

class TriMeshWithEdgePatches(TriMesh):
    def __init__(self, trimesh=None, geomname="unitsquare", hnew=None):
        TriMesh.__init__(self, trimesh=trimesh, geomname=geomname, hnew=hnew)
        # self.edgesx = self.x[self.edges].mean(axis=1)
        # self.edgesy = self.y[self.edges].mean(axis=1)
        # self.xedges = self.mesh.x[self.mesh.edges].mean(axis=1)
        # self.yedges = self.mesh.y[self.mesh.edges].mean(axis=1)
        # xedge = 0.5*( self.x[ self.edges[:,0]]+  self.x[ self.edges[:,1]])
        # yedge = 0.5*( self.y[ self.edges[:,0]]+  self.y[ self.edges[:,1]])
        # assert linalg.norm(self.edgesx-xedge) < 1e-12
        # assert linalg.norm(self.edgesy-yedge) < 1e-12
        self.constructPatches()

    def __str__(self):
        return 'TriMeshWithEdgePatches: nvert/ncells/nedges: %d %d %d' % (len(self.x), len(self.triangles), len(self.edges))

    def testPatches(self):
        problem = False
        for i in range(self.nintedges):
            ie = self.intedges[i]
            for jj in range(4):
                iej = self.patches_ind[i, jj, 0]
                graduS = np.array([self.edgesx[iej] - self.edgesx[ie], self.edgesy[iej] - self.edgesy[ie]])
                i0 = self.patches_ind[i, jj, 1]
                i1 = self.patches_ind[i, jj, 2]
                a0 = self.patches_coef[i, jj, 0]
                a1 = self.patches_coef[i, jj, 1]
                graduR = np.array([a0 * (self.edgesx[i0] - self.edgesx[ie]) + a1 * (self.edgesx[i1] - self.edgesx[ie]),
                                       a0 * (self.edgesy[i0] - self.edgesy[ie]) + a1 * (self.edgesy[i1] - self.edgesy[ie])])
                graduT = graduS - graduR
                if linalg.norm(graduT) > 1e-14:
                    print('graduS', graduS, 'graduR', graduR, 'i=', i)
                    raise ValueError('wrong patch')
        if problem:
            self.plotPatches(plt)
            raise ValueError("wrong")
            # else:
            #     print 'no difference for iv=', iv, 'iv2=', iv2
    def constructPatches(self):
        nintedges = self.nintedges
        self.patches_ind = -np.ones( (nintedges, 4, 3), dtype=int )
        self.patches_coef = np.zeros( (nintedges, 4, 2), dtype=np.float64 )

        for i in range(nintedges):
            ie = self.intedges[i]
            iKl = self.cellsOfEdge[ie, 0]
            iKr = self.cellsOfEdge[ie, 1]
            # edges seront triés, mais séparémment à gauche et à droite
            if iKl < 0:
                edges = np.setxor1d(self.edgesOfCell[iKr], [ie])
            elif iKr < 0:
                edges = np.setxor1d(self.edgesOfCell[iKl], [ie])
            else:
                edges = np.concatenate( (np.setxor1d(self.edgesOfCell[iKl], [ie]),np.setxor1d(self.edgesOfCell[iKr], [ie] )) )
            # print 'ie', ie, 'edges', edges, ' <---- ', self.edgesOfCell[iKl], self.edgesOfCell[iKr]
            S = np.zeros( (4,2), dtype=np.float64 )
            for jj in range(4):
                iej = edges[jj]
                S[jj][0] = self.edgesx[iej] - self.edgesx[ie]
                S[jj][1] = self.edgesy[iej] - self.edgesy[ie]
            SPl = np.vstack((S[0], S[1]))
            SPl = linalg.inv(SPl)
            SPr = np.vstack((S[2], S[3]))
            SPr = linalg.inv(SPr)

            for jj in range(4):
                iej = edges[jj]
                self.patches_ind[i, jj, 0] = iej
                if jj < 2:
                    self.patches_ind[i, jj, 1] = edges[2]
                    self.patches_ind[i, jj, 2] = edges[3]
                    ac = np.dot(SPr.T, S[jj])
                    self.patches_coef[i, jj, 0] = ac[0]
                    self.patches_coef[i, jj, 1] = ac[1]
                else:
                    self.patches_ind[i, jj, 1] = edges[0]
                    self.patches_ind[i, jj, 2] = edges[1]
                    ac = np.dot(SPl.T, S[jj])
                    self.patches_coef[i, jj, 0] = ac[0]
                    self.patches_coef[i, jj, 1] = ac[1]
                # print  self.patches_ind[i]
                # print  self.patches_coef[i]
                if jj < 2:
                    if linalg.norm(ac[0]*S[2] + ac[1]*S[3] - S[jj]) >= 1e-12:
                        print(i, jj, 'S[jj]' , S[jj],  ac[0]*S[0] + ac[1]*S[1])
                        raise ValueError('not ok')
                else:
                    if linalg.norm(ac[0]*S[0] + ac[1]*S[1] - S[jj]) >= 1e-12:
                        print(i, jj, 'S[jj]', S[jj], ac[0] * S[0] + ac[1] * S[1])
                        raise ValueError('not ok')

    class PatchPlotter(object):
        def __init__(self, ax, trimesh):
            self.ax = ax
            assert isinstance(trimesh, TriMeshWithEdgePatches)
            self.trimesh = trimesh
            self.canvas = ax.figure.canvas
            self.cid = self.canvas.mpl_connect('button_press_event', self)

        def __call__(self, event):
            # Ignore clicks outside axes
            if event.inaxes != self.ax.axes:
                return
            xdif = self.trimesh.edgesx[self.trimesh.intedges] - event.xdata
            ydif = self.trimesh.edgesy[self.trimesh.intedges] - event.ydata
            dist = xdif ** 2 + ydif ** 2
            ieint = np.argmin(dist)
            ie = self.trimesh.intedges[ieint]
            tris = self.trimesh.cellsOfEdge[ie]
            self.ax.clear()
            props = dict(boxstyle='round', facecolor='wheat')
            (xe, ye) = self.trimesh.edgesx[ie], self.trimesh.edgesy[ie]
            self.ax.text(xe, ye, r'%d' % (ie), fontweight='bold', bbox=props)
            for it in tris:
                self.ax.text(self.trimesh.centersx[it], self.trimesh.centersy[it], r'%d' % (it), color='r')
            col = ['b', 'r']
            x = [];
            y = [];
            sx = [];
            sy = []
            print('ie=', ie, 'patches_ind', self.trimesh.patches_ind[ieint], 'patches_coef', self.trimesh.patches_coef[ieint])
            for ii in range(4):
                iei = self.trimesh.patches_ind[ieint, ii, 0]
                i0 = self.trimesh.patches_ind[ieint, ii, 1]
                i1 = self.trimesh.patches_ind[ieint, ii, 2]
                a0 = self.trimesh.patches_coef[ieint, ii, 0]
                a1 = self.trimesh.patches_coef[ieint, ii, 1]
                s = a0+a1
                a0 /= s
                a1 /= s
                dx = a0*(self.trimesh.edgesx[i0] - self.trimesh.edgesx[ie]) + a1*(self.trimesh.edgesx[i1] - self.trimesh.edgesx[ie])
                dy = a0*(self.trimesh.edgesy[i0] - self.trimesh.edgesy[ie]) + a1*(self.trimesh.edgesy[i1] - self.trimesh.edgesy[ie])
                self.ax.plot( [xe,self.trimesh.edgesx[iei]], [ye,self.trimesh.edgesy[iei]], '-b')
                self.ax.plot( [xe,xe+dx], [ye,ye+dy], '--g')
                # self.ax.text(self.trimesh.edgesx[iei], self.trimesh.edgesy[iei],'%d - %d' % (i0, i1))
            #     if patch[5] != -1:
            #         iv0 = self.trimesh.edges[patch[5], 0]
            #         iv1 = self.trimesh.edges[patch[5], 1]
            #         xm = 0.5 * (self.trimesh.x[iv0] + self.trimesh.x[iv1])
            #         ym = 0.5 * (self.trimesh.y[iv0] + self.trimesh.y[iv1])
            #         self.ax.text(xm, ym, '%d' % (patch[5]), color='g')
            #     5 * (self.trimesh.edges[patch[5], 0] + self.trimesh.edges[patch[5], 1])
            #     x.append(self.trimesh.x[inode])
            #     y.append(self.trimesh.y[inode])
            #     normal = 0.5 * (
            #     patch[3] * self.trimesh.normals[patch[1]] + patch[4] * self.trimesh.normals[patch[2]])
            #     sx.append(normal[0])
            #     sy.append(normal[1])
            # self.ax.quiver(x, y, sx, sy, headwidth=5, scale=2., units='xy', color='y')
            # self.ax.triplot(self.trimesh, 'k--')
            self.trimesh._preparePlot(self.ax, 'Patches', setlimits=False)
            self.canvas.draw()

    def plotPatches(self, plt):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.plotVerticesAndCells(ax1, False)
        self.plotEdges(ax2, True)
        self._preparePlot(ax3, 'Patches', setlimits=False)
        patchplotter = self.PatchPlotter(ax3, self)
        plt.show()


# ------------------------------------- #

if __name__ == '__main__':
    trimesh = TriMeshWithEdgePatches(hnew=0.9)
    # trimesh.plot(plt, edges=True)
    trimesh.testPatches()
    trimesh.plotPatches(plt)