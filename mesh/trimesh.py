# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import os, sys
import matplotlib.pyplot as plt
import meshio
import matplotlib.tri
import numpy as np
import scipy
from mesh import geometry
import pygmsh
#import vtk
from scipy import linalg


class TriMesh(matplotlib.tri.Triangulation):
    def __init__(self, trimesh=None, geomname="unitsquare", hmean=None):
        if trimesh is not None:
            matplotlib.tri.Triangulation.__init__(self, trimesh.x, trimesh.y, trimesh.triangles)
            self.nedges = trimesh.nedges
            self.ncells = trimesh.ncells
            self.nnodes = trimesh.nnodes
            self.bdryvert = trimesh.bdryvert
            self.nbdryvert = trimesh.nbdryvert
            self.intvert = trimesh.intvert
            self.normals = trimesh.normals
            self.area = trimesh.area
            self.cellsOfEdge = trimesh.cellsOfEdge
            self.edgesOfCell = trimesh.edgesOfCell
            self.centersx = trimesh.centersx
            self.centersy = trimesh.centersy
            self.edgesx = trimesh.edgesx
            self.edgesy = trimesh.edgesy
            self.bdryedges = trimesh.bdryedges
            self.intedges = trimesh.intedges
            self.nbdryedges = trimesh.nbdryedges
            self.nintedges = trimesh.nintedges
            self.bdrylabels = trimesh.bdrylabels
            self.geomname = trimesh.geomname
            return
        self.geomname = geomname
        filenamemsh = geomname + '.msh'
        if hmean is not None or not os.path.isfile(filenamemsh):
            geom = geometry.Geometry(geomname=geomname, h=hmean)
            geom.runGmsh(newgeometry=(hmean is not None))
        mesh = meshio.read(filename=filenamemsh)
        points, cells, celldata = mesh.points, mesh.cells, mesh.cell_data
        bdrylabelsmsh = celldata['line']['gmsh:physical']
        matplotlib.tri.Triangulation.__init__(self, x=points[:, 0], y=points[:, 1], triangles=cells['triangle'])
        self.nedges = len(self.edges)
        self.ncells = len(self.triangles)
        self.nnodes = len(self.x)
        # self.bdryvert = set(self.triangles.flat[np.flatnonzero(self.neighbors == -1)])
        # self.intvert = set(np.arange(self.nnodes)).difference(self.bdryvert)
        self.bdryvert = self.triangles.flat[np.flatnonzero(self.neighbors == -1)]
        self.intvert = np.setxor1d(np.arange(self.nnodes), self.bdryvert)
        self.nbdryvert = len(self.bdryvert)
        self.normals = None
        self.area = None
        self.cellsOfEdge = None
        self.edgesOfCell = None
        self.centersx = self.x[self.triangles].mean(axis=1)
        self.centersy = self.y[self.triangles].mean(axis=1)
        self.edgesx = self.x[self.edges].mean(axis=1)
        self.edgesy = self.y[self.edges].mean(axis=1)
        self.construcCellEdgeConnectivity()
        self.construcNormalsAndAreas()
        # self.bdryedges = set(np.flatnonzero(np.any(self.cellsOfEdge == -1, axis=1)))
        # self.intedges = set(np.arange(self.nedges)).difference(self.bdryedges)
        self.bdryedges = np.flatnonzero(np.any(self.cellsOfEdge == -1, axis=1))
        self.intedges = np.setxor1d(np.arange(self.nedges), self.bdryedges)
        self.nbdryedges = len(self.bdryedges)
        self.nintedges = len(self.intedges)
        self.bdrylabels = None
        self.constructBoundaryEdges(bdrylabelsmsh, cells['line'])
        # self.patches_tris = None
        # self.patches_bnodes = None
        # self.patchinfo = None
        # self.patches_opind = None
        # self.patches_opcoeff = None
        print(self)
    def __str__(self):
        return "TriMesh: nvert/ncells/nedges: %d %d %d bdrylabels=%s" % (len(self.x), len(self.triangles), len(self.edges), list(self.bdrylabels.keys()))
    def write(self, filename, dirname = "out", point_data=None, cell_data=None):
        points = np.zeros((self.nnodes, 3))
        points[:, 0:2] = np.stack((self.x, self.y), axis=-1)
        assert np.all(points[:, 0] == self.x)
        assert np.all(points[:, 1] == self.y)
        cells = {'triangle': self.triangles}
        # if (cell_data is not None) and (type(cell_data) is not dict):
        #     cell_data = {'U': cell_data}
        # if (point_data is not None) and (type(point_data) is not dict):
        #     point_data = {'U': point_data}
        if cell_data is not None:
            cell_data_meshio = {'triangle': cell_data}
        else:
            cell_data_meshio=None
        dirname = dirname + os.sep + "mesh"
        if not os.path.isdir(dirname) :
            os.makedirs(dirname)
        filename = os.path.join(dirname, filename)
        mesh = meshio.Mesh(points, cells)
        meshio.write(filename=filename, mesh=mesh)
        # meshio.write(filename=filename, mesh=mesh, point_data=point_data, cell_data=cell_data_meshio)
        # meshio.write(filename=filename, points=points, cells=cells, point_data=point_data, cell_data=cell_data_meshio,field_data=None)

    def construcNormalsAndAreas(self):
        sidesx = self.y[self.edges[:, 0]] - self.y[self.edges[:, 1]]
        sidesy = self.x[self.edges[:, 1]] - self.x[self.edges[:, 0]]
        self.normals = np.stack((sidesx, sidesy), axis=-1)
        elem = self.triangles
        sidesx0 = self.x[elem[:, 2]] - self.x[elem[:, 1]]
        sidesx1 = self.x[elem[:, 0]] - self.x[elem[:, 2]]
        sidesx2 = self.x[elem[:, 1]] - self.x[elem[:, 0]]
        sidesx = np.stack((sidesx0, sidesx1, sidesx2), axis=-1)
        sidesy0 = self.y[elem[:, 2]] - self.y[elem[:, 1]]
        sidesy1 = self.y[elem[:, 0]] - self.y[elem[:, 2]]
        sidesy2 = self.y[elem[:, 1]] - self.y[elem[:, 0]]
        sidesy = np.stack((sidesy0, sidesy1, sidesy2), axis=-1)
        self.area = 0.5 * np.abs(-sidesx[:, 2] * sidesy[:, 1] + sidesy[:, 2] * sidesx[:, 1])
    def construcCellEdgeConnectivity(self):
        self.cellsOfEdge = -1 * np.ones(shape=(self.nedges, 2), dtype=int)
        self.edgesOfCell = np.zeros(shape=(self.ncells, 3), dtype=int)
        eovs = {}
        for (ie, iv) in enumerate(self.edges):
            iv.sort()
            eovs[tuple(iv)] = ie
        for (it, iv) in enumerate(self.triangles):
            for ii in range(3):
                ivs = [iv[(ii + 1) % 3], iv[(ii + 2) % 3]]
                ivs.sort()
                ie = eovs[tuple(ivs)]
                self.edgesOfCell[it, ii] = ie
                if iv[(ii + 1) % 3] == self.edges[ie, 0] and iv[(ii + 2) % 3] == self.edges[ie, 1]:
                    self.cellsOfEdge[ie, 1] = it
                else:
                    self.cellsOfEdge[ie, 0] = it
                    # print 'self.cellsOfEdge', self.cellsOfEdge
                    # print 'self.edgesOfCell', self.edgesOfCell
    def _preparePlot(self, ax, title='no title', setlimits=False, lw=1, color='k'):
        ax.triplot(self, lw=lw, color=color)
        if setlimits:
            (xmin, xmax) = ax.get_xlim()
            (ymin, ymax) = ax.get_ylim()
            (deltax, deltay) = (xmax - xmin, ymax - ymin)
            ax.set_xlim(xmin - 0.1 * deltax, xmax + 0.1 * deltax)
            ax.set_ylim(ymin - 0.1 * deltay, ymax + 0.1 * deltay)
        ax.set_title(title)
    def plotVertices(self, ax):
        props = dict(boxstyle='round', facecolor='wheat')
        for iv in range(len(self.x)):
            ax.text(self.x[iv], self.y[iv], r'%d' % (iv), fontweight='bold', bbox=props)
    def plotVerticesAndCells(self, ax, plotlocalNumbering=True):
        self._preparePlot(ax, 'Nodes and Cells')
        self.plotVertices(ax)
        for it in range(len(self.triangles)):
            ax.text(self.centersx[it], self.centersy[it], r'%d' % (it), color='r', fontweight='bold')
            if plotlocalNumbering:
                for ii in range(3):
                    iv = self.triangles[it, ii]
                    ax.text(0.75 * self.x[iv] + 0.25 * self.centersx[it], 0.75 * self.y[iv] + 0.25 * self.centersy[it],
                             r'%d' % (ii), color='g', fontweight='bold')
    def plotboundaryLabels(self, ax):
        self._preparePlot(ax, 'Boundary labels')
        colors=['r', 'g', 'b']
        count=0
        for col, ind in self.bdrylabels.items():
            x=[]
            y=[]
            for ii,i in enumerate(ind):
                indv = self.edges[i]
                x.append( self.x[indv])
                y.append( self.y[indv])
                if ii==0:
                    ax.plot(self.x[indv], self.y[indv], color=colors[count%len(colors)], label=str(col), lw=3)
                else:
                    ax.plot(self.x[indv], self.y[indv], color=colors[count % len(colors)], lw=3)
            count += 1
        ax.legend(loc='best')
        # ax.legend().set_draggable(state=None)
        # ax.legend().draggable()

    def plotEdges(self, ax, plotNormals=True):
        self._preparePlot(ax, 'Edges')
        for ie in range(len(self.edges)):
            x = 0.66 * self.x[self.edges[ie, 0]] + 0.34 * self.x[self.edges[ie, 1]]
            y = 0.66 * self.y[self.edges[ie, 0]] + 0.34 * self.y[self.edges[ie, 1]]
            ax.text(x, y, r'%d' % (ie), color='b', fontweight='bold')
        if plotNormals:
            x = 0.5 * (self.x[self.edges[:, 0]] + self.x[self.edges[:, 1]])
            y = 0.5 * (self.y[self.edges[:, 0]] + self.y[self.edges[:, 1]])
            ax.quiver(x, y, self.normals[:, 0], self.normals[:, 1])
    def plotEdgesOfCells(self, ax):
        self._preparePlot(ax, 'Edges of Cells')
        props = dict(boxstyle='round', facecolor='wheat')
        for iv in range(len(self.x)):
            ax.text(self.x[iv], self.y[iv], r'%d' % (iv), fontweight='bold', bbox=props)
        for it in range(len(self.triangles)):
            ax.text(self.centersx[it], self.centersy[it], r'%d' % (it), color='r', fontweight='bold')
            for ii in range(3):
                ie = self.edgesOfCell[it, ii]
                x = 0.35 * (self.x[self.edges[ie, 0]] + self.x[self.edges[ie, 1]])
                y = 0.35 * (self.y[self.edges[ie, 0]] + self.y[self.edges[ie, 1]])
                x += 0.3 * self.centersx[it]
                y += 0.3 * self.centersy[it]
                ax.text(x, y, r'%d(%d)' % (ie,ii), color='g', fontweight='bold')
    def plotCellsOfEdges(self, ax):
        self._preparePlot(ax, 'Cells of Edges')
        props = dict(boxstyle='round', facecolor='wheat')
        for iv in range(len(self.x)):
            ax.text(self.x[iv], self.y[iv], r'%d' % (iv), fontweight='bold', bbox=props)
        for it in range(len(self.triangles)):
            ax.text(self.centersx[it], self.centersy[it], r'%d' % (it), color='r', fontweight='bold')
        for ie in range(len(self.edges)):
            xe = 0.5 * self.x[self.edges[ie, 0]] + 0.5 * self.x[self.edges[ie, 1]]
            ye = 0.5 * self.y[self.edges[ie, 0]] + 0.5 * self.y[self.edges[ie, 1]]
            for ii in range(2):
                ic = self.cellsOfEdge[ie,ii]
                if ic >=0:
                    x = 0.8*xe + 0.2*self.centersx[ic]
                    y = 0.8*ye + 0.2*self.centersy[ic]
                    ax.text(x, y, r'%d (%d)' % (ic, ii), color='g', fontweight='bold')
    def plot(self, plt, edges=False, edgesOfCells=False, cellsOfEdges=False, boundaryLabels=False):
        nplots = 1
        if edges: nplots += 1
        if edgesOfCells: nplots += 1
        if cellsOfEdges: nplots += 1
        if boundaryLabels: nplots += 1
        nrows = (nplots+1)/2
        ncols = min(2,nplots)
        print('nplots', nplots, 'nrows', nrows, 'ncols', ncols)
        fig = plt.figure(figsize=(ncols*5,nrows*5))
        iplot = 1
        ax = fig.add_subplot(nrows, ncols, iplot)
        self.plotVerticesAndCells(ax, True)
        iplot += 1
        if edges:
            ax = fig.add_subplot(nrows, ncols, iplot)
            iplot += 1
            self.plotVerticesAndCells(ax, False)
            self.plotEdges(ax, True)
        if edgesOfCells:
            ax = fig.add_subplot(nrows, ncols, iplot)
            iplot += 1
            self.plotEdgesOfCells(ax)
        if cellsOfEdges:
            ax = fig.add_subplot(nrows, ncols, iplot)
            iplot += 1
            self.plotCellsOfEdges(ax)
        if boundaryLabels:
            ax = fig.add_subplot(nrows, ncols, iplot)
            iplot += 1
            self.plotboundaryLabels(ax)
        plt.show()
    def constructBoundaryEdges(self, bdrylabelsmsh, lines):
        """
        we suppose, self.edges is sorted !!!
        :param bdrylabelsmsh:
        :param lines:
        :return:
        """
        if len(bdrylabelsmsh) != self.nbdryedges:
            raise ValueError("wrong number of boundary labels %d != %d (self.nbdryedges)" %(len(bdrylabelsmsh),self.nbdryedges))
        if len(lines) != self.nbdryedges:
            raise ValueError("wrong number of lines %d != %d (self.nedges)" % (len(lines), self.nbdryedges))
        self.bdrylabels = {}
        colors, counts = np.unique(bdrylabelsmsh, return_counts=True)
        # print ("colors, counts", colors, counts)
        for i in range(len(colors)):
            self.bdrylabels[colors[i]] = -np.ones( (counts[i]), dtype=np.int32)
        lines = np.sort(lines)
        # print("lines", lines)
        # print("self.edges", self.edges)

        n = self.nedges
        A = np.zeros( lines.shape[0], dtype=np.int32)
        B = np.zeros( n, dtype=np.int32)
        for i in range(len(A)):
            A[i] = n*lines[i,0] + lines[i,1]
        for i in range(len(B)):
            B[i] = n*self.edges[i,0] + self.edges[i,1]

        #http://numpy-discussion.10968.n7.nabble.com/How-to-find-indices-of-values-in-an-array-indirect-in1d-td41972.html
        B_sorter = np.argsort(B)
        B_sorted = B[B_sorter]
        B_sorted_index = np.searchsorted(B_sorted, A)

        # Go back into the original index:
        B_index = B_sorter[B_sorted_index]
        valid = B.take(B_index, mode='clip') == A
        if not np.all(valid):
            raise ValueError("Did not find indices", valid)
        toto = B_index[valid]
        counts = {}
        for key in list(self.bdrylabels.keys()): counts[key]=0
        for i in range(len(toto)):
            if np.any(lines[i] != self.edges[toto[i]]):
                raise ValueError("Did not find boundary indices")
            color = bdrylabelsmsh[i]
            self.bdrylabels[color][counts[color]] = toto[i]
            counts[color] += 1
        # print ("self.bdrylabels", self.bdrylabels)
    def plotVtkBdrylabels(self):
        filenamevtk = self.geomname + '.vtk'
        points = np.zeros((self.nnodes, 3))
        points[:, 0:2] = np.stack((self.x, self.y), axis=-1)
        cells = {'triangle': self.triangles}
        # cells['line'] = self.edges[self.bdryedges]
        cells['line'] = self.edges
        # bdrycolors = np.zeros(self.nbdryedges)
        bdrycolors = -10*np.ones(self.nedges + self.ncells)
        for col, list in self.bdrylabels.items():
            for l in list:
                bdrycolors[l] = col
        celldata = {'line': {'bdrylabel': bdrycolors}}
        meshio.write(filenamevtk, points, cells, cell_data=celldata)
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filenamevtk)
        reader.ReadAllScalarsOn()
        reader.Update()
        vtkdata = reader.GetOutput()
        celldata = vtkdata.GetCellData().GetArray(0)
        print('celldata', celldata)
        drange = celldata.GetRange()
        print('drange', drange)

        # scalar_range = vtkdata.GetScalarRange()
        datamapper = vtk.vtkDataSetMapper()
        datamapper.SetInputConnection(reader.GetOutputPort())
        datamapper.SetInputData(vtkdata)
        datamapper.SetScalarRange(drange)
        datamapper.ScalarVisibilityOn()
        lut = vtk.vtkLookupTable()
        lut.SetHueRange([0.6667, 0.0])
        datamapper.SetLookupTable(lut)
        meshActor = vtk.vtkActor()
        meshActor.SetMapper(datamapper)
        meshActor.GetProperty().SetRepresentationToWireframe()
        axes = vtk.vtkAxes()
        axesMapper = vtk.vtkPolyDataMapper()
        axesMapper.SetInputConnection(axes.GetOutputPort())
        axesActor = vtk.vtkActor()
        axesActor.SetMapper(axesMapper)
        ren= vtk.vtkRenderer()
        # ren.AddActor( axesActor )

        dataactor = vtk.vtkActor()
        dataactor.SetMapper(datamapper)
        ren.AddActor( dataactor )

        scalarbaractor = vtk.vtkScalarBarActor()
        scalarbaractor.SetLookupTable(datamapper.GetLookupTable())
        scalarbaractor.SetTitle("bdrylabels")
        ren.AddActor( scalarbaractor )

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
    def plotDownwindInfo(self, downwindinfo):
        nplots = 2
        nrows = (nplots+1)/2
        ncols = min(2,nplots)
        fig = plt.figure(figsize=(ncols*5,nrows*5))
        iplot = 1
        ax = fig.add_subplot(nrows, ncols, iplot)
        self.plotVerticesAndCells(ax, plotlocalNumbering = True)
        iplot += 1
        ax = fig.add_subplot(nrows, ncols, iplot)
        self._preparePlot(ax, 'Downwind points')
        # self.plotVertices(ax)
        for it in range(len(self.triangles)):
            xc = self.centersx[it]
            yc = self.centersy[it]
            i0 = downwindinfo[0][it, 0]
            i1 = downwindinfo[0][it, 1]
            s = downwindinfo[1][it, 0]
            p = downwindinfo[1][it, 1]
            xdown = p * self.x[i0] + (1 - p) * self.x[i1]
            ydown = p * self.y[i0] + (1 - p) * self.y[i1]
            l = np.sqrt((self.centersx[it]-xdown)**2 + (self.centersy[it]-ydown)**2)
            # print 'l, s', l, s
            # ax.text(xdown, ydown, 'X', color='b', fontweight='bold')
            ax.plot(xc, yc, 'X', color='r')
            ax.plot(xdown, ydown, 'X', color='b')
            ax.quiver(xc, yc, xdown-xc, ydown-yc, headwidth=5, scale=1.2, units='xy', color='y')
        plt.show()
    def computeDownwindInfo(self, beta):
        ncells = self.ncells
        assert beta.shape == (ncells, 2)
        downwindindices = -1*np.ones( (ncells, 4) , dtype=np.int64)
        dowindcoefs = np.zeros( (ncells, 2) , dtype=np.float64)
        for ic in range(ncells):
            A = np.zeros(shape=(2, 2), dtype=np.float64)
            b = np.zeros(shape=(2), dtype=np.float64)
            found = False
            for ii in range(3):
                ii0 = ii
                ii1 = (ii+1)%3
                i0 = self.triangles[ic,ii0]
                i1 = self.triangles[ic,ii1]
                A[0, 0] = beta[ic, 0]
                A[1, 0] = beta[ic, 1]
                A[0, 1] = self.x[i1] - self.x[i0]
                A[1, 1] = self.y[i1] - self.y[i0]
                try:
                    A = linalg.inv(A)
                except:
                    print('not inversible')
                    continue
                b[0] = self.x[i1] - self.centersx[ic]
                b[1] = self.y[i1] - self.centersy[ic]
                x = np.dot(A,b)
                if x[0]>=0 and x[1] >= 0 and  x[1] <= 1:
                    found = True
                    downwindindices[ic, 0] = i0
                    downwindindices[ic, 1] = i1
                    downwindindices[ic, 2] = ii0
                    downwindindices[ic, 3] = ii1
                    b2 = np.array([x[1] * self.x[i0] + (1 - x[1]) * self.x[i1] - self.centersx[ic],
                                   x[1] * self.y[i0] + (1 - x[1]) * self.y[i1] - self.centersy[ic]])
                    if scipy.linalg.norm(x[0]*beta[ic]-b2) > 1e-15:
                        print('error in delta', scipy.linalg.norm(x[0]*beta[ic]-b2))
                        print('beta[ic]', beta[ic])
                        print('b2', b2)
                        print('delta', x[0])
                        assert 0
                    dowindcoefs[ic, 0] = x[0]
                    dowindcoefs[ic, 1] = x[1]
            if not found:
                print('problem in cell ', ic, 'beta', beta[ic])
                self.plot(plt)
                import sys
                sys.exit(1)
        return [downwindindices, dowindcoefs]

    # def computeNodePatches(self):
    #     if self.patches_tris is not None:
    #         return
    #     self.constructNodePatchIndices()
    #     self.computeNodePatchInfo()
    #     self.computeNodeSideInfo()
    # def testNodePatchesOld(self):
    #     problem = False
    #     for iv in range(self.nnodes):
    #         boundary = -1 in self.patches_bnodes[iv][:, 5]
    #         # if boundary: continue
    #         for i, patch in enumerate(self.patches_bnodes[iv]):
    #             iv2 = patch[0]
    #         # for i, patch in self.patchinfo[iv].iteritems():
    #         #     iv2 = patch[0][0]
    #             gradS = np.array([self.x[iv2] - self.x[iv], self.y[iv2] - self.y[iv]])
    #             i0 = self.patches_opind[iv][i, 0]
    #             i1 = self.patches_opind[iv][i, 1]
    #             a0 = self.patches_opcoeff[iv][i, 0]
    #             a1 = self.patches_opcoeff[iv][i, 1]
    #             if not boundary:
    #                 if a0 > 1e-8 or a1 > 1e-8:
    #                     print('problem in iv', iv, 'iv2', iv2, 'a0 a1', a0, a1, 'i0, i1', i0, i1)
    #                     problem = True
    #             gradR = np.array([a0 * (self.x[i0] - self.x[iv]) + a1 * (self.x[i1] - self.x[iv]),a0 * (self.y[i0] - self.y[iv]) + a1 * (self.y[i1] - self.y[iv])])
    #             if linalg.norm(gradR-gradS)>1e-14:
    #                 print('iv', iv, 'iv2', iv2)
    #                 print('gradS', gradS, 'gradR', gradR)
    #                 self.plotPatches(plt)
    #                 raise ValueError('wrong')
    #     if problem:
    #         self.plotPatches(plt)
    #         raise ValueError("wrong")
    #             # else:
    #             #     print 'no difference for iv=', iv, 'iv2=', iv2
    # def testNodePatches(self):
    #     problem = False
    #     x, y = self.x, self.y
    #     for iv in range(self.nnodes):
    #         boundary = -1 in self.patches_bnodes[iv][:, 5]
    #         for ii,patch in self.patchinfo[iv].items():
    #             iv2 = patch[0][0]
    #             ie = patch[0][1]
    #             gradS = np.array([x[iv2] - x[iv], y[iv2] - y[iv]])
    #             for ii in range(patch[1].shape[0]):
    #                 i0 = patch[1][ii][0]
    #                 i1 = patch[1][ii][1]
    #                 a0 = patch[2][ii][0]
    #                 a1 = patch[2][ii][1]
    #                 gradR = np.array([a0 * (x[i0] - x[iv]) + a1 * (x[i1] - x[iv]),a0 * (y[i0] - y[iv]) + a1 * (y[i1] - y[iv])])
    #                 if linalg.norm(gradR-gradS)>1e-14:
    #                     print('gradR', gradR)
    #                     print('gradS', gradS)
    #                     raise ValueError('wrong')
    #                     problem = True
    #     if problem:
    #         self.plotPatches(plt)
    #         raise ValueError("wrong")
    #             # else:
    #             #     print 'no difference for iv=', iv, 'iv2=', iv2
    # def constructNodePatchIndices(self):
    #     self.patches_tris = dict((i,[]) for i in range(self.nnodes))
    #     for it in range(self.ncells):
    #         tri = self.triangles[it]
    #         for ii in range(3):
    #             inode = tri[ii]
    #             self.patches_tris[inode].append( (it,ii) )
    #     patches_bsides = {}
    #     self.patches_bnodes = {}
    #     # chercher les edges autour du patch
    #     for iv,patch in self.patches_tris.items():
    #         edges_unique = {}
    #         edges_twice = []
    #         for it,iit in patch:
    #             for ii in range(3):
    #                 ie = self.edgesOfCell[it,ii]
    #                 if ie in list(edges_unique.keys()):
    #                     del edges_unique[ie]
    #                     edges_twice.append(ie)
    #                 else:
    #                     edges_unique[ie] = it
    #         npatch = len(edges_unique)
    #         patches_bsides[iv] = -np.ones((npatch,2), dtype=int)
    #         self.patches_bnodes[iv] = -np.ones((npatch,6), dtype=int)
    #         for ii, (ie, it) in enumerate(edges_unique.items()):
    #             patches_bsides[iv][ii,0] = ie
    #             patches_bsides[iv][ii,1] = it
    #         nodes = []
    #         for ii in range(npatch):
    #             ie = patches_bsides[iv][ii,0]
    #             for iii in range(2):
    #                 nodes.append(self.edges[ie,iii])
    #         nodes = np.unique(nodes)
    #         nodesi = {}
    #         for ii in range(npatch):
    #             nodesi[nodes[ii]]=ii
    #         for ii in range(npatch):
    #             self.patches_bnodes[iv][ii,0] = nodes[ii]
    #         for ii in range(npatch):
    #             ie = patches_bsides[iv][ii, 0]
    #             it = patches_bsides[iv][ii, 1]
    #             sign = 2*(it == self.cellsOfEdge[ie,0])-1
    #             for iii in range(2):
    #                 inode = nodesi[self.edges[ie, iii]]
    #                 if self.patches_bnodes[iv][inode,1] == -1:
    #                     self.patches_bnodes[iv][inode, 1] = ie
    #                     self.patches_bnodes[iv][inode, 3] = sign
    #                 else:
    #                     self.patches_bnodes[iv][inode, 2] = ie
    #                     self.patches_bnodes[iv][inode, 4] = sign
    #             for iii in range(3):
    #                 ie2 = self.edgesOfCell[it,iii]
    #                 if self.edges[ie2,0] == iv:
    #                     if self.edges[ie2,1] in nodesi:
    #                         self.patches_bnodes[iv][nodesi[self.edges[ie2,1]],5] = ie2
    #                 elif self.edges[ie2,1] == iv:
    #                     if self.edges[ie2, 0] in nodesi:
    #                         self.patches_bnodes[iv][nodesi[self.edges[ie2, 0]],5] = ie2
    # def computeNodePatchInfo(self):
    #     self.patchinfo = {}
    #     for iv in range(self.nnodes):
    #         # indices des deux aretes par tri et des tris qui ne contiennent PAS cette arrete
    #         boundary = -1 in self.patches_bnodes[iv][:, 5]
    #         npatch = self.patches_bnodes[iv].shape[0]
    #         tris = self.patches_tris[iv]
    #         ntris = len(tris)
    #         edgeoftri = -np.ones((ntris, 2), dtype=int)
    #         trisofedge = {}
    #         for i in range(npatch):
    #             edge = self.patches_bnodes[iv][i,5]
    #             trisofedge[i] = {i for i in range(ntris)}
    #             for itri in range(ntris):
    #                 tri = tris[itri][0]
    #                 for ii in range(3):
    #                     if edge == self.edgesOfCell[tri,ii]:
    #                         if edgeoftri[itri,0]==-1:
    #                             edgeoftri[itri, 0] = i
    #                             trisofedge[i].remove(itri)
    #                         else:
    #                             edgeoftri[itri, 1] = i
    #                             trisofedge[i].remove(itri)
    #                             break
    #         self.patchinfo[iv] = {}
    #         for i in range(npatch):
    #             ntriinpatch = len(trisofedge[i])
    #             iv2 = self.patches_bnodes[iv][i,0]
    #             ie = self.patches_bnodes[iv][i,5]
    #             self.patchinfo[iv][i] = [ [iv2, ie, -1], -np.ones((ntriinpatch, 2), dtype=int), np.zeros((ntriinpatch, 2), dtype=np.float64)]
    #         # print 'trisofedge', trisofedge
    #         # print 'edgeoftri', edgeoftri
    #         assert not -1 in edgeoftri
    #         # compute all edges
    #         S = np.zeros(shape=(npatch, 2), dtype=np.float64)
    #         # invind = {}
    #         for i,patch in enumerate(self.patches_bnodes[iv]):
    #             # invind[patch[0]] = i
    #             S[i,0] = self.x[patch[0]]-self.x[iv]
    #             S[i,1] = self.y[patch[0]]-self.y[iv]
    #         for itri in range(ntris):
    #             tri = tris[itri][0]
    #             SP = np.vstack((S[edgeoftri[itri, 0]], S[edgeoftri[itri, 1]]))
    #             try:
    #                 SP = linalg.inv(SP)
    #             except:
    #                 print('matrix singular, iv', iv, patch[0])
    #                 print(self.patches_bnodes[iv])
    #                 print('SP', SP)
    #                 self.plotPatches(plt)
    #                 raise ValueError("matrix singular")
    #             for j in range(npatch):
    #                 if itri in trisofedge[j]:
    #                     index = list(trisofedge[j]).index(itri)
    #                     sa = np.dot(SP.T, S[j])
    #                     self.patchinfo[iv][j][1][index][0] = self.patches_bnodes[iv][edgeoftri[itri, 0],0]
    #                     self.patchinfo[iv][j][1][index][1] = self.patches_bnodes[iv][edgeoftri[itri, 1],0]
    #                     self.patchinfo[iv][j][2][index] = sa
    #                     if np.all(sa<=0):
    #                         self.patchinfo[iv][j][0][2] = index
    #         for j in range(npatch):
    #             if self.patchinfo[iv][j][0][2] ==-1:
    #                 if not boundary:
    #                     print('not found', self.patchinfo[iv][j][2])
    #                     print('iv', iv)
    #                     print('j', j, self.patchinfo[iv][j])
    #                     print('patch', self.patchinfo[iv])
    #                     self.plotPatches(plt)
    #                     raise ValueError("opposite edge not found")
    #         # print 'self.patchinfo[iv]', self.patchinfo[iv]
    # def computeNodeSideInfo(self):
    #     self.patches_opind = {}
    #     self.patches_opcoeff = {}
    #     for iv in range(self.nnodes):
    #         boundary = -1 in self.patches_bnodes[iv][:, 5]
    #         # if boundary: continue
    #         npatch = self.patches_bnodes[iv].shape[0]
    #         self.patches_opind[iv] = -np.ones((npatch,2), dtype=int)
    #         self.patches_opcoeff[iv] = np.zeros((npatch,2), dtype=np.float64)
    #         assert npatch==len(self.patchinfo[iv])
    #         for iii,patch in self.patchinfo[iv].items():
    #             if patch[0][1] == -1: continue
    #             found = False
    #             for ii in range(patch[1].shape[0]):
    #                 a0 = patch[2][ii][0]
    #                 a1 = patch[2][ii][1]
    #                 if a0<=0.0 and a1<=0.0:
    #                     if found:
    #                         print("problem in iv={} a0={} a1={}".format(iv, a0, a1))
    #                         print(self.patches_opcoeff[iv][0], self.patches_opcoeff[iv][1])
    #                         print(patch)
    #                         self.plotNodePatches(plt)
    #                         raise ValueError("problem !")
    #                     found = True
    #                     self.patches_opind[iv][iii, 0] = patch[1][ii, 0]
    #                     self.patches_opind[iv][iii, 1] = patch[1][ii, 1]
    #                     self.patches_opcoeff[iv][iii, 0] = a0
    #                     self.patches_opcoeff[iv][iii, 1] = a1
    #             if not found:
    #                 # print 'iv', iv, 'patch', patch
    #                 # print patch[2][:,0]+patch[2][:,1]
    #                 if patch[1].shape[0]==0:
    #                     self.patches_opind[iv][iii, 0] = patch[0][0]
    #                     self.patches_opind[iv][iii, 1] = iv
    #                     self.patches_opcoeff[iv][iii, 0] = 1.0
    #                     self.patches_opcoeff[iv][iii, 1] = 0.0
    #                 else:
    #                     ii = np.argmin( patch[2][:,0]+patch[2][:,1] )
    #                 # print iii, npatch, ii
    #                 # print self.patches_opind[iv]
    #                 # print patch[2]
    #                 # a0 = patch[2][ii][0]
    #                     self.patches_opind[iv][iii, 0] = patch[1][ii, 0]
    #                     self.patches_opind[iv][iii, 1] = patch[1][ii, 1]
    #                     self.patches_opcoeff[iv][iii, 0] = patch[2][ii, 0]
    #                     self.patches_opcoeff[iv][iii, 1] = patch[2][ii, 1]
    #                 # else:
    #                 #     print 'iv', iv
    #                 #     print 'iv2', patch[0][0], '(ie)', patch[0][1]
    #                 #     print patch
    #                 #     self.plotPatches(plt)
    #                 #     stop
    # class PatchPlotterEdges(object):
    #     def __init__(self, ax, trimesh):
    #         self.ax = ax
    #         assert isinstance(trimesh, TriMesh)
    #         self.trimesh = trimesh
    #         self.canvas = ax.figure.canvas
    #         self.cid = self.canvas.mpl_connect('button_press_event', self)
    #     def __call__(self, event):
    #         # Ignore clicks outside axes
    #         if event.inaxes != self.ax.axes:
    #             return
    #         xdif = self.trimesh.x- event.xdata
    #         ydif = self.trimesh.y- event.ydata
    #         dist = xdif**2 + ydif**2
    #         iv = np.argmin(dist)
    #         tris =  [x[0] for x in self.trimesh.patches_tris[iv]]
    #         ntri = self.trimesh.ncells
    #         self.ax.clear()
    #         props = dict(boxstyle='round', facecolor='wheat')
    #         self.ax.text(self.trimesh.x[iv], self.trimesh.y[iv], r'%d' % (iv), fontweight='bold', bbox=props)
    #         for it in tris:
    #             self.ax.text(self.trimesh.centersx[it], self.trimesh.centersy[it], r'%d' % (it), color='r', fontweight='bold')
    #         col = ['b', 'r']
    #         x=[]; y=[]; sx=[]; sy=[]
    #         print('iv=', iv, 'bnodes', self.trimesh.patches_bnodes[iv])
    #         for patch in self.trimesh.patches_bnodes[iv]:
    #             inode = patch[0]
    #             self.ax.text(self.trimesh.x[inode], self.trimesh.y[inode], '%d - %d' %(patch[1], patch[2]))
    #             if patch[5] != -1:
    #                 iv0 = self.trimesh.edges[patch[5], 0]
    #                 iv1 = self.trimesh.edges[patch[5], 1]
    #                 xm = 0.5*(self.trimesh.x[iv0] + self.trimesh.x[iv1])
    #                 ym = 0.5*(self.trimesh.y[iv0] + self.trimesh.y[iv1])
    #                 self.ax.text(xm, ym, '%d' %(patch[5]), color='g')
    #             5 * (self.trimesh.edges[patch[5], 0] + self.trimesh.edges[patch[5], 1])
    #             x.append(self.trimesh.x[inode])
    #             y.append(self.trimesh.y[inode])
    #             normal = 0.5*( patch[3]*self.trimesh.normals[patch[1]] + patch[4]*self.trimesh.normals[patch[2]])
    #             sx.append(normal[0])
    #             sy.append(normal[1])
    #         self.ax.quiver(x, y, sx, sy, headwidth=5, scale=1., units='xy', color='y')
    #         self.ax.triplot(self.trimesh, 'k--')
    #         self.trimesh._preparePlot(self.ax, 'Patches', setlimits=True)
    #         self.canvas.draw()
    # class PatchPlotterNodes(object):
    #     def __init__(self, ax, trimesh):
    #         self.ax = ax
    #         assert isinstance(trimesh, TriMesh)
    #         self.trimesh = trimesh
    #         self.canvas = ax.figure.canvas
    #         self.cid = self.canvas.mpl_connect('button_press_event', self)
    #     def __call__(self, event):
    #         # Ignore clicks outside axes
    #         if event.inaxes != self.ax.axes:
    #             return
    #         xdif = self.trimesh.x- event.xdata
    #         ydif = self.trimesh.y- event.ydata
    #         dist = xdif**2 + ydif**2
    #         iv = np.argmin(dist)
    #         print('iv', iv, 'patch', self.trimesh.patchinfo[iv])
    #         tris =  [x[0] for x in self.trimesh.patches_tris[iv]]
    #         ntri = self.trimesh.ncells
    #         self.ax.clear()
    #         props = dict(boxstyle='round', facecolor='wheat')
    #         self.ax.text(self.trimesh.x[iv], self.trimesh.y[iv], r'%d' % (iv), fontweight='bold', bbox=props)
    #         for patch in self.trimesh.patches_bnodes[iv]:
    #             inode = patch[0]
    #             self.ax.text(self.trimesh.x[inode], self.trimesh.y[inode], '%d' %(inode), fontweight='bold', bbox=props, color='b')
    #         self.ax.triplot(self.trimesh, 'k--')
    #         self.trimesh._preparePlot(self.ax, 'Patches', setlimits=True)
    #         self.canvas.draw()
    # def plotNodePatches(self, plt):
    #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    #     self.plotVerticesAndCells(ax1, False)
    #     self.plotEdges(ax2, True)
    #     self._preparePlot(ax3, 'Patches Edges', setlimits=False)
    #     self._preparePlot(ax4, 'Patches Nodes', setlimits=False)
    #     patchplotteredges = self.PatchPlotterEdges(ax3, self)
    #     patchplotternodes = self.PatchPlotterNodes(ax4, self)
    #     plt.show()
    # def plotDownwindEdgeInfo(self, downwindinfo, downwindinfoedge):
    #     nplots = 4
    #     nrows = (nplots+1)/2
    #     ncols = min(2,nplots)
    #     fig = plt.figure(figsize=(ncols*3,nrows*3))
    #     iplot = 1
    #     ax = fig.add_subplot(nrows, ncols, iplot)
    #     self.plotVerticesAndCells(ax, plotlocalNumbering = True)
    #     iplot += 1
    #     ax = fig.add_subplot(nrows, ncols, iplot)
    #     self._preparePlot(ax, 'Edges')
    #     self.plotVerticesAndCells(ax, False)
    #     self.plotEdges(ax, True)
    #     iplot += 1
    #     ax = fig.add_subplot(nrows, ncols, iplot)
    #     self._preparePlot(ax, 'Downwind points')
    #     self.plotVertices(ax)
    #     for it in range(len(self.triangles)):
    #         ax.text(self.centersx[it], self.centersy[it], 'X', color='r', fontweight='bold')
    #         i0 = downwindinfo[0][it, 0]
    #         i1 = downwindinfo[0][it, 1]
    #         s = downwindinfo[1][it, 0]
    #         p = downwindinfo[1][it, 1]
    #         xdown = p * self.x[i0] + (1 - p) * self.x[i1]
    #         ydown = p * self.y[i0] + (1 - p) * self.y[i1]
    #         l = np.sqrt((self.centersx[it]-xdown)**2 + (self.centersy[it]-ydown)**2)
    #         # print 'l, s', l, s
    #         ax.text(xdown, ydown, 'X', color='b', fontweight='bold')
    #     iplot += 1
    #     ax = fig.add_subplot(nrows, ncols, iplot)
    #     self._preparePlot(ax, 'Downwind edges')
    #     self.plotVertices(ax)
    #     for ie in range(self.nedges):
    #         iv0 = self.edges[ie, 0]
    #         iv1 = self.edges[ie, 1]
    #         pindices, pcoefs = downwindinfoedge[0][ie], downwindinfoedge[1][ie]
    #         p = pcoefs[0][0]
    #         w = pcoefs[0][1]
    #         # if np.all( pindices==-1): continue
    #         xe = [ self.x[iv0], self.x[iv1] ]
    #         ye = [ self.y[iv0], self.y[iv1] ]
    #         plt.plot( xe, ye, color='r')
    #         plt.text(np.mean(xe), np.mean(ye), r"p=%g" %p)
    #         # print 'ie', ie, 'iv0, iv1', iv0, iv1, 'p', p
    #     plt.show()


# ------------------------------------- #

if __name__ == '__main__':
    trimesh = TriMesh(geomname="backwardfacingstep", hmean=0.7)
    trimesh.plot(plt, boundaryLabels=True)
    # print('0')
    # test = "simple"
    # # test = "nodepatches"
    # if test == "simple":
    #     # trimesh = TriMesh(hnew=0.8)
    #     trimesh = TriMesh(geomname="backwardfacingstep", hnew=0.7)
    #     # trimesh.plot(plt, edges=True, edgesOfCells=True, cellsOfEdges=True)
    #     trimesh.plot(plt, boundaryLabels=True)
    #     # trimesh.plotVtkBdrylabels()
    # else:
    #     trimesh = TriMesh(hnew=0.9)
    #     trimesh.computeNodePatches()
    #     trimesh.testNodePatches()
    #     trimesh.testNodePatchesOld()
    #     trimesh.plotNodePatches()
