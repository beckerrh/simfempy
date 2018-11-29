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


class TriangleMesh(matplotlib.tri.Triangulation):
    def __init__(self, geomname="unitsquare", hmean=None):
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
        self.bdryedges = np.flatnonzero(np.any(self.cellsOfEdge == -1, axis=1))
        self.intedges = np.setxor1d(np.arange(self.nedges), self.bdryedges)
        self.nbdryedges = len(self.bdryedges)
        self.nintedges = len(self.intedges)
        self.bdrylabels = None
        self.constructBoundaryEdges(bdrylabelsmsh, cells['line'])
        print(self)
    def __str__(self):
        return "TriangleMesh({}): nvert/ncells/nedges: {}/{}/{} bdrylabels={}".format(self.geomname, len(self.x), len(self.triangles), len(self.edges), list(self.bdrylabels.keys()))
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
        # print('celldata', celldata)
        drange = celldata.GetRange()
        # print('drange', drange)

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

# ------------------------------------- #

if __name__ == '__main__':
    tmesh = TriangleMesh(geomname="backwardfacingstep", hmean=0.7)
    tmesh.plot(plt, boundaryLabels=True)
