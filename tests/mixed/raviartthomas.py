# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg

from fempy import solvers


class RaviartThomas(solvers.newtonsolver.NewtonSolver):
    """
    Fct de base de RT0 = sigma * 0.5 * |S|/|K| (x-x_N)
    """
    def __init__(self, **kwargs):
        solvers.newtonsolver.NewtonSolver.__init__(self, **kwargs)

    def setMesh(self, mesh):
        self.mesh = mesh
        self.pointsf = self.mesh.points[self.mesh.faces].mean(axis=1)
        from fempy import meshes
        # meshes.plotmesh.plotmesh(self.mesh, localnumbering=True)

    def solve(self):
        return self.solveLinear()

    def rt(self, ic, x):
        xn, yn, zn, dim = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.points[:, 2], self.mesh.dimension
        base = np.zeros((dim+1,dim), dtype=np.float64)
        for ii in range(dim+1):
            ie =  self.mesh.facesOfCells[ic, ii]
            iv =  self.mesh.simplices[ic, ii]
            scale = self.mesh.sigma[ic,ii] * linalg.norm( self.mesh.normals[ie]) /  self.mesh.dV[ic]/dim
            for jj in range(dim):
                base[ii, jj] = scale *  (x[jj] -  self.mesh.points[iv, jj])
            # base[ii] = scale * (x - self.mesh.points[iv])[::dim]
        return base

    def rteval(self, ic, x, y, vc):
        v = np.zeros(2, dtype=np.float64)
        dv = np.zeros( (2,2), dtype=np.float64)
        for ii in range(3):
            ie =  self.mesh.facesOfCells[ic, ii]
            iv =  self.mesh.triangles[ic, ii]
            sigma = self.mesh.sigma[ic, ii]
            scale = 0.5 * linalg.norm( self.mesh.normals[ie]) /  self.mesh.dV[ic]
            v[0] += sigma * scale *  (x -  self.mesh.x[iv]) * vc[ii]
            v[1] += sigma * scale *  (y -  self.mesh.y[iv]) * vc[ii]
            dv[0,0] += sigma * scale * vc[ii]
            dv[0,1] += 0.0
            dv[1,0] += 0.0
            dv[1,1] += sigma * scale * vc[ii]
        return (v, dv)


    def rtpgrad(self, ic, x, y):
        base = np.zeros((3,2), dtype=np.float64)
        basegrad = np.zeros((3), dtype=np.float64)
        for ii in range(3):
            ie =  self.mesh.facesOfCells[ic, ii]
            iv =  self.mesh.triangles[ic, ii]
            sigma = self.mesh.sigma[ic,ii]
            scale = 0.5 * linalg.norm( self.mesh.normals[ie]) /  self.mesh.dV[ic]
            base[ii, 0] = sigma * scale *  (x -  self.mesh.x[iv])
            base[ii, 1] = sigma * scale *  (y -  self.mesh.y[iv])
            basegrad[ii] = sigma * scale
        return (base, basegrad)

    def computeVCells(self, u):
        ncells, nfaces, nnodes, dim =  self.mesh.ncells, self.mesh.nfaces, self.mesh.nnodes, self.mesh.dimension
        x, y, z = self.mesh.points[:, 0], self.mesh.points[:, 1], self.mesh.points[:, 2]
        assert u.shape[0] == nfaces
        v = np.zeros((dim,ncells))
        for ic in range(ncells):
            rt = self.rt(ic,  self.mesh.pointsc[ic])
            v[:,ic] = np.dot( rt.T , u[ self.mesh.facesOfCells[ic]] )
        return v

    def computeVEdges(self, u):
        nfaces =  self.mesh.nfaces
        ncells =  self.mesh.ncells
        xf, yf, zf = self.pointsf[:, 0], self.pointsf[:, 1], self.pointsf[:, 2]
        assert u.shape[0] == nfaces
        vex = np.zeros(nfaces)
        vey = np.zeros(nfaces)
        for ie in range(nfaces):
            xe, ye = xf[ie], yf[ie]
            ic =  self.mesh.cellsOfFaces[ie]
            ic = ic[np.where(ic!=-1)]
            if len(ic)==1:
                rt0 = self.rt(ic[0], xe, ye)
                vex[ie] = np.dot(rt0[:, 0], u[ self.mesh.facesOfCells[ic[0], :]])
                vey[ie] = np.dot(rt0[:, 1], u[ self.mesh.facesOfCells[ic[0], :]])
            else:
                rt0 = self.rt(ic[0], xe, ye)
                vx0 = np.dot( rt0[:, 0] , u[ self.mesh.facesOfCells[ic[0], :]] )
                vy0 = np.dot( rt0[:, 1] , u[ self.mesh.facesOfCells[ic[0], :]])
                rt1 = self.rt(ic[1], xe, ye)
                vx1 = np.dot(rt1[:, 0], u[ self.mesh.facesOfCells[ic[1], :]])
                vy1 = np.dot(rt1[:, 1], u[ self.mesh.facesOfCells[ic[1], :]])
                vex[ie] = 0.5*(vx0+vx1)
                vey[ie] = 0.5 * (vy0 + vy1)
        vx = np.zeros(ncells)
        vy = np.zeros(ncells)
        for ic in range(ncells):
            for ii in range(3):
                ie =  self.mesh.facesOfCells[ic,ii]
                vx[ic] += vex[ie]/3.0
                vy[ic] += vey[ie] / 3.0
        return vx, vy

    def computeVNodes(self, u):
        nfaces =  self.mesh.nfaces
        nnodes =  self.mesh.nnodes
        x =  self.mesh.x
        y =  self.mesh.y
        assert u.shape[0] == nfaces
        v0 = np.zeros(nnodes)
        v1 = np.zeros(nnodes)
        for iv in range( self.mesh.nnodes):
            cv = np.array((x[iv],y[iv]))
            patches =  self.mesh.patches_bnodes[iv]
            patches2 = patches[np.where(patches[:,5]!=-1)[0]]
            npatch = patches2.shape[0]
            dist = np.zeros(npatch)
            for i,patch in enumerate(patches2):
                ie = patch[5]
                ce = np.array((self.xedge[ie],self.yedge[ie])) - cv
                dist[i] = linalg.norm(ce)
            dist /= np.sum(dist)
            for i,patch in enumerate(patches2):
                ie = patch[5]
                ud = u[ie]*dist[i]/linalg.norm( self.mesh.normals[ie])
                # ud = u[ie]/linalg.norm( self.mesh.normals[ie])/float(npatch)
                v0[iv] += ud *  self.mesh.normals[ie][0]
                v1[iv] += ud *  self.mesh.normals[ie][1]
        return v0,v1

# ------------------------------------- #

if __name__ == '__main__':
    print("so far no test")
