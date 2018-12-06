# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    fempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(fempypath)
from raviartthomas import RaviartThomas
import fempy.tools.analyticalsolution

class Laplace(RaviartThomas):
    """
    Fct de base de RT0 = sigma * 0.5 * |S|/|K| (x-x_N)
    """
    def __init__(self, problem):
        RaviartThomas.__init__(self)
        self.dirichlet = None
        self.rhs = None
        self.solexact = None
        self.problem = problem
        self.defineProblem(problem=problem)

    def defineProblem(self, problem):
        problemsplit = problem.split('_')
        if problemsplit[0] == 'Analytic':
            if problemsplit[1] == 'Linear':
                solexact = analyticalsolution.AnalyticalSolution('0.3 * x + 0.7 * y')
            elif problemsplit[1] == 'Quadratic':
                solexact = analyticalsolution.AnalyticalSolution('x*x+2*y*y')
            elif problemsplit[1] == 'Hubbel':
                solexact = analyticalsolution.AnalyticalSolution('(1-x*x)*(1-y*y)')
            elif problemsplit[1] == 'Exponential':
                solexact = analyticalsolution.AnalyticalSolution('exp(x-0.7*y)')
            elif problemsplit[1] == 'Sinus':
                solexact = fempy.tools.analyticalsolution.AnalyticalSolution('sin(x+0.2*y*y)')
            else:
                raise ValueError("unknown analytic solution: {}".format(problemsplit[1]))
            self.dirichlet = solexact
            self.solexact = solexact
        else:
            raise ValueError("unownd problem {}".format(problem))

    def solve(self, iter, dirname):
        return self.solveLinear()

    def postProcess(self, u):
        nedges =  self.mesh.nedges
        info = {}
        cell_data = {'p': u[nedges:]}
        # v0,v1 = self.computeVCells(u[:nedges])
        v0,v1 = self.computeVEdges(u[:nedges])
        cell_data['v0'] = v0
        cell_data['v1'] = v1
        point_data = {}
        if self.solexact:
            err, pe, vex, vey = self.computeError(self.solexact, u, (v0, v1))
            cell_data['perr'] =np.abs(pe - u[nedges:])
            cell_data['verrx'] =np.abs(vex + v0)
            cell_data['verry'] =np.abs(vey + v1)
            info['error'] = err
        info['timer'] = self.timer
        info['runinfo'] = self.runinfo
        return point_data, cell_data, info

    def computeError(self, solexact, u, vcell):
        nedges =  self.mesh.nedges
        ncells =  self.mesh.ncells
        errors = {}
        p = u[nedges:]
        vx, vy = vcell
        xc, yc = self.mesh.centersx, self.mesh.centersy
        pex = solexact(xc, yc)
        vexx = solexact.x(xc, yc)
        vexy = solexact.y(xc, yc)
        errp = np.sqrt(np.sum((pex-p)**2* self.mesh.area))
        errv = np.sqrt(np.sum( (vexx+vx)**2* self.mesh.area +  (vexy+vy)**2* self.mesh.area ))
        errors['pL2'] = errp
        errors['vcL2'] = errv
        return errors, pex, vexx, vexy

    def computeRhs(self):
        dirichlet, rhs = self.dirichlet, self.rhs
        nedges =  self.mesh.nedges
        bdryedges =  self.mesh.bdryedges
        bcells = None
        solexact = self.solexact
        xmid, ymid =  self.mesh.centersx,  self.mesh.centersy
        if solexact:
            assert rhs is None
            bcells = -(solexact.xx(xmid, ymid) + solexact.yy(xmid, ymid))* self.mesh.area[:]
        elif rhs:
            assert solexact is None
            bcells = rhs(xmid, ymid) *  self.mesh.area[:]
        bsides = np.zeros(nedges)
        for (count, ie) in enumerate(bdryedges):
            sigma = 1.0
            if  self.mesh.cellsOfEdge[ie, 0]==-1:
                sigma = -1.0
            ud = dirichlet(self.xedge[ie], self.yedge[ie])
            bsides[ie] = -linalg.norm( self.mesh.normals[ie])*sigma*ud
        return np.concatenate((bsides, bcells))

    def matrix(self):
        AI = self.matrixInterior()
        return AI.tocsr()

    def matrixInterior(self):
        """
        on suppose que  self.mesh.edgesOfCell[ic, kk] et oppose à elem[ic,kk] !!!
        """

        ncells =  self.mesh.ncells
        nedges =  self.mesh.nedges
        elem =  self.mesh.triangles

        # test indices
        for ic in range(ncells):
            tric = set(list(elem[ic]))
            for kk in range(3):
                ie =  self.mesh.edgesOfCell[ic, kk]
                trie = set( self.mesh.edges[ie])
                if trie.union([elem[ic,kk]]) != tric:
                    raise ValueError("problem: tric=%s trie=%s" %(str(tric), str(trie)))


        nlocal = 15
        index = np.zeros(nlocal*ncells, dtype=int)
        jndex = np.zeros(nlocal*ncells, dtype=int)
        A = np.zeros(nlocal*ncells, dtype=np.float64)
        edges = np.zeros(3, dtype=int)
        sigma = np.zeros(3, dtype=np.float64)
        scale = np.zeros(3, dtype=np.float64)
        rt = np.zeros((3,2), dtype=np.float64)
        count = 0
        for ic in range(ncells):
            # v-psi
            edges =  self.mesh.edgesOfCell[ic]
            for ii in range(3):
                iei = edges[ii]
                sigma[ii] = 2.0*( self.mesh.cellsOfEdge[iei,0]==ic)-1.0
                scale[ii] = 0.5*linalg.norm( self.mesh.normals[iei])/ self.mesh.area[ic]
            for ii in range(3):
                for jj in range(3):
                    index[count+3*ii+jj] = edges[ii]
                    jndex[count+3*ii+jj] = edges[jj]
            for kk in range(3):
                iek =  self.mesh.edgesOfCell[ic, kk]
                xk = self.xedge[iek]
                yk = self.yedge[iek]
                for ii in range(3):
                    rt[ii,0] = sigma[ii] * scale[ii] * (xk -  self.mesh.x[elem[ic, ii]])
                    rt[ii,1] = sigma[ii] * scale[ii] * (yk -  self.mesh.y[elem[ic, ii]])
                    # print 'test rt', kk, ii, np.dot(rt[ii],  self.mesh.normals[iek])/linalg.norm( self.mesh.normals[iek])
                # for ll in range(3):
                #     iel =  self.mesh.edgesOfCell[ic, ll]
                #     print 'test rt', kk, ll, np.dot(rt[ii], self.mesh.normals[iel])

                for ii in range(3):
                    for jj in range(3):
                        A[count+3*ii+jj] +=  self.mesh.area[ic]* np.dot(rt[ii], rt[jj])/3.0
            # p-psi v-chi
            index[count + 9] = ic + nedges
            index[count + 10] = ic + nedges
            index[count + 11] = ic + nedges
            jndex[count + 12] = ic + nedges
            jndex[count + 13] = ic + nedges
            jndex[count + 14] = ic + nedges
            for ii in range(3):
                ie = edges[ii]
                jndex[count+9+ii] = ie
                index[count+12+ii] = ie
                adiv = linalg.norm( self.mesh.normals[ie])* sigma[ii]
                A[count+9+ii] = adiv
                A[count+12+ii] = -adiv
            count += nlocal
        return scipy.sparse.coo_matrix((A, (index, jndex)), shape=(nedges+ncells, nedges+ncells))

# ------------------------------------- #

if __name__ == '__main__':
    import fempy.tools.comparerrors
    import sys
    for module in sys.modules.keys():
        if 'fempy' in module:
            print(module)
    problem = 'Analytic_Quadratic'
    problem = 'Analytic_Sinus'

    methods = {}
    methods['poisson'] = Laplace(problem=problem)

    comp = fempy.tools.comparerrors.CompareErrors(methods, plot=False)
    comp.compare(h=[1.0, 0.5, 0.25, 0.125, 0.062])
