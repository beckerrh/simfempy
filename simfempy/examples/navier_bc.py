assert __name__ == '__main__'
# in shell
from operator import ge
import os, sys
simfempypath = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir,'simfempy'))
sys.path.insert(0,simfempypath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pygmsh
from simfempy.meshes import plotmesh 
from simfempy.applications.stokes import Stokes
from simfempy.applications.navierstokes import NavierStokes
from simfempy.applications.problemdata import ProblemData
from simfempy.meshes.simplexmesh import SimplexMesh
from scipy.interpolate import interp1d 
from scipy.optimize import least_squares

#===============================================
def plotcurve(mesh, result, linecolor, label):
    nodes = np.unique(mesh.linesoflabel[linecolor])
    vt = result.data['point']['V_0'][nodes]
    x = mesh.points[nodes,0]
    i = np.argsort(x)
    plt.plot(x[i], vt[i], label=label)


#===============================================
def getvt(test, linecolor, h=0.1):
    mesh = test[0](h)
    data = problemData(navier=test[1])
    model = Stokes(mesh=mesh, problemdata=data)
    result = model.solve()
    lines = mesh.linesoflabel[linecolor]
    nodes = np.unique(mesh.linesoflabel[linecolor])
    # print(f"{mesh.points[lines].shape}")
    return result.data['point']['V_0'][nodes], mesh.points[nodes,0]
    
#===============================================
def getres(lam, f0, h=0.1):
    if isinstance(lam, np.ndarray):
        assert lam.shape == (1,)
        lam = lam[0]
    # print("lam ", lam)
    test = channelWithNavier, lam
    vt, x = getvt(test, linecolor=2002, h=h)
    f = interp1d(x, vt)
    r = f(x)-f0(x)
    return np.linalg.norm(r)

def plot(h=0.1):
    test = channelWithOscillation, None
    vt0, x0 = getvt(test, linecolor=99999, h=h)
    f0 = interp1d(x0, vt0)
    lams = np.linspace(0.1, 2, 41)
    ress = np.empty_like(lams)
    for i,lam in enumerate(lams):
        ress[i] = getres(lam, f0, h=h)
    i = np.argmin(ress)
    print(f"{lams[i]=} {ress[i]=}")
    plt.plot(lams, ress)
    plt.show()

def opt(h=0.1):
    test = channelWithOscillation, None
    vt0, x0 = getvt(test, linecolor=99999, h=h)
    f0 = interp1d(x0, vt0)
    res = least_squares(getres, x0=10, kwargs={'f0':f0, 'h':h}, bounds=(0,np.inf))
    print(f"{res=}")


#===============================================
def main(h):
    tests = {"WithNavier": (channelWithNavier, 1.0), "WithOscillation": (channelWithOscillation, None)}
    tests = {"WithNavier": (channelWithNavier, 0.1)}
    results, meshs = {}, {}
    for t in tests:
        meshs[t] = tests[t][0](h)
        data = problemData(mu=0.01, navier=tests[t][1])
        model = Stokes(mesh=meshs[t], problemdata=data)
        # model = NavierStokes(mesh=meshs[t], problemdata=data, linearsolver='scipy_gcrotmk@1@100@full', precond_v='pyamg@aggregation@none@schwarz' )
        # model = NavierStokes(mesh=meshs[t], problemdata=data, linearsolver='spsolve')
        results[t] = model.solve()
        filename = f"{tests[t][0].__name__}_{tests[t][1]}"+'.vtu'
        meshs[t].write(filename, data=results[t].data)
    fig = plt.figure(figsize=(10, 8))
    ntests = len(tests)
    gs = gridspec.GridSpec(ntests, 3, wspace=0.2, hspace=0.2)
    for i,t in enumerate(tests):
        plotmesh.meshWithData(meshs[t], data=results[t].data, title=t, fig=fig, outer=gs[i,0])
        plotmesh.meshWithData(meshs[t], title=t, fig=fig, outer=gs[i,1],quiver_data={"V":list(results[t].data['point'].values())}) 
    ax = fig.add_subplot(gs[ntests//2, 2])
    plt.sca(ax)
    for i,t in enumerate(tests):
        linecolor = 99999 if tests[t][1]==None else 2002
        plotcurve(mesh=meshs[t], result=results[t], linecolor=linecolor, label=t)
    plt.legend()
    plt.grid()
    plt.show()
 
#===============================================
def problemData(mu=0.1, navier=None):
    data = ProblemData()
   # boundary conditions  
    data.bdrycond.set("Dirichlet", [2000,2005])
    data.bdrycond.set("Neumann", [2003])  
    if navier is not None:  data.bdrycond.set("Navier", [2002])  
    #dirichlet
    data.bdrycond.fct[2005] = [lambda x, y, z:  (1-y)*(2+y),  lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    if navier is not None: data.params.scal_glob["navier"] = navier
    data.ncomp = 2
    return data
#===============================================
def channelWithNavier(h= 0.2):
    x = [-5, 5, 5, 1.5, -1.5, -5]
    y = [-2, -2, 1, 1, 1, 1]
    ms = np.full_like(x, h)
    ms[3] = ms[4] = 0.1*h    
    X = np.vstack([x,y,np.zeros(len(x))]).T
    with pygmsh.geo.Geometry() as geom:
         # create the polygon
        p = geom.add_polygon(X, mesh_size = list(ms) )
        geom.add_physical(p.surface, label="100")
        dirlines = [p.lines[i] for i in range(0,len(p.lines),2)]
        geom.add_physical(dirlines, label="2000")
        geom.add_physical(p.lines[1], label="2003")                                       
        geom.add_physical(p.lines[3], label="2002")                                       
        geom.add_physical(p.lines[-1], label="2005")                                       
        mesh = geom.generate_mesh()
    return SimplexMesh(mesh=mesh)
#===============================================
def channelWithRectBump(h= 0.2):
    x = [-5, 5, 5, 1.5, 1.5, -1.5, -1.5, -5]
    y = [-2, -2, 1, 1, 3, 3, 1, 1]
    ms = np.full_like(x, h)
    ms[3] = ms[6] = 0.1*h    
    X = np.vstack([x,y,np.zeros(len(x))]).T
    with pygmsh.geo.Geometry() as geom:
         # create the polygon
        p = geom.add_polygon(X, mesh_size = list(ms) )
        #------------------------------------------------
        l6 = geom.add_line(p.points[3], p.points[6])
        geom.in_surface(l6, p.surface)
        geom.add_physical(l6, label="99999")
        #------------------------------------------------
        geom.add_physical(p.surface, label="100")
        dirlines = [p.lines[i] for i in range(len(p.lines)-1) if i != 1]
        geom.add_physical(dirlines, label="2000")
        geom.add_physical(p.lines[1], label="2003")                                       
        geom.add_physical(p.lines[-1], label="2005")                                       
        mesh = geom.generate_mesh()
    return SimplexMesh(mesh=mesh)
#===============================================
def channelWithTriBump(h= 0.2):
    x = [-5, 5, 5, 1.5, 0, -1.5, -5]
    y = [-2, -2, 1, 1, 4, 1, 1]
    ms = np.full_like(x, h)
    ms[3] = ms[5] = 0.1*h    
    X = np.vstack([x,y,np.zeros(len(x))]).T
    with pygmsh.geo.Geometry() as geom:
         # create the polygon
        p = geom.add_polygon(X, mesh_size = list(ms) )
        #------------------------------------------------
        l6 = geom.add_line(p.points[3], p.points[5])
        geom.in_surface(l6, p.surface)
        geom.add_physical(l6, label="99999")
        #------------------------------------------------
        geom.add_physical(p.surface, label="100")
        dirlines = [p.lines[i] for i in range(len(p.lines)-1) if i != 1]
        geom.add_physical(dirlines, label="2000")
        geom.add_physical(p.lines[1], label="2003")                                       
        geom.add_physical(p.lines[-1], label="2005")                                       
        mesh = geom.generate_mesh()
    return SimplexMesh(mesh=mesh)
#===============================================
def channelWithOscillation(h= 0.2):
    ncpoints = 301
    xc = np.linspace(1.5, -1.5, ncpoints)
    yc = 1 + 0.5*np.abs(np.cos(np.pi*xc))
    X = np.empty(shape=(ncpoints+4,3))
    X[:,2] = 0
    X[0,:2] = [-5, -2]
    X[1,:2] = [5, -2]
    X[2,:2] = [5, 1]
    X[3:3+ncpoints,0] = xc
    X[3:3+ncpoints,1] = yc
    X[-1,:2] = [-5, 1]
    ms = np.full_like(X[:,0], h)
    ms[[3,2+ncpoints]] *= 0.2
    with pygmsh.geo.Geometry() as geom:
        # create the polygon
        points, lines = [], []
        npoints = X.shape[0]
        for i in range(npoints):
            points.append(geom.add_point(X[i], mesh_size=ms[i]))
        # for i in range(npoints-1):
        #     lines.append(geom.add_line(points[i], points[i+1]))
        # lines.append(geom.add_line(points[-1], points[0]))
        lines.append(geom.add_line(points[0], points[1]))
        lines.append(geom.add_line(points[1], points[2]))
        lines.append(geom.add_line(points[2], points[3]))
        lines.append(geom.add_bspline(points[3:3+ncpoints]))
        lines.append(geom.add_line(points[-2], points[-1]))
        lines.append(geom.add_line(points[-1], points[0]))
        curve_loop = geom.add_curve_loop(lines)
        surface = geom.add_plane_surface(curve_loop)


        # p = geom.add_polygon(X, mesh_size = list(ms) )
        # points, lines, surface = p.points, p.lines, p.surface
        #------------------------------------------------
        # l6 = geom.add_line(geom.add_point(X[3],mesh_size=0.1), geom.add_point(X[-2]))
        l6 = geom.add_line(points[3], points[-2])
        geom.in_surface(l6, surface)
        geom.add_physical(l6, label="99999")
        # ------------------------------------------------
        geom.add_physical(surface, label="100")
        dirlines = [lines[i] for i in range(len(lines)-1) if i != 1]
        geom.add_physical(dirlines, label="2000")
        geom.add_physical(lines[1], label="2003")                                       
        geom.add_physical(lines[-1], label="2005")                                       
        mesh = geom.generate_mesh()
        # plotmesh.plotmesh(mesh)
        plt.show()
    return SimplexMesh(mesh=mesh)

#================================================================#
if __name__ == '__main__':
    main(h=0.2)
    # plot(h=0.2)
    # opt(h=0.2)

