import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pygmsh
from simfempy.meshes import plotmesh 
from simfempy.applications.stokes import Stokes
from simfempy.applications.problemdata import ProblemData
from simfempy.meshes.simplexmesh import SimplexMesh 

#===============================================
def main(h):
    meshWB, data = channelWithBump(h)
    print(f"{meshWB=}")
    model = Stokes(mesh=meshWB, problemdata=data)
    resultWB = model.solve()
    meshWN, data = channelWithNavier(h)
    print(f"{meshWN=}")
    model = Stokes(mesh=meshWN, problemdata=data)
    resultWN = model.solve()
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
    plotmesh.meshWithData(meshWB, data=resultWB.data, title="WithBump", fig=fig, outer=outer[0,0])
    plotmesh.meshWithData(meshWB, title="WithBump", fig=fig, outer=outer[0,1],quiver_data={"V":list(resultWB.data['point'].values())})
    plotmesh.meshWithData(meshWN, data=resultWN.data, title="WithNavier", fig=fig, outer=outer[1,0])
    plotmesh.meshWithData(meshWN, title="WithNavier", fig=fig, outer=outer[1,1],quiver_data={"V":list(resultWN.data['point'].values())}) 
    # plotmesh.meshWithBoundaries(mesh)
    plt.show()
    plt.savefig("solution.png")

    nodes = np.unique(meshWB.linesoflabel[99999])
    vWB = resultWB.data['point']['V_0'][nodes]
    xWB = meshWB.points[nodes,0]
    iWB = np.argsort(xWB)
    nodes = np.unique(meshWN.linesoflabel[2002])
    vWN = resultWN.data['point']['V_0'][nodes]
    xWN = meshWN.points[nodes,0]
    iWN = np.argsort(xWN)
    plt.plot(xWB[iWB], vWB[iWB], color = 'blue', label="with bump")
    plt.plot(xWN[iWN], vWN[iWN], color = 'green', label="with navier")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("compare_vt.png")

#===============================================
def channelWithNavier(h= 0.1, mu=0.1):
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
    #---------------------------------------------------------------------------
    data = ProblemData()
   # boundary conditions  
    data.bdrycond.set("Dirichlet", [2000,2005])
    data.bdrycond.set("Neumann", [2003])  
    data.bdrycond.set("Navier", [2002])  
    #dirichlet
    data.bdrycond.fct[2005] = [lambda x, y, z:  1,  lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    data.params.scal_glob["navier"] = 0.2
    data.ncomp = 2
    return SimplexMesh(mesh=mesh), data

#===============================================
def channelWithBump(h= 0.1, mu=0.1):
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
    #---------------------------------------------------------------------------
    data = ProblemData()
   # boundary conditions  
    data.bdrycond.set("Dirichlet", [2000,2005])
    data.bdrycond.set("Neumann", [2003])  
    #dirichlet
    data.bdrycond.fct[2005] = [lambda x, y, z:  1,  lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    data.ncomp = 2
    return SimplexMesh(mesh=mesh), data

#===============================================
def poiseuille_function(h= 0.1, mu=0.1):
#We define "x" as the length of the cavity
    #"y" is a function which modelize a side of the domain $\Omega$. $\Omega$ correspond to a polygone here
    nx = 500
    x = np.linspace(-5,5, nx)
    y = np.ones_like(x)
    #fonction rectangle
    y2 = (x[200:300])*0 +10**(-6) +3
    with pygmsh.geo.Geometry() as geom:
        #vertex is an array containing the coordinates -in 3D- of the vertices of the polygone.
        #Here, we have almost them with "y". But, if "y" is equal to a constant for instance, we have to close the figure!
        #That's why we have to add one (or two as we did below) points.       
        vertex = np.zeros((nx+2, 3))
        vertex[:nx, 0] = x
        vertex[:nx, 1] = y
        vertex[200:300,1] = y2
        #To close the polygon: (Take care to the order of the nodes!)
        vertex[-2,0] = 5
        vertex[-2,1] = -2
        vertex[-1,0] = -5
        vertex[-1,1] = -2
        #to slim the meshing.
        ms = 0.01*np.ones(vertex.shape[0])
        ms[::2] = 1
        # create the polygon
        p = geom.add_polygon(vertex, mesh_size = h )
        # ------
        # p1 = geom.add_point([-1.5, 1, 0], mesh_size = 0.2*h)
        # p2 = geom.add_point([ 1.5, 1, 0], mesh_size = 0.2*h)
        # l6 = geom.add_line(p1, p2)
        # # l6 = geom.add_line(p.points[200], p.points[299])
        # geom.in_surface(l6, p.surface)
        # geom.add_physical(l6, label="99999")
        # ------

        geom.add_physical(p.surface, label="100")
        #We name the different sides of the polygone
        #Lines between the points given by the function "y"
        geom.add_physical(p.lines[0:200], label="2000")
        geom.add_physical(p.lines[200:300], label="2001")                                       
        geom.add_physical(p.lines[300:-3], label="2002")
        #The other sides
        geom.add_physical(p.lines[-3], "2003")
        geom.add_physical(p.lines[-2], "2004")
        geom.add_physical(p.lines[-1], "2005")

        #Virtual Segment
        # p1 = geom.add_point([vertex[200, 0], vertex[200, 1], vertex[200, 2]], mesh_size = 0.2*h)
        # p2 = geom.add_point([vertex[299, 0], vertex[299, 1], vertex[299, 2]], mesh_size = 0.2*h)
        # l6 = geom.add_line(p.points[200], p.points[299])
        # geom.in_surface(l6, p.surface)
        # geom.add_physical(l6, label="99999")
        mesh = geom.generate_mesh()
    #---------------------------------------------------------------------------
    data = ProblemData()
   # boundary conditions  
    data.bdrycond.set("Dirichlet", [2000,2001,2002,2004,2005])
    data.bdrycond.set("Neumann", [2003])  
    #dirichlet
    data.bdrycond.fct[2005] = [lambda x, y, z:  1,  lambda x, y, z: 0]
    # parameters
    data.params.scal_glob["mu"] = mu
    data.params.scal_glob["navier"] = 10**(-6)
    data.ncomp = 2
    return SimplexMesh(mesh=mesh), data

#================================================================#
if __name__ == '__main__':
    main(h=0.2)

