from mesh import Mesh
# from refinement import *
import refinement
import numpy as np
import matplotlib.pyplot as plt


geometry = 'Lshape'
methods = ['NVB1', 'NVB3', 'NVB5', 'NVBEdge', 'RGB']

for method in methods:
    # load and plot initial mesh
    mesh = Mesh.loadFromFolder(geometry)
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(f'{geometry} - {method}')
    ax0.triplot(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)
    ax0.set_title('Initial mesh')
    ax0.set(adjustable='box', aspect='equal')

    # refine first element (or edge) of mesh
    markedElements = np.array([4,6,11])
    refinementRoutine = refinement.getRefinementRoutine(mesh, method)
    refineData = refinementRoutine.prepareRefinementData(markedElements)
    newNodes, newElements = refinementRoutine.refineMesh(refineData)
    mesh = Mesh(newNodes, newElements)

    # plot final mesh
    ax1.triplot(mesh.nodes[:,0], mesh.nodes[:,1], mesh.elements)
    ax1.set_title('Refined mesh')
    ax1.set(adjustable='box', aspect='equal')
    plt.show()
