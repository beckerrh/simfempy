import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#----------------------------------------------------------------#
def _plotVerticesAndCells(x, y, triangles, centersx, centersy, ax=plt, plotlocalNumbering=True):
    props = dict(boxstyle='round', facecolor='wheat')
    for iv in range(len(x)):
        ax.text(x[iv], y[iv], r'%d' % (iv), fontweight='bold', bbox=props)
    for it in range(len(triangles)):
        ax.text(centersx[it], centersy[it], r'%d' % (it), color='r', fontweight='bold')
        if plotlocalNumbering:
            for ii in range(3):
                iv = triangles[it, ii]
                ax.text(0.75 * x[iv] + 0.25 * centersx[it], 0.75 * y[iv] + 0.25 * centersy[it],
                        r'%d' % (ii), color='g', fontweight='bold')


#=================================================================#
def meshWithBoundaries(meshdata, ax=plt):
    try:
        points, cells, point_data, cell_data, field_data = meshdata
        x, y = points[:,0], points[:,1]
        tris = cells['triangle']
        lines = cells['line']
        bdrylabels = cell_data['line']['gmsh:physical']
    except:
        x, y, tris, lines, bdrylabels = meshdata
    assert len(bdrylabels) == lines.shape[0]
    colors = np.unique(bdrylabels)
    print("colors", colors)
    ax.triplot(x, y, tris, color='k')
    pltcolors = 'bgrcmyk'
    patches=[]
    for i,color in enumerate(colors):
        ind = (bdrylabels == color)
        linescolor = lines[ind]
        patches.append(mpatches.Patch(color=pltcolors[i], label=color))
        for line in linescolor:
            ax.plot(x[line],y[line], color=pltcolors[i], lw=4)
    ax.legend(handles=patches)
    ax.set_title("Mesh and Boundary Labels")

#=================================================================#
def meshWithNodesAndTriangles(plotdata, ax=plt):
    x, y, tris, cx, cy = plotdata
    plt.triplot(x, y, tris, color='k', lw=1)
    _plotVerticesAndCells(x, y, tris, cx, cy, ax=ax)
    try:
        ax.set_title("Nodes and Triangles")
    except:
        ax.title("Nodes and Triangles")