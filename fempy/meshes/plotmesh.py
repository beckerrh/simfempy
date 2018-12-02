import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#----------------------------------------------------------------#
def _settitle(ax, text):
    try:
        ax.set_title(text)
    except:
        ax.title(text)


#----------------------------------------------------------------#
def _plotVerticesAndCellsLabels(x, y, triangles, centersx, centersy, ax=plt, plotlocalNumbering=True):
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
        x, y, tris, lines, bdrylabels =  meshdata.x, meshdata.y, meshdata.triangles, meshdata.lines, meshdata.bdrylabelsmsh
    except:
        points, cells, point_data, cell_data, field_data = meshdata
        x, y = points[:,0], points[:,1]
        tris = cells['triangle']
        lines = cells['line']
        bdrylabels = cell_data['line']['gmsh:physical']
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
    _settitle(ax, "Mesh and Boundary Labels")

#=================================================================#
def meshWithNodesAndTriangles(meshdata, ax=plt):
    try:
        x, y, tris, cx, cy =  meshdata.x, meshdata.y, meshdata.triangles, meshdata.centersx, meshdata.centersy
    except:
        x, y, tris, cx, cy = meshdata
    plt.triplot(x, y, tris, color='k', lw=1)
    _plotVerticesAndCellsLabels(x, y, tris, cx, cy, ax=ax)
    _settitle(ax, "Nodes and Triangles")

#=================================================================#
def meshWithData(meshdata, point_data, cell_data, ax=plt):
    try:
        x, y, tris, cx, cy =  meshdata.x, meshdata.y, meshdata.triangles, meshdata.centersx, meshdata.centersy
    except:
        raise ValueError("cannot get data from meshdata")
    ax.triplot(x, y, tris, color='gray', lw=1)
    for pdn, pd in point_data.items():
        assert x.shape == pd.shape
        cs = ax.tricontourf(x, y, tris, pd)
        plt.colorbar()
        _settitle(ax, pdn)
