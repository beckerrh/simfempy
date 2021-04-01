import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d

#----------------------------------------------------------------#
def _settitle(ax, text):
    try:
        ax.set_title(text)
    except:
        ax.title(text)

#----------------------------------------------------------------#
def _plotNodeLabels(x, y, z, ax=plt):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], r'%d' % (i), fontweight='bold', bbox=props)
#=================================================================#
def _plotCells(x, y, z, tets, ax=plt):
    for i in range(len(tets)):
        ax.plot(x[tets[i]], y[tets[i]], z[tets[i]], color=(0.5,0.5,0.5))
#=================================================================#
def _plotCellLabels(tets, xc, yc, zc, ax=plt):
    for i in range(len(tets)):
        ax.text(xc[i], yc[i], zc[i], r'%d' % (i), color='r', fontweight='bold', fontsize=10)
#=================================================================#
def meshWithBoundaries(x, y, z, tets, faces, bdrylabels, nodelabels=False, ax=plt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    ax.set_zlabel(r'z')
    # if nodelabels: _plotNodeLabels(x, y, z, ax=ax)
    _plotCells(x, y, z, tets, ax=ax)
    cmap = plt.get_cmap("tab10")
    patches=[]
    i=0
    for color, bdryfaces in bdrylabels.items():
        patches.append(mpatches.Patch(color=cmap(i), label=color))
        for ie in bdryfaces:
            poly3d = [ [x[f], y[f], z[f]] for f in faces[ie]]
            ax.add_collection3d(Poly3DCollection([poly3d], facecolors=cmap(i), linewidths=1))
        i += 1
    ax.legend(handles=patches)
    _settitle(ax, "Mesh and Boundary Labels")
#=================================================================#
def plotmesh(mesh, **kwargs):
    import pyvista
    import vtk
    if 'ax' in kwargs: ax = kwargs.pop('ax')
    else: ax = plt
    title = 'Mesh'
    if 'title' in kwargs: title = kwargs.pop('title')
    x, y, z, tets = mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2], mesh.simplices

    xyz = np.stack((x, y, z)).T
    ntets = tets.shape[0]
    cell_type = vtk.VTK_TETRA*np.ones(ntets, dtype=int)
    offset = 5*np.arange(ntets)
    cells = np.insert(tets, 0, 4, axis=1).ravel()
    grid = pyvista.UnstructuredGrid(offset, cells, cell_type, xyz)

    plotter = pyvista.Plotter()
    plotter.renderer.SetBackground(255, 255, 255)
    plotter.add_axes()
    plotter.add_mesh(grid, showedges=False, opacity=0.6, color='gray')
    plotter.remove_scalar_bar()
    pltcolors = 'bgrcmykbgrcmyk'
    for i, (color, vertices) in enumerate(mesh.verticesoflabel.items()):
        # print("color", color, "vertices", vertices)
        points = mesh.points[vertices]
        nv = vertices.shape[0]
        # plotter.add_points(points, scalars=i*np.ones(nv), cmap=None, flip_scalars=True)
        plotter.add_points(points, color=pltcolors[i], cmap=None, flip_scalars=True, point_size=15)
        # plotter.add_text('x', points[0], color=pltcolors[i])
        # patches.append(mpatches.Patch(color=pltcolors[i], label=color))
        # for vertex in vertices:
        #     ax.plot(x[vertex], y[vertex],'X', color=pltcolors[i])
    cpos = plotter.show(title="U")

    # ax.triplot(x, y, tris, color='k')
    # if ax ==plt:
    #     plt.gca().set_aspect(aspect='equal')
    #     ax.xlabel(r'x')
    #     ax.ylabel(r'y')
    # else:
    #     ax.set_aspect(aspect='equal')
    #     ax.set_xlabel(r'x')
    #     ax.set_ylabel(r'y')
    # celllabels = mesh.cell_labels
    # cnt = ax.tripcolor(x, y, tris, facecolors=celllabels, edgecolors='k', cmap='jet', alpha=0.4)
    # # clb = plt.colorbar(cnt)
    # # clb.set_label("cellcolors")
    # pltcolors = 'bgrcmykbgrcmyk'
    # patches=[]
    # for i, (color, vertices) in enumerate(mesh.verticesoflabel.items()):
    #     patches.append(mpatches.Patch(color=pltcolors[i], label=color))
    #     for vertex in vertices:
    #         ax.plot(x[vertex], y[vertex],'X', color=pltcolors[i])
    # ax.legend(handles=patches)
    # _settitle(ax, title)
# =================================================================#
def plotMeshWithCellData(ax, cdn, cd, x, y, z, tets, alpha):
    if tets.shape[0] != cd.shape[0]:
        raise ValueError("wrong length in '{}' {}!={}".format(cdn, tets.shape[0], cd.shape[0]))
    _plotCells(x, y, z, tets, ax)
    ax.plot(x, y, z, 'k.')
    # cnt = ax.plot_surface(x, y, z, tris, facecolors=cd, edgecolors='k', cmap='jet')
    # ax.set_aspect(aspect='equal')
    # clb = plt.colorbar(cnt, ax=ax)
    # # clb.ax.set_title(cdn)
    # clb.set_label(cdn)
    _settitle(ax, cdn)
# =================================================================#
def meshWithDataPlt(**kwargs):
    x, y, tris, xc, yc = kwargs['x'], kwargs['y'], kwargs['tris'], kwargs['xc'], kwargs['yc']
    nplots = len(point_data)+len(cell_data)
    if nplots==0:
        print("meshWithData() no point_data")
        return
    fig = plt.figure(figsize=plt.figaspect(1/nplots))
    if suptitle: fig.suptitle(suptitle)
    axs = []
    for i in range(nplots):
        axs.append(fig.add_subplot("1{:1d}{:1d}".format(nplots, i+1), projection='3d'))
    count=0
    for pdn, pd in point_data.items():
        ax = axs[count]
        xyz = np.stack((x,y,z)).T
        print("xyz", xyz.shape)
        vts = xyz[tets, :]
        tri = mplot3d.art3d.Poly3DCollection(vts)
        tri.set_alpha(0.2)
        tri.set_color('grey')
        ax.add_collection3d(tri)
        ax.plot(x,y,z, 'k.')
        ax.set_axis_off()
        # for i in range(len(tets)):
        #     ax.plot(x[tets[i]], y[tets[i]], z[tets[i]])
            # ax.plot_trisurf(x[tets[i]], y[tets[i]], z[tets[i]])
        _settitle(ax, pdn)
        count += 1
    if title: fig.canvas.set_window_title(title)
    plt.show()
# ================================================================#
def meshWithData(**kwargs):
    import pyvista
    import vtk

    x, y, z, tets = kwargs['x'], kwargs['y'], kwargs['z'], kwargs['tets']
    addplots = []
    if 'addplots' in kwargs: addplots = kwargs['addplots']
    if addplots is None: addplots=[]
    point_data, cell_data, quiver_cell_data, translate_point_data = None, None, None, None
    title, suptitle = None, None
    if 'data' in kwargs:
        point_data = kwargs['data']['point']
        cell_data = kwargs['data']['cell']
    # if 'point_data' in kwargs: point_data = kwargs['point_data']
    # if 'cell_data' in kwargs: cell_data = kwargs['cell_data']
    if 'quiver_cell_data' in kwargs: quiver_cell_data = kwargs['quiver_cell_data']
    if 'translate_point_data' in kwargs: translate_point_data = kwargs['translate_point_data']
    if 'numbering' in kwargs: numbering = kwargs['numbering']
    if 'title' in kwargs: title = kwargs['title']
    if 'suptitle' in kwargs: suptitle = kwargs['suptitle']

    xyz = np.stack((x, y, z)).T
    ntets = tets.shape[0]
    cell_type = vtk.VTK_TETRA*np.ones(ntets, dtype=int)
    cells = np.insert(tets, 0, 4, axis=1).ravel()
    grid = pyvista.UnstructuredGrid(cells, cell_type, xyz)
    # offset = 5*np.arange(ntets)
    # grid = pyvista.UnstructuredGrid(offset, cells, cell_type, xyz)

    if translate_point_data:
        assert len(point_data.keys()) ==3
        u, unames = [], []
        for k,v in point_data.items():
            unames.append(k)
            u.append(v)
        scale = translate_point_data
        un = np.sqrt(u[0]**2+u[1]**2+u[2]**2)
        xyz2 = np.stack((x+scale*u[0], y+scale*u[1], z+scale*u[2])).T
        grid2 = pyvista.UnstructuredGrid(offset, cells, cell_type, xyz2)
        plotter = pyvista.Plotter()
        plotter.renderer.SetBackground(255,255,255)
        plotter.add_axes()
        # plotter.add_mesh(grid, stitle="U=0", showedges=False, opacity=0.6, color='gray')
        plotter.add_mesh(grid, stitle="U=0", opacity=0.6, color='gray')
        plotter.remove_scalar_bar()
        # plotter.add_mesh(grid2, scalars=un, stitle="U", showedges=True, interpolatebeforemap=True)
        plotter.add_mesh(grid2, scalars=un, stitle="U")
        plotter.remove_scalar_bar()
        cpos = plotter.show(title="U")
    else:
        plotter = pyvista.Plotter()
        count=0
        for pdn, pd in point_data.items():
            # grid.plot(scalars=pd, stitle=pdn)
            # plotter.add_mesh(grid, scalars=pd, stitle=pdn,showedges=True,interpolatebeforemap=True)
            plotter.add_mesh(grid, scalars=pd, stitle=pdn)
            count += 1
        cpos = plotter.show()
    return
#=================================================================#
def meshWithData2(**kwargs):
    import pyvista
    import vtk

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xlabel('Hello from VTK!', size=16)
    ax.bar(xrange(10), p.rand(10))
    # The pyvistamageImporter will treat a python string as a void pointer
    importer = pyvistamageImport()
    importer.SetDataScalarTypeToUnsignedChar()
    importer.SetNumberOfScalarComponents(4)

    # It's upside-down when loaded, so add a flip filter
    imflip = pyvistamageFlip()
    imflip.SetInput(importer.GetOutput())
    imflip.SetFilteredAxis(1)

    # Map the plot as a texture on a cube
    cube = vtkCubeSource()

    cubeMapper = vtkPolyDataMapper()
    cubeMapper.SetInput(cube.GetOutput())

    cubeActor = vtkActor()
    cubeActor.SetMapper(cubeMapper)

    # Create a texture based off of the image
    cubeTexture = vtkTexture()
    cubeTexture.InterpolateOn()
    cubeTexture.SetInput(imflip.GetOutput())
    cubeActor.SetTexture(cubeTexture)

    ren = vtkRenderer()
    ren.AddActor(cubeActor)

    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    # Powers of 2 image to be clean
    w, h = 1024, 1024
    dpi = canvas.figure.get_dpi()
    fig.set_figsize_inches(w / dpi, h / dpi)
    canvas.draw()  # force a draw

    # This is where we tell the image importer about the mpl image
    extent = (0, w - 1, 0, h - 1, 0, 0)
    importer.SetWholeExtent(extent)
    importer.SetDataExtent(extent)
    importer.SetImportVoidPointer(canvas.buffer_rgba(0, 0), 1)
    importer.Update()

    iren.Initialize()
    iren.Start()

