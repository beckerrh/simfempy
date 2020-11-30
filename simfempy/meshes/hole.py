import numpy as np

def hole(geom, xc, yc, r, mesh_size, label, make_surface=False, circle=False):
    """
    :param xc,yc,r: position and size of hole
    :param label:
    :param make_surface:
    :param lcar:
    :return: hole
    """
    # add z-component
    if circle:
        hole = geom.add_circle(x0=[xc,yc], radius=r, mesh_size=mesh_size, make_surface=make_surface)
        if make_surface:
            geom.add_physical(hole.surface, label=str(label))
            for j in range(len(hole.lines)): geom.add_physical(hole.lines[j], label=f"{10 * int(label) + j}")
        else:
            for i,c in enumerate(hole.curve_loop.curves):
                geom.add_physical(c, label=f"{int(label) + i}")
    else:
        z=0
        hcoord = [[xc-r, yc-r], [xc-r, yc+r], [xc+r, yc+r], [xc+r, yc-r]]
        xhole = np.insert(np.array(hcoord), 2, z, axis=1)
        hole = geom.add_polygon(points=xhole, mesh_size=mesh_size, make_surface=make_surface)
        if make_surface:
            geom.add_physical(hole.surface, label=str(label))
            for j in range(len(hole.lines)): geom.add_physical(hole.lines[j], label=f"{10*int(label)+j}")
        else:
            for j in range(len(hole.lines)): geom.add_physical(hole.lines[j], label=f"{int(label)+j}")
    return hole
