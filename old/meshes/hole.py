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
        lines = hole.curve_loop.curves
    else:
        z=0
        hcoord = [[xc-r, yc-r], [xc-r, yc+r], [xc+r, yc+r], [xc+r, yc-r]]
        xhole = np.insert(np.array(hcoord), 2, z, axis=1)
        hole = geom.add_polygon(points=xhole, mesh_size=mesh_size, make_surface=make_surface)
        lines = hole.lines
    if make_surface:
        geom.add_physical(hole.surface, label=str(label))
        labels = [f"{10 * int(label) + i}" for i in range(len(lines))]
    else:
        labels = [f"{int(label) + i}" for i in range(len(lines))]
    # print(f"{labels=}")
    for lab,l in zip(labels, lines):
        # print(f"{lab=} {l=}")
        geom.add_physical(l, label=lab)
    return hole