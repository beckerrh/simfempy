import pygmsh
import numpy as np
from simfempy.tools import npext
from simfempy.meshes import pygmshext


# ----------------------------------------------------------------#
def createMesh2d(**kwargs):
    geometry = pygmsh.built_in.Geometry()

    h = kwargs['h']
    hmeasure = kwargs.pop('hmeasure')
    nmeasures = kwargs.pop('nmeasures')
    measuresize = kwargs.pop('measuresize')
    x0, x1 = -1.4, 1.4

    hhole = kwargs.pop('hhole')
    nholes = kwargs.pop('nholes')
    nholesy = int(np.sqrt(nholes))
    nholesx = int(nholes/nholesy)
    print("hhole", hhole, "nholes", nholes, "nholesy", nholesy, "nholesx", nholesx)
    holes, hole_labels = pygmshext.add_holesnew(geometry, h=h, hhole=hhole, x0=x0, x1=x1, y0=x0, y1=x1, nholesx=nholesx, nholesy=nholesy)
    # un point additionnel pas espace entre segments-mesure
    num_sections = 3*nmeasures
    spacing = np.empty(num_sections)
    labels = np.empty(num_sections, dtype=int)
    spacesize = (1-nmeasures*measuresize)/nmeasures
    if spacesize < 0.1*measuresize:
        maxsize = 1/(nmeasures*1.1)
        raise ValueError("measuresize too big (max={})".format(maxsize))
    spacing[0] = 0
    spacing[1] = spacing[0] + measuresize
    spacing[2] = spacing[1] + 0.5*spacesize
    for i in range(1,nmeasures):
        spacing[3*i] = spacing[3*i-1] + 0.5*spacesize
        spacing[3*i+1] = spacing[3*i] + measuresize
        spacing[3*i+2] = spacing[3*i+1] + 0.5*spacesize
    labels[0] = 1000
    labels[1] = labels[0] + 1
    labels[2] = labels[1] + 0
    for i in range(1,nmeasures):
        labels[3*i] = labels[3*i-1] + 1
        labels[3*i+1] = labels[3*i] + 1
        labels[3*i+2] = labels[3*i+1] + 0
    labels = 1000*np.ones(num_sections, dtype=int)
    labels[::3] += np.arange(1,nmeasures+1)
    lcars = hmeasure*np.ones(num_sections+1)
    lcars[3::3] = h
    # print("lcars",lcars)
    # print("labels",labels)
    # labels[3::3] = labels[-1]

    circ = pygmshext.add_circle(geometry, 3 * [0], 2, lcars=lcars, h=h, num_sections=num_sections, holes=holes, spacing=spacing)

    vals, inds = npext.unique_all(labels)
    for val, ind in zip(vals, inds):
        geometry.add_physical([circ.line_loop.lines[i] for i in ind], label=int(val))

    geometry.add_physical(circ.plane_surface, label=100)
    # print("circ", dir(circ.line_loop))

    # with open("welcome.geo","w") as file: file.write(geometry.get_code())
    mesh = pygmsh.generate_mesh(geometry, verbose=False)
    mesh = OLD.simfempy.meshes.simplexmesh.SimplexMesh(mesh=mesh)
    measure_labels = labels[::3]
    other_labels = set.difference(set(np.unique(labels)),set(np.unique(measure_labels)))
    return mesh, hole_labels, measure_labels, other_labels


#----------------------------------------------------------------#
def problemdef(h, nholes, nmeasures, volt=4):
    h = h
    hhole, hmeasure = 0.2*h, 0.1*h
    measuresize = 0.02
    nholes = nholes
    mesh, hole_labels, electrode_labels, other_labels = createMesh2d(h=h, hhole=hhole, hmeasure=hmeasure, nholes=nholes, nmeasures=nmeasures, measuresize=measuresize)
    param_labels = hole_labels
    measure_labels = electrode_labels
    assert nmeasures == len(measure_labels)

    voltage = volt*np.ones(nmeasures)
    # voltage = volt*np.arange(nmeasures, dtype=float)
    step = max(2,nmeasures//min(nmeasures,4))
    voltage[::step] *= -1
    voltage -= np.mean(voltage)
    print("voltage", voltage)

    bdrycond = OLD.simfempy.applications.problemdata.BoundaryConditions()
    for label in other_labels:
        bdrycond.type[label] = "Neumann"
    for i,label in enumerate(electrode_labels):
        # bdrycond.type[label] = "Robin"
        # bdrycond.param[label] = 10000
        bdrycond.type[label] = "Dirichlet"
        # bdrycond.param[label] = 10000
        bdrycond.fct[label] = OLD.simfempy.solvers.optimize.RhsParam(voltage[i])
    postproc = {}
    postproc['measured'] = "bdrydn:{}".format(','.join( [str(l) for l in electrode_labels]))
    problemdata = OLD.simfempy.applications.problemdata.ProblemData(bdrycond=bdrycond, postproc=postproc)
    kwargs ={'problemdata':problemdata, 'measure_labels':measure_labels, 'param_labels':param_labels}
    return mesh, kwargs
    # eit = EIT(problemdata=problemdata, measure_labels=measure_labels, param_labels=param_labels, diffglobalinv=diffglobalinv)
    # eit.setMesh(mesh)
    return eit


