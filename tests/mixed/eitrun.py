# assert __name__ == '__main__'
from os import sys, path
# simfempypath = path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))),'simfempy')
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(simfempypath)

print("sys.path",sys.path)
import simfempy.applications
import numpy as np
import matplotlib.pyplot as plt

import eitdef, eitlin, eitexp

#----------------------------------------------------------------#
def test(nholes=2, percrandom = 0., plot=True):
    h = 0.5
    nmeasures = 32
    diffglobalinv = 1
    mesh, kwargs = eitdef.problemdef(h, nholes, nmeasures, volt=4)
    kwargs['diffglobalinv'] = diffglobalinv
    parammethod = "lin"
    if parammethod == "lin":
        eit = eitlin.EIT(**kwargs)
    elif parammethod == "exp":
        eit = eitexp.EIT(**kwargs)
    else:
        raise ValueError("unknown parammethod '{}'".format(parammethod))
    eit.setMesh(mesh)

    regularize = 0.000
    diffinv0 = diffglobalinv*np.ones(nholes)
    optimizer = simfempy.solvers.optimize.Optimizer(eit, nparam=nholes, nmeasure=nmeasures, regularize=regularize,
                                                    param0=eit.diffinv2param(diffinv0))
    refdiffinv = diffglobalinv*np.ones(nholes)
    if nholes==36:
        refdiffinv[7] /= 10
        refdiffinv[11] /= 10
        refdiffinv[17] /= 10
    elif nholes == 4:
        refdiffinv[0] /= 10
        refdiffinv[3] /= 5
    else:
        # refdiffinv[::2] /= 5
        # refdiffinv[1::2] /= 10
        refdiffinv[::2] /= 5
        refdiffinv[1::2] /= 10
    print("refdiffinv",refdiffinv)

    refdata, perturbeddata = optimizer.create_data(refparam=eit.diffinv2param(refdiffinv), percrandom=percrandom)
    if plot: eit.plotter.plot(info=eit.info)
    
    # perturbeddata[::2] *= 1.3
    # perturbeddata[1::2] *= 0.7
    # print("refdata",refdata)
    # print("perturbeddata",perturbeddata)

    initialdiffinv = diffglobalinv*np.ones(nholes)
    print("initialdiffinv",initialdiffinv)

    bounds = False
    if bounds:
        bounds = (eit.diffinv2param(0.01*diffglobalinv), eit.diffinv2param(diffglobalinv))
        methods = optimizer.boundmethods
        methods = ['trf','dogbox']
    else:
        bounds = None
        methods = optimizer.methods

    # optimizer.hestest = True
    methods = optimizer.lsmethods.copy()
    methods.append("trust-ncg")
    methods.append("L-BFGS-B")
    values, valformat = optimizer.testmethods(x0=eit.diffinv2param(initialdiffinv), methods=methods, bounds=bounds, plot=False)
    # eit.plotter.plot(info=eit.info)

    latex = simfempy.tools.latexwriter.LatexWriter(filename="mincompare_{}".format(nholes))
    latex.append(n=methods, nname='method', nformat="20s", values=values, valformat=valformat)
    latex.write()
    latex.compile()
    return methods, values, valformat


#----------------------------------------------------------------#
def plotJhat():
    h = 0.4
    nmeasures = 8
    nholes = 2
    diffglobalinv = 1
    # eit = problemdef(h, nholes, nmeasures, diffglobalinv)
    mesh, kwargs = eitdef.problemdef(h, nholes, nmeasures, volt=4)
    kwargs['diffglobalinv'] = diffglobalinv
    eit = eitlin.EIT(**kwargs)
    eit.setMesh(mesh)

    optimizer = simfempy.solvers.optimize.Optimizer(eit, nparam=nholes, nmeasure=nmeasures)
    refdiffinv = diffglobalinv*np.ones(nholes, dtype=float)
    refdiffinv[::2] /= 5
    refdiffinv[1::2] /= 10
    percrandom = 0.1
    refdata, perturbeddata = optimizer.create_data(refparam=refdiffinv, percrandom=percrandom)
    eit.plotter.plot(info=eit.info)

    n = 25
    c = np.empty(shape=(n,n,nmeasures))
    px = np.linspace(0.1*refdiffinv[0], 20*refdiffinv[0], n)
    py = np.linspace(0.1*refdiffinv[1], 20*refdiffinv[1], n)
    param = np.empty(2, dtype=float)
    for i in range(n):
        print("")
        for j in range(n):
            print(end="$")
            param[0] = px[i]
            param[1] = py[j]
            data, u = eit.computeRes(param)
            # print("data", data)
            # print("param", param, "data",data)
            c[i,j] = data
    xx, yy = np.meshgrid(px, py)
    ncols = min(nmeasures,3)
    nrows = nmeasures//3 + bool(nmeasures%3)
    ncols = 1
    nrows = 3
    # print("nrows, ncols", nrows, ncols)
    fig, axs = plt.subplots(ncols, nrows, figsize=(nrows*4.5,ncols*4), squeeze=False)
    fig.suptitle("rand = {}%".format(percrandom))
    # aspect = (np.max(x)-np.mean(x))/(np.max(y)-np.mean(y))
    ind = [0,7]
    for i in range(nrows-1):
        # ax = axs[i // ncols, i % ncols]
        ax = axs[0,i]
        cnt = ax.contourf(xx, yy, np.abs(c[:,:,ind[i]]), 16, cmap='jet')
        # ax.set_aspect(1)
        # clb = plt.colorbar(cnt, ax=ax)
        ax.set_title(r"$|c_{}(u)-c_0|$".format(ind[i]))
    Jhat = np.sum(c*c, axis=(2))
    Jhat /= np.max(Jhat)
    # print("Jhat", Jhat)
    ax = axs[0,-1]
    CS = ax.contour(px, py, Jhat, levels=np.linspace(0.,0.8,20))
    ax.clabel(CS, inline=1, fontsize=8)
    ax.set_title(r'$\hat J$')
    plt.show()


#================================================================#

nholess = [2, 4, 9, 16, 25]
valuesall = []
for nholes in nholess:
    methods, values, valformat = test(nholes, plot=False)
    valuesall.append(values['nf'])
valuesall = np.array(valuesall)
print("valuesall", valuesall)
for i,m in enumerate(methods):
    plt.plot(nholess, valuesall[:,i], 'X-', label=m)
plt.legend()
plt.show()

# plotJhat()
