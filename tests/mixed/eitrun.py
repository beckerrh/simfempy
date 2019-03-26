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
    h = 0.8
    nmeasures = 4
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

    regularize = 0.00
    diffinv0 = diffglobalinv*np.ones(nholes)
    optimizer = simfempy.solvers.optimize.Optimizer(eit, nparam=nholes, nmeasure=nmeasures, regularize=regularize, param0=eit.diffinv2param(diffinv0))
    refdiffinv = diffglobalinv*np.ones(nholes)
    if nholes==25:
        refdiffinv[7] /= 100
        refdiffinv[11] /= 100
        refdiffinv[17] /= 100
    else:
        ri = np.random.randint(0,3,nholes)
        for i in range(nholes):
            if ri[i] == 1: refdiffinv[i] /= 100
            elif ri[i] == 2: refdiffinv[i] /= 50

    print("refdiffinv",refdiffinv)

    optimizer.create_data(refparam=eit.diffinv2param(refdiffinv), percrandom=percrandom, plot=plot)

    # perturbeddata[::2] *= 1.3
    # perturbeddata[1::2] *= 0.7
    # print("refdata",refdata)
    # print("perturbeddata",perturbeddata)

    initialdiffinv = diffglobalinv*np.ones(nholes)
    print("initialdiffinv",initialdiffinv)

    bounds = True
    if bounds:
        bounds = (eit.diffinv2param(0.01*diffglobalinv), eit.diffinv2param(diffglobalinv))
        print("bounds", bounds)
        methods = optimizer.boundmethods
        methods = ['trf','dogbox']
    else:
        bounds = None
        methods = optimizer.methods

    # optimizer.hestest = True
    # methods = optimizer.lsmethods.copy()
    # methods.append("trust-ncg")
    # methods.append("L-BFGS-B")
    methods = ['trf']
    # methods = optimizer.boundmethods.copy()
    # methods.append("lm")

    values, valformat, xall = optimizer.testmethods(x0=eit.diffinv2param(initialdiffinv), methods=methods, bounds=bounds, plot=plot, verbose=2)
    # eit.plotter.plot(info=eit.info)

    latex = simfempy.tools.latexwriter.LatexWriter(filename="mincompare_{}".format(nholes))
    latex.append(n=methods, nname='method', nformat="20s", values=values, valformat=valformat)
    latex.write()
    latex.compile()
    return methods, values, valformat, refdiffinv, np.array(xall, dtype=float)


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
    percrandom = 0.
    refdata, perturbeddata = optimizer.create_data(refparam=refdiffinv, percrandom=percrandom)
    eit.plotter.plot(info=eit.info)

    n = 30
    c = np.empty(shape=(n,n,nmeasures))
    px = np.linspace(0.1*refdiffinv[0], 10*refdiffinv[0], n)
    py = np.linspace(0.1*refdiffinv[1], 10*refdiffinv[1], n)
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
    fig.suptitle("rand = {}%".format(100*percrandom))
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
def testholes():
    nholess = [2, 4, 9]
    # nholess = [2, 4, 9, 16, 25]
    # nholess = [4, 6, 8, 12, 16, 20, 25]
    # nholess = [25]
    valuesall = {'nf':[], 's':[], 'err':[]}
    for nholes in nholess:
        methods, values, valformat, refdiffinv, xall = test(nholes, plot=True)
        for k in valuesall:
            if k=='err':
                errors = np.sum((xall-refdiffinv)**2, axis=1)
                print("errors", errors)
                valuesall[k].append(errors)
            else: valuesall[k].append(values[k])
    for k in valuesall: valuesall[k] = np.array(valuesall[k])
    # print("valuesall", valuesall)
    nrows = len(valuesall.keys())
    fig, axs = plt.subplots(1, nrows, figsize=(3*nrows,4), squeeze=False)
    for i,(k,v) in enumerate(valuesall.items()):
        ax =axs[0,i]
        for i,m in enumerate(methods):
            ax.plot(nholess, v[:,i], 'X-', label=m)
        ax.legend()
        ax.set_title(k)
    plt.show()


#================================================================#
def testrandom():
    valuesall = {'nf':[], 's':[], 'err':[]}
    percrandoms = [0.0001, 0.001]
    for pr in percrandoms:
        methods, values, valformat, refdiffinv, xall = test(nholes=25, percrandom=pr, plot=True)
        for k in valuesall:
            if k=='err':
                errors = np.sum((xall-refdiffinv)**2, axis=1)
                print("errors", errors)
                valuesall[k].append(errors)
            else: valuesall[k].append(values[k])
    for k in valuesall: valuesall[k] = np.array(valuesall[k])
    # print("valuesall", valuesall)
    nrows = len(valuesall.keys())
    fig, axs = plt.subplots(1, nrows, figsize=(3*nrows,4), squeeze=False)
    for i,(k,v) in enumerate(valuesall.items()):
        ax = axs[0,i]
        for i,m in enumerate(methods):
            ax.plot(percrandoms, v[:,i], 'X-', label=m)
        ax.legend()
        ax.set_title(k)
    plt.show()

#================================================================#

# plotJhat()
testholes()
# testrandom()
