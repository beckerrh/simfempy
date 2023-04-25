import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.insert(0, simfempypath)

from simfempy.tools.comparemethods import CompareMethods
from simfempy.examples.app_navierstokes import NavierStokes
from simfempy.examples.incompflow import schaeferTurek2d


# def postproc(info):
#     bdrynflux = info.pop('bdrynflux')
#     info['drag'] = -50*bdrynflux[0]
#     info['lift'] = -50*bdrynflux[1]
#     info['err_drag'] =  5.57953523384+50*bdrynflux[0]
#     info['err_lift'] =  0.010618937712+50*bdrynflux[1]

def run(paramsdict, modelargs={}, **kwargs):
    # niter = kwargs.pop('niter', 3)
    # h1 = kwargs.pop('h1', 1)
    # h = [h1*0.5**i for i in range(niter)]
    mesh, data = schaeferTurek2d()
    modelargs['problemdata'] = data
    def createMesh(h): return schaeferTurek2d(h)[0]
    # kwargs['postproc'] = postproc
    comp =  CompareMethods(createMesh=createMesh, paramsdict=paramsdict, model=NavierStokes, modelargs=modelargs, **kwargs)
    result = comp.compare()

#================================================================#
if __name__ == '__main__':
    # paramsdict = {'convmethod': ['lps','supg'], 'linearsolver': ['spsolve', 'gcrotmk', 'bicgstab'], 'precond_p': 'schur'}
    paramsdict = {'convmethod': ['lps'], 'linearsolver': ['spsolve']}
    run(paramsdict, niter=4, h1=0.5)
