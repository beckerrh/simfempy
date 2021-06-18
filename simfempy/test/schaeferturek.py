import sys
from os import path
simfempypath = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.insert(0,simfempypath)

from simfempy.tools.comparemethods import CompareMethods
from simfempy.applications.navierstokes import NavierStokes
from simfempy.examples.incompflow import schaeferTurek


def postproc(info):
    bdrynflux = info.pop('bdrynflux')
    info['drag'] = -50*bdrynflux[0]
    info['lift'] = -50*bdrynflux[1]
    info['err_drag'] =  5.57953523384+50*bdrynflux[0]
    info['err_lift'] =  0.010618937712+50*bdrynflux[1]

def run(paramsdict, applicationargs={}, **kwargs):
    # niter = kwargs.pop('niter', 3)
    # h1 = kwargs.pop('h1', 1)
    # h = [h1*0.5**i for i in range(niter)]
    mesh, data = schaeferTurek()
    applicationargs['problemdata'] = data
    def createMesh(h): return schaeferTurek(h)[0]
    kwargs['postproc'] = postproc
    comp =  CompareMethods(createMesh=createMesh, paramsdict=paramsdict, application=NavierStokes, applicationargs=applicationargs, **kwargs)
    result = comp.compare()

#================================================================#
if __name__ == '__main__':
    # paramsdict = {'convmethod': ['lps','supg'], 'linearsolver': ['umf', 'gcrotmk', 'bicgstab'], 'precond_p': 'schur'}
    paramsdict = {'convmethod': ['lps'], 'linearsolver': ['gcrotmk'], 'precond_p': 'schur'}
    run(paramsdict, niter=3, h1=0.5)