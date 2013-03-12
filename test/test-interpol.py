# test-interpol.py dim= side= ...  12mar
# for testfunc in [...]:
# for method in [Barypol, Intergrid ...]:
#   |interpolated - exact|
# uniform-random query points

from __future__ import division
import sys
from time import time
import numpy as np
import scipy

from barypol.barypol import Barypol  # ../barypol/barypol.pyx ../barypol/h/barypol.h
from intergrid.intergrid import Intergrid  # ../intergrid/intergrid.py
from testfuncs import testfuncs  # [ cos_xav ... on unit cube ]
import util as ut

print 80 * "-"
print "versions: numpy %s  scipy %s  python %s" % (
    np.__version__, scipy.__version__, sys.version.split()[0] )

methods = [Barypol, Intergrid]
dim = 3
side = 10  # uniform grid side^dim
nquery = 1e6
jmethod = -1
jfunc = -1
dtype = np.float64  # map_coord only
order = 1
save = 0
seed = 0

exec( "\n".join( sys.argv[1:] ))  # run this.py side= ...  from sh or ipython
np.set_printoptions( 2, threshold=100, edgeitems=10, suppress=True )
np.random.seed(seed)

nquery = int(nquery)
if jmethod >= 0:
    methods = [methods[jmethod]]
if jfunc >= 0:
    testfuncs = [testfuncs[jfunc]]
shape = dim * [side]

    # side^d x d  000 .. 111 --
grid = np.array( list( np.ndindex(*shape) ), dtype ) / (side - 1)
lo = np.array( dim * [0.] )
hi = np.array( dim * [1.] )
query = np.random.uniform( size=(nquery, dim) ).astype(dtype)

#...............................................................................
print "%s  shape %s  nquery %d " % ( __file__, shape, nquery )

for testfunc in testfuncs:
    funcname = testfunc.__name__
    funcgrid = testfunc( grid ).reshape(shape)
    print ""

    for method in methods:
        methodname = method.__name__
        query_out = np.empty( nquery )

        interpolator = method( funcgrid, lo=lo, hi=hi, order=order, verbose=0 )
        t0 = time()
        interpolator.at( query, query_out )
        ms = (time() - t0) * 1000
        print "%4d ms  |%-9s - exact|  %s:" % (ms, methodname, funcname) ,

        exact = testfunc(query)
        ut.avmaxdiff( query_out, exact, query )
        if save:
            sav = "%s-%s.npy" % (methodname, funcname)
            print "np.save >", sav
            np.save( sav, query_out.astype(np.float32) )

