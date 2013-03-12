# test-bary.py minimal

from __future__ import division
import sys
import numpy as np
from barypol import Barypol
    # cython/setup.py: cython barypol.pyx ../h/barypol.h -> barypol.so

dim = 2
side = 5  # side^dim points in the unit cube

exec( "\n".join( sys.argv[1:] ))  # run this.py n= ...  from sh or ipython
np.set_printoptions( 1, threshold=100, edgeitems=10, suppress=True )

shape = np.array( [side] * dim )
griddata = np.arange( shape.prod(), dtype=float ).reshape(shape)
lo = np.zeros( dim )
hi = shape - 1.
print "griddata:", shape, "\n", griddata

#...............................................................................
barypol = Barypol( griddata, lo, hi )  # data -> cy -> C class Barypol

    # interpolate at uniformly-spaced points --
igrid = list( np.ndindex(*shape) )
query_points = np.array( igrid, dtype=float ) / (shape - 1)  # 000 .. 111 not hi
print "query_points:", query_points.shape, "\n", query_points.T
out = np.ones( len(query_points) ) * np.NaN

#...............................................................................
barypol.at( query_points, out )  # interpolate
print "barypol:\n", out.reshape(shape)

# todo: plot

