# cython barypol.pyx 12mar
# http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html
# http://stackoverflow.com/questions/tagged/cython numpy c++ arrays

import numpy as np
cimport numpy as np  # Cython/Includes/numpy.pxd

cdef extern from "barypol.h":
    cppclass Barypol_f64:
        Barypol_f64( double*, long*, int, double*, double* )  except +
                    # N-d griddata, shape, dim, lo, hi
        void at( double*, long, double* )
                    # query_points[ npt x dim ] -> out[ npt ]
        int dim

#...............................................................................
cdef class Barypol:
    """ interpolate in a uniform d-dimensional grid
    using dim + 1 corners of a simplex (triangle, tetrahedron ...) around each query point.  
    After Munos and Moore, "Variable Resolution Discretization in Optimal Control", 1999,
    [pdf](http://repository.cmu.edu/cgi/viewcontent.cgi?article=1259&context=robotics).

    Example
    -------
        # set up an interpolator function "barypol" --
    barypol = Barypol( griddata, lo=lo, hi=hi )
        # interpolate at query_points Nquery x dim --
    query_values = np.zeros(Nquery)
    barypol.at( query_points, query_values )  # -> Nquery values

    Parameters
    ----------
    `griddata`: e.g. 4d [lat, lon, alt, time] -> temperature, np.float64
    `lo, hi`: user coordinates of the corners griddata[0 0 0 ...] and griddata[-1 -1 -1 ...]

    `query_points`: Nquery x dim points are clipped to `lo hi`,
        then linearly scaled to `griddata.shape`,
        then interpolated in a d+1 simplex in griddata.

    See also
    --------
    [Intergrid](http://denis-bz.github.com/docs/intergrid.html)
    [Barypol](http://denis-bz.github.com/docs/barypol.html)
    [Barypol on github](http://github.com/denis-bz/interpol/barypol)
    """

#...............................................................................
    cdef Barypol_f64* thisp

    def __init__( self, griddata, lo, hi, **kw):
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] flat = griddata.ravel()
        cdef np.ndarray[long, ndim=1] shape = np.array( griddata.shape )  # long ?
        cdef int dim = griddata.ndim
        cdef np.ndarray[np.float64_t, ndim=1] lo_ = np.array( lo, dtype=float )
        cdef np.ndarray[np.float64_t, ndim=1] hi_ = np.array( hi, dtype=float )
        assert dim > 1, "Barypol: griddata must have ndim > 1"

        self.thisp = new Barypol_f64(
            &flat[0], &shape[0], dim, &lo_[0], &hi_[0] )

#...............................................................................
    def __call__( self,
        np.ndarray[np.float64_t, ndim=2, mode="c"] query_points,
        np.ndarray[np.float64_t, ndim=1] out
        ):
        cdef long npt = query_points.shape[0]
        cdef int dim = query_points.shape[1]
        assert dim == self.thisp.dim, \
            "Barypol: query_points must be npt x dim %d, not %d" % (
                self.thisp.dim, dim )
        # if out is None:
        #     cdef np.ndarray out = np.zeros( npt, dtype=float )  # not here
        #     self.thisp.at( &query_points[0,0], npt, &out[0] )
        #     return out

        self.thisp.at( &query_points[0,0], npt, &out[0] )

    at = __call__

