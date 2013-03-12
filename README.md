Small fast interpolators:

[Intergrid](http://denis-bz.github.com/docs/intergrid.html):
interpolate in an N-d box grid, uniform or non-uniform.  
This is just a wrapper for scipy.ndimage.map_coordinates and numpy.interp.

[Barypol](http://denis-bz.github.com/docs/barypol.html):
interpolate in a uniform N-d box grid, using d + 1 corners of a simplex  
(triangle, tetrahedron ...) around each query point.  
From Munos and Moore, "Variable Resolution Discretization in Optimal Control",  
1999, 24p; see the pictures and example on pp. 4-5.  
It's implemented in a C++ header barypol.h, with a Cython wrapper.

In 4d, 5d, 6d, Intergrid does around 3M, 2M, .8M interpolations / second.  
Barypol is ~ 5 times faster, but not as smooth.  
(Both will of course become cache-bound for large grids.)  
See interpol/test/*.log; ymmv.

Comments are welcome, testcases most welcome.

