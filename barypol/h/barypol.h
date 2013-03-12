// barypol.h: barycentric interpolation on an N-dimensional uniform grid
// in time to sort d coordinates, ~ d ln d
// from Munos and Moore, "Variable Resolution Discretization in Optimal Control"
// 1999, 24p; pictures, example p. 4-5
/*
    set up a Barypol interpolator:
        Barypol<T> bary( T griddata[], int shape[], int dim, T lo[], T hi[] )
    interpolate at npt points:
        bary.at( T query_points[], npt, T out[] )
        does
        for each query point:
            clip, scale -> a cube in griddata
            -> dim+1 corners of a triangle / tetrahedron / simplex around the point
            -> sum wj * grid[ cornerj ],  wj >= 0, sum wj = 1

	Parameters
    ----------
	Barypol( T griddata[], int shape[], int dim, T lo[], T hi[]  all const )
        griddata: a flat array of prod(shape) numbers
        shape: e.g. C [2][3][5] / numpy shape
        dim: here 3
        lo, hi: user coordinates for griddata[0], griddata[last]
            so barypol(lo) == griddata[0]
            barypol(hi) == griddata[last]
            defaults: lo NULL -> 0, hi NULL -> shape - 1

    bary.at( T query_points[], npt, T out[] )
        query_points: a flat array of npt x dim numbers
        npt: number of points
        out:

 */

// Copyright (c) 2013 Denis Bzowy <denis-bz-py@t-online.de>
// comments welcome, testcases most welcome
// License: GPLv3 http://en.wikipedia.org/wiki/GNU_General_Public_License
// no use in closed-source


#include <algorithm>
#include <assert.h>
#include <stdio.h>

#include "error.h"  // Fatal( str, ... ): printf stderr, assert(0)
#include "pair.h"  // val, i pairs to argsort
#include "printvec.h"

#ifndef Barypol_test
#define Barypol_test 0
#endif
static char Barypol_version[] = "2013-03-12 mar denis";

//..............................................................................
template< typename T >
class Barypol
{
public:
	const T* griddata;
	long* shape;
	const int dim;
	double* lo;
	double* hi_lo;
private:
	long* strides;
	double* xscale;
	Pair<T>* xsort;
public:

//..............................................................................
	Barypol(
		const T* griddata_,
		const long* shape_,
		const int dim_,  // len(shape)
		const T* lo_ = 0,  // default all 0
		const T* hi_ = 0   // default shape - 1
		)
		: griddata(griddata_),
        shape( new long[dim_] ),
        dim(dim_),
		lo( new double[dim] ),
		hi_lo( new double[dim] ),
		strides( new long[dim] ),
		xscale( new double[dim] ),
		xsort( new Pair<T>[dim+2] )

	{  // split off .cpp ? mttiw
			// xscale so (clip x - lo) * xscale -> [0, shape - 1)
			// (better class Scale w maps a la Intergrid)
		for( int j = 0; j < dim; j ++ ){
            shape[j] = shape_[j];
			double l = (lo_ ? lo_[j] : 0);
			double h = (hi_ ? hi_[j] : shape[j] - 1);
			if( l < h ){
				lo[j] = l;
				hi_lo[j] = h - l;
				double t = shape[j] - 1;
				if( t > 0 )  t -= 1e-6;  // so xint strictly < edge
				xscale[j] = t / hi_lo[j];
			} else if( l == h ){  // shape 1
				lo[j] = hi_lo[j] = xscale[j] = 0;
			} else {
				Fatal( "lo %g >= hi %g", l, h );
			}
		}
			// shape [any 10 2] -> strides [20 2 1] --
		long p = 1;
		for( int j = dim - 1; j >= 0; j -- ){
			strides[j] = p;
			p *= shape[j];
		}
		xsort[0].val = 0;  // 0, x[dim] sorted, 1  so weights val[j+1] - val[j]
		xsort[dim+1].val = 1;
	}

	// ~Barypol() {}  // ?

//..............................................................................
	double _at1( const T* X )  // barypol 1 point
		// clip, scale, int + frac, argsort, sum: see Munos & Moore
	{
		if( Barypol_test )
			printvec<T> ( "test barypol: ", X, dim );
		long jflat = 0;  // -> griddata[ corner ]
		for( int j = 0; j < dim; j ++ ){
			T x = X[j] - lo[j];
			if( x < 0 )
				x = 0;
			else if( x > hi_lo[j] )
				x = hi_lo[j];
			x *= xscale[j];

			long xint = x;
			if( not (xint == 0  || (0 <= xint  && xint < shape[j] - 1 )))
                Fatal( "x %g  xint %d  shape %d  xscale %g",
					x, xint, shape[j], xscale[j] );
				// cython keeps going
			jflat += xint * strides[j];
			xsort[j+1].val = x - xint;
			xsort[j+1].i = j;
		}

		std::sort( xsort + 1, xsort + 1 + dim );  // argsort

			// dim+1 bary weights * corner vals --
		double sum = 0;
		for( int j = dim; j >= 0; j -- ){
			double wt = xsort[j+1].val - xsort[j].val;
			sum += wt * griddata[jflat];
			if( Barypol_test )
				printf( "%d %.3g %.3g  ", jflat, wt, sum );
			jflat += strides[ xsort[j].i ];  // walk Hamming cube
		}
		if( Barypol_test )
            printf( "\n" );

		return sum;
	}

//..............................................................................
		// at: query[ npt, dim ] -> out[npt]
	void at( const T* query, const long npt, T* out )
	{
		for( long j = 0, jx = 0;  j < npt;  j ++, jx += dim ){
			out[j] = _at1( &query[jx] );  // double -> T
		}
	}

};  // Barypol

typedef Barypol<double> Barypol_f64;
