// test-barypol.c

#include <stdio.h>
#include <stdlib.h>  // random

#include "barypol.h"

#ifndef T
#define T float
#endif

// extern int usec();  // since prev usec()

int main( int argc, char* argv[] )
{
    int n = 2, nloop = 1;
    int *ns[] = { &n, &nloop, 0 },
        **np = ns;
    char* arg;
    for( argv ++;  (arg = *argv) && *np;  argv ++, np ++ )
        **np = atof( arg );
    printf( "n %d  nloop %d\n", n, nloop );
	// srandom(1);

	T griddata[] = { 0, 1,
                     2, 3 };
	int shape[] = { 2, 2 };

    const int nq = 4;
	T qpt[] = { 0, 0,  0, 1,  1, 0,  1, 1 };
	T out[nq];

//..............................................................................
	Barypol<T> bary( griddata, shape, 2 );
	bary.at( qpt, nq, out );

    printvec<T> ( out, nq, "barypol: " );
}
