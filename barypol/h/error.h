#ifndef _error_h
#define _error_h

#include <assert.h>
#include <stdio.h>

#define Fatal( str, ... ) \
	{ fprintf( stderr, "Fatal error: " str "\n", __VA_ARGS__ ); \
	  assert( 0 ); \
	}

#endif
