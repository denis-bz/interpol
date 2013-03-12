#ifndef _printvec_h
#define _printvec_h

#include <stdio.h>

template< typename T >
void printvec( const char* str, const T* x, const int len )
{
    if( str && str[0] )
		printf( str );  // -Wno-format-security
    for( int j = 0; j < len;  j ++ ){
        printf( " %.3g", (double) x[j] );
    }
    printf( "\n" );
}

#endif
