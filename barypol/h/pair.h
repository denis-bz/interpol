// { val, int } pairs to argsort
#ifndef _pair_h
#define _pair_h

template< typename T >
class Pair {
public:
	T val;
	int i;

	Pair():				val(0), i(0) {}
	Pair( T v, int j ): val(v), i(j) {}
	
	bool operator< ( const Pair& other ) const
	{	return val < other.val;
	}
};

#endif
