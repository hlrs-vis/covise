#ifndef COMMON_INCLUDE
#define COMMON_INCLUDE

#include <math.h>
// common things ...
#define  DIM(x)   (sizeof(x)/sizeof(*x))
#define  IS_0(x)  ((fabs(x) > 0.00001) ? 0 : 1)
#define  ABS(a)	  ( (a) >= (0) ? (a) : -(a) )
#define  SIGN(a)  ( (a) >= (0) ? (1) : -(1) )
#define  MIN(a,b) ( (a) <  (b) ? (a) :	(b) )
#define  MAX(a,b) ( (a) >  (b) ? (a) :	(b) )

#endif // COMMON_INCLUDE
