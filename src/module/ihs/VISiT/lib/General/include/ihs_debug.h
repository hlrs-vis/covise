
#define  WD_S(x)     D_S(x);W_S(x)
#define  WD_SS(x,y)  D_SS(x,y);W_SS(x,y)
#define  WD_I(x,y)   D_I(x,y);W_I(x,y)
#define  WD_D(x,y)   D_D(x,y);W_D(x,y)

#ifdef   IHS_DEBUG

#ifndef  EXTERN
#define  EXTERN extern
#endif                                            /* EXTERN	*/

EXTERN FILE *ihs_debug;

//#define	D_OPEN	{if (!ihs_debug) { ihs_debug = fopen("/tmp/ihs_debug", "w"); } }
#define  D_OPEN   {ihs_debug = stderr;}
#define  D_S(x)   {D_OPEN;if (ihs_debug)   {fputs((x), ihs_debug);fflush(ihs_debug);}}
#define  D_SS(x,y)   {D_OPEN;if (ihs_debug)   {fputs((x), ihs_debug);fputs((y), ihs_debug);fputs("\n", ihs_debug);fflush(ihs_debug);}}
#define  D_I(x,y) {D_OPEN;if (ihs_debug)   {fprintf(ihs_debug, "%s %d\n", (x), (y));fflush(ihs_debug);}}
#define  D_D(x,y) {D_OPEN;if (ihs_debug)   {fprintf(ihs_debug, "%s %f\n", (x), (y));fflush(ihs_debug);}}

#else                                             /* IHS_DEBUG	*/

#define  D_S(x)
#define  D_SS(x,y)
#define  D_I(x, y)
#define  D_D(x,y)
#endif                                            /* IHS_DEBUG	*/

#define  W_S(x)   {{fputs((x), stderr);fflush(stderr);}}
#define  W_SS(x,y)   {{fputs((x), stderr);fputs((y), stderr);;fputs("\n", stderr);fflush(stderr);}}
#define  W_I(x,y) {if (stderr)    {fprintf(stderr, "%s %d\n", (x), (y));fflush(stderr);}}
#define  W_D(x,y) {if (stderr)    {fprintf(stderr, "%s %f\n", (x), (y));fflush(stderr);}}

#ifndef  DIM
#define  DIM(x)   (sizeof(x)/sizeof(*(x)))
#endif                                            /* DIM	*/
