/*.BA*/
/*.KA{C 12}{Zweidim., Oberfl"achen--, B\'ezier--,
            B--Splines}
           {Zweidimensionale Splines, Oberfl"achensplines,
            B\'ezier--Splines, B--Splines}*/
/*.BE*/
/* ---------------------- DEKLARATIONEN bikub.h --------------------- */

/***********************************************************************
 * Include File fuer die Files bikub.c, bezier.c                        *
 ***********************************************************************/

#ifndef BIKUB_H_INCLUDED
#define BIKUB_H_INCLUDED

int  bikub1   (int n, int m, mat4x4** mat, REAL* x, REAL* y);
int  bikub2   (int n, int m, mat4x4** mat, REAL* x, REAL* y);
int  bikub3   (int n, int m, mat4x4** mat,
REAL* x, REAL* y, REAL*** fn);
int  bsval  (int n, int m, mat4x4** mat,
REAL* x, REAL* y, REAL xcoord,
REAL ycoord, REAL* value);
int  xyintv (int n,  int m,  REAL* x,  REAL* y,
int* i, int* j, REAL* xi, REAL* yj,
REAL xcoord, REAL ycoord);

int kubbez                                        /* Bezierpunkte einer Bezier-Spline-Kurve berechnen ...*/
(
REAL   *b[],                                      /* Gewichtspunkte ...........*/
REAL   *d[],                                      /* Bezierpunkte .............*/
int    anz_interpol,
double *laenge,                                   /* Bogenlaengen der Stuetzpunkte*/
double *laenge_plus,                              /* Bogenlaenge dimensionslos */
int    m,                                         /* Anzahl der Splinestuecke  */
int    dim                                        /* 2,3 fuer ebene, Raumkurve */
);                                                /* Fehlercode ...............*/

int valbez                                        /* Auswertung einer Bezier-Spline-Kurve ...............*/

(int    modus,
REAL   t,                                         /* Parameterwert t aus [0,1] */
int    m,                                         /* Anzahl Splinestuecke      */
int    dim,                                       /* 2,3 fuer ebene, Raumkurve */
REAL   *b[],                                      /* Bezierpunkte              */
int    anz_interpol,
double *laenge_plus,                              /* Bogenlaenge dimensionslos */
REAL   *x,                                        /* kartesische Koordinaten   */
REAL   *y,
REAL   *z
);                                                /* Fehlercode                */

int ablbez                                        /* Auswertung einer Bezier-Spline-Kurve ...............*/

(int    modus,
REAL   t,                                         /* Parameterwert t aus [0,1] */
int    m,                                         /* Anzahl Splinestuecke      */
int    dim,                                       /* 2,3 fuer ebene, Raumkurve */
REAL   *b[],                                      /* Bezierpunkte              */
int    anz_interpol,
double *laenge_plus,                              /* Bogenlaenge dimensionslos */
REAL   *dx,                                       /* kartesische Koordinaten   */
REAL   *dy,
REAL   *dz
);                                                /* Fehlercode                */

int  bezier(REAL*** b, REAL*** d, int typ, int m, int n, REAL eps);
int  rechvp(REAL*** b, int m, int n, REAL vp,
int num, REAL *points[]);
int  rechwp(REAL*** b, int m, int n, REAL wp,
int num, REAL *points[]);

int mokube                                        /* Bezierpunkte einer interpolierenden Kurve berechnen */
(
REAL   *b[],                                      /* Gewichtspunkte ...........*/
REAL   *d[],                                      /* Bezierpunkte .............*/
int    m,                                         /* Anzahl der Splinestuecke  */
int    dim,                                       /* 2,3 fuer ebene, Raumkurve */
REAL   eps                                        /* Interpolationsgenauigkeit */
);                                                /* Fehlercode ...............*/
#endif

/* -------------------------- ENDE bikub.h -------------------------- */
