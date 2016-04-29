/*.BA*/

/*.FE{}{Auswertung von Polynomsplines}
       {Auswertung von Polynomsplines}*/

/*.BE*/
/* -------------------- DEKLARATIONEN spliwert.h -------------------- */

REAL spwert                                       /* Auswertung eines kubischen Polynomsplines .......*/
(
int  n,                                           /* Anzahl der Splinestuecke ...............*/
REAL xwert,                                       /* Auswertungsstelle ......................*/
REAL a[],                                         /* Splinekoeffizienten von (x-x[i])^0 .....*/
REAL b[],                                         /* Splinekoeffizienten von (x-x[i])^1 .....*/
REAL c[],                                         /* Splinekoeffizienten von (x-x[i])^2 .....*/
REAL d[],                                         /* Splinekoeffizienten von (x-x[i])^3 .....*/
REAL x[],                                         /* Stuetzstellen ..........................*/
REAL ausg[]                                       /* 1., 2., 3. Ableitung des Splines .......*/
);                                                /* Funktionswert des Splines ..............*/

void pspwert                                      /* Auswertung eines parametr. kub. Polynomsplines ......*/
(
int      n,                                       /* Anzahl der Splinestuecke ..........*/
REAL     twert,                                   /* Auswertungsstelle .................*/
REAL     t[],                                     /* Stuetzstellen .....................*/
REAL     ax[],                                    /* x-Splinekoeff. von (t-t[i])^0 .....*/
REAL     bx[],                                    /* x-Splinekoeff. von (t-t[i])^1 .....*/
REAL     cx[],                                    /* x-Splinekoeff. von (t-t[i])^2 .....*/
REAL     dx[],                                    /* x-Splinekoeff. von (t-t[i])^3 .....*/
REAL     ay[],                                    /* y-Splinekoeff. von (t-t[i])^0 .....*/
REAL     by[],                                    /* y-Splinekoeff. von (t-t[i])^1 .....*/
REAL     cy[],                                    /* y-Splinekoeff. von (t-t[i])^2 .....*/
REAL     dy[],                                    /* y-Splinekoeff. von (t-t[i])^3 .....*/
REAL     *sx,                                     /* x-Koordinate, .....................*/
REAL     *sy,                                     /* y-Koordinate des Splinewerts ......*/
REAL     *xabl,                                   /* 0. - 3. Ableitung des Splines .....*/
REAL   *yabl
);

REAL hmtwert                                      /* Auswertung eines Hermite-Polynomsplines .......*/
(
int  n,                                           /* Anzahl der Splinestuecke ..............*/
REAL x0,                                          /* Auswertungsstelle .....................*/
REAL a[],                                         /* Splinekoeffizient von (x-x[i])^0 ......*/
REAL b[],                                         /* Splinekoeffizient von (x-x[i])^1 ......*/
REAL c[],                                         /* Splinekoeffizient von (x-x[i])^2 ......*/
REAL d[],                                         /* Splinekoeffizient von (x-x[i])^3 ......*/
REAL e[],                                         /* Splinekoeffizient von (x-x[i])^4 ......*/
REAL f[],                                         /* Splinekoeffizient von (x-x[i])^5 ......*/
REAL x[],                                         /* n+1 Stuetzstellen .....................*/
REAL ausg[]                                       /* 1. - 5. Ableitung des Splines .........*/
);                                                /* Funktionswert des Splines .............*/

void pmtwert                                      /* Auswertung eines parametr. Hermite-Polynomsplines ...*/
(
int      n,                                       /* Anzahl der Splinestuecke ..........*/
REAL     twert,                                   /* Auswertungsstelle .................*/
REAL     t[],                                     /* Stuetzstellen .....................*/
REAL     ax[],                                    /* x-Splinekoeff. von (t-t[i])^0 .....*/
REAL     bx[],                                    /* x-Splinekoeff. von (t-t[i])^1 .....*/
REAL     cx[],                                    /* x-Splinekoeff. von (t-t[i])^2 .....*/
REAL     dx[],                                    /* x-Splinekoeff. von (t-t[i])^3 .....*/
REAL     ex[],                                    /* x-Splinekoeff. von (t-t[i])^4 .....*/
REAL     fx[],                                    /* x-Splinekoeff. von (t-t[i])^5 .....*/
REAL     ay[],                                    /* y-Splinekoeff. von (t-t[i])^0 .....*/
REAL     by[],                                    /* y-Splinekoeff. von (t-t[i])^1 .....*/
REAL     cy[],                                    /* y-Splinekoeff. von (t-t[i])^2 .....*/
REAL     dy[],                                    /* y-Splinekoeff. von (t-t[i])^3 .....*/
REAL     ey[],                                    /* y-Splinekoeff. von (t-t[i])^4 .....*/
REAL     fy[],                                    /* y-Splinekoeff. von (t-t[i])^5 .....*/
REAL     *sx,                                     /* x-Koordinate, .....................*/
REAL     *sy,                                     /* y-Koordinate des Splinewerts ......*/
abl_mat2 ausp                                     /* 0. - 5. Ableitung des Splines .....*/
);

int strwert                                       /* Auswertung eines transf.-param. kub. Polynomsplines ..*/
(
REAL phi,                                         /* Auswertungsstelle ....................*/
int  n,                                           /* Anzahl der Splinestuecke .............*/
REAL phin[],                                      /* Stuetzstellen (Winkel) ...............*/
REAL a[],                                         /* Splinekoeff. von (phi-phin[i])^0 .....*/
REAL b[],                                         /* Splinekoeff. von (phi-phin[i])^1 .....*/
REAL c[],                                         /* Splinekoeff. von (phi-phin[i])^2 .....*/
REAL d[],                                         /* Splinekoeff. von (phi-phin[i])^3 .....*/
REAL phid,                                        /* Drehwinkel des Koordinatensystems ....*/
REAL px,                                          /* Koordinaten des ......................*/
REAL py,                                          /* Verschiebepunktes P ..................*/
REAL ablei[],                                     /* 0. - 3. Ableitung nach x .............*/
REAL *xk,                                         /* x-Koordinate, ........................*/
REAL *yk,                                         /* y-Koordinate des Splinewertes ........*/
REAL *c1,                                         /* 1. Ableitung des Splines (dr/dphi) ...*/
REAL *ckr                                         /* Kruemmung der Splinekurve bei phi ....*/
);                                                /* Fehlercode ...........................*/

/* ------------------------- ENDE spliwert.h ------------------------ */
