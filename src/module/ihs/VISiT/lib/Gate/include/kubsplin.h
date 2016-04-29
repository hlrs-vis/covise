/*.BA*/
/*.KA{C 10}{Interpolierende Polynomsplines}
           {Interpolierende Polynomsplines zur Konstruktion glatter
            Kurven}*/
/*.BE*/
/*.FE{C 10.1}
     {Polynomsplines dritten Grades}
     {Polynomsplines dritten Grades}*/

/* -------------------- DEKLARATIONEN kubsplin.h -------------------- */

int spline                                        /* nichtparametrischer kubischer Polynomspline .......*/
(
int  m,                                           /* Anzahl der Stuetzstellen ............*/
REAL x[],                                         /* Stuetzstellen .......................*/
REAL y[],                                         /* Stuetzwerte .........................*/
int  marg_cond,                                   /* Art der Randbedingung ...............*/
REAL marg_0,                                      /* linke Randbedingung .................*/
REAL marg_n,                                      /* rechte Randbedingung ................*/
int  save,                                        /* dynamische Hilfsfelder sichern? .....*/
REAL b[],                                         /* Splinekoeffizienten von (x-x[i]) ....*/
REAL c[],                                         /* Splinekoeffizienten von (x-x[i])^2 ..*/
REAL d[]                                          /* Splinekoeffizienten von (x-x[i])^3 ..*/
);                                                /* Fehlercode ..........................*/

int parspl                                        /* parametrischer kubischer Polynomspline .......*/
(
int  m,                                           /* Anzahl der Stuetzpunkte .............*/
REAL x[],                                         /* x-Koordinaten der Stuetzpunkte ......*/
REAL y[],                                         /* y-Koordinaten der Stuetzpunkte ......*/
int  marg_cond,                                   /* Art der Randbedingung ...............*/
REAL marg_0[],                                    /* linke Randbedingungen ...............*/
REAL marg_n[],                                    /* rechte Randbedingungen ..............*/
int  cond_t,                                      /* Parameterstuetzstellen vorgegeben? ..*/
REAL t[],                                         /* Parameterstuetzstellen ..............*/
REAL bx[],                                        /* x-Splinekoeffiz. fuer (t-t[i]) ......*/
REAL cx[],                                        /* x-Splinekoeffiz. fuer (t-t[i])^2 ....*/
REAL dx[],                                        /* x-Splinekoeffiz. fuer (t-t[i])^3 ....*/
REAL by[],                                        /* y-Splinekoeffiz. fuer (t-t[i]) ......*/
REAL cy[],                                        /* y-Splinekoeffiz. fuer (t-t[i])^2 ....*/
REAL dy[]                                         /* y-Splinekoeffiz. fuer (t-t[i])^3 ....*/
);                                                /* Fehlercode ..........................*/

int spltrans                                      /* transformiert-parametr. kub. Polynomspline .......*/
(
int  m,                                           /* Anzahl der Stuetzpunkte ..............*/
REAL x[],                                         /* Stuetzstellen ........................*/
REAL y[],                                         /* Stuetzwerte ..........................*/
int  mv,                                          /* Art der Koordinatenverschiebung ......*/
REAL px[],                                        /* Koordinaten des ......................*/
REAL py[],                                        /* Verschiebepunktes P ..................*/
REAL a[],                                         /* Splinekoeff. von (phi-phin[i])^0 .....*/
REAL b[],                                         /* Splinekoeff. von (phi-phin[i]) .......*/
REAL c[],                                         /* Splinekoeff. von (phi-phin[i])^2 .....*/
REAL d[],                                         /* Splinekoeff. von (phi-phin[i])^3 .....*/
REAL phin[],                                      /* Winkelkoordinaten der Stuetzpunkte ...*/
REAL *phid                                        /* Drehwinkel des Koordinatensystems ....*/
);                                                /* Fehlercode ...........................*/

/* ------------------------- ENDE kubsplin.h ------------------------ */
