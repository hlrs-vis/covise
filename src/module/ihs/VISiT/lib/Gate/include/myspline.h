int spline                                        /* nichtparametrischer kubischer Polynomspline .......*/
(
int  m,                                           /* Anzahl der Stuetzstellen ............*/
double x[],                                       /* Stuetzstellen .......................*/
double y[],                                       /* Stuetzwerte .........................*/
int  marg_cond,                                   /* Art der Randbedingung ...............*/
double marg_0,                                    /* linke Randbedingung .................*/
double marg_n,                                    /* rechte Randbedingung ................*/
int  save,                                        /* dynamische Hilfsfelder sichern? .....*/
double b[],                                       /* Splinekoeffizienten von (x-x[i]) ....*/
double c[],                                       /* Splinekoeffizienten von (x-x[i])^2 ..*/
double d[]                                        /* Splinekoeffizienten von (x-x[i])^3 ..*/
);                                                /* Fehlercode ..........................*/

double spwert                                     /* Auswertung eines kubischen Polynomsplines .......*/
(
int  n,                                           /* Anzahl der Splinestuecke ...............*/
double xwert,                                     /* Auswertungsstelle ......................*/
double a[],                                       /* Splinekoeffizienten von (x-x[i])^0 .....*/
double b[],                                       /* Splinekoeffizienten von (x-x[i])^1 .....*/
double c[],                                       /* Splinekoeffizienten von (x-x[i])^2 .....*/
double d[],                                       /* Splinekoeffizienten von (x-x[i])^3 .....*/
double x[],                                       /* Stuetzstellen ..........................*/
double ausg[]                                     /* 1., 2., 3. Ableitung des Splines .......*/
);                                                /* Funktionswert des Splines ..............*/

int parspl                                        /* parametrischer kubischer Polynomspline .......*/
(
int  m,                                           /* Anzahl der Stuetzpunkte .............*/
double x[],                                       /* x-Koordinaten der Stuetzpunkte ......*/
double y[],                                       /* y-Koordinaten der Stuetzpunkte ......*/
int  marg_cond,                                   /* Art der Randbedingung ...............*/
double marg_0[],                                  /* linke Randbedingungen ...............*/
double marg_n[],                                  /* rechte Randbedingungen ..............*/
int  cond_t,                                      /* Parameterstuetzstellen vorgegeben? ..*/
double t[],                                       /* Parameterstuetzstellen ..............*/
double bx[],                                      /* x-Splinekoeffiz. fuer (t-t[i]) ......*/
double cx[],                                      /* x-Splinekoeffiz. fuer (t-t[i])^2 ....*/
double dx[],                                      /* x-Splinekoeffiz. fuer (t-t[i])^3 ....*/
double by[],                                      /* y-Splinekoeffiz. fuer (t-t[i]) ......*/
double cy[],                                      /* y-Splinekoeffiz. fuer (t-t[i])^2 ....*/
double dy[]                                       /* y-Splinekoeffiz. fuer (t-t[i])^3 ....*/
);                                                /* Fehlercode ..........................*/

void pspwert                                      /* Auswertung eines parametr. kub. Polynomsplines ......*/
(
int      n,                                       /* Anzahl der Splinestuecke ..........*/
double     twert,                                 /* Auswertungsstelle .................*/
double     t[],                                   /* Stuetzstellen .....................*/
double     ax[],                                  /* x-Splinekoeff. von (t-t[i])^0 .....*/
double     bx[],                                  /* x-Splinekoeff. von (t-t[i])^1 .....*/
double     cx[],                                  /* x-Splinekoeff. von (t-t[i])^2 .....*/
double     dx[],                                  /* x-Splinekoeff. von (t-t[i])^3 .....*/
double     ay[],                                  /* y-Splinekoeff. von (t-t[i])^0 .....*/
double     by[],                                  /* y-Splinekoeff. von (t-t[i])^1 .....*/
double     cy[],                                  /* y-Splinekoeff. von (t-t[i])^2 .....*/
double     dy[],                                  /* y-Splinekoeff. von (t-t[i])^3 .....*/
double     *sx,                                   /* x-Koordinate, .....................*/
double     *sy,                                   /* y-Koordinate des Splinewerts ......*/
double     *xabl,
double     *yabl
);
