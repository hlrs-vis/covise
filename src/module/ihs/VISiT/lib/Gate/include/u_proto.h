/* --------------------- DEKLARATIONEN u_proto.h -------------------- */

/*--------------------------------------------------------------------*
 * Include Datei zur Vordeklaration aller Bibliotheksfunktionen.      *
 *                                                                    *
 *--------------------------------------------------------------------*/

#ifndef U_PROTO_H_INCLUDED

/* Gegen mehrfaches includieren absichern */

#define U_PROTO_H_INCLUDED

/*--------------------------------------------------------------------*
 * Vordeklaration aller externen Bibliotheksfunktionen                *
 *--------------------------------------------------------------------*/

/*--------------------------------------------------------------------*
 * P 2  Numerische Verfahren zur Loesung nichtlinearer Gleichungen ...*
 *--------------------------------------------------------------------*/

int newton (                                      /* Eindimensionales Newton Verfahren .........*/
REALFCT  fct,                                     /* Funktion ........................*/
REALFCT  fderv,                                   /* 1. Ableitung ....................*/
REAL   * x,                                       /* Startwert / Loesung .............*/
REAL   * fval,                                    /* Funktionswert an Loesung........ */
int    * iter                                     /* Iterationszahl ..................*/
);

int newpoly (                                     /* Newton Verfahren fuer Polynome ...........*/
int      n,                                       /* Polynomgrad .....................*/
REAL     coeff[],                                 /* Koeffizientenvektor .............*/
REAL *   x,                                       /* Startwert / Loesung .............*/
REAL *   fval,                                    /* Funktionswert an x ..............*/
int *    iter                                     /* Iterationszahl ..................*/
);

int polval (                                      /* Newton Verfahren fuer Polynome ...........*/
int      n,                                       /* Grad des Polynoms ...............*/
REAL     coeff[],                                 /* Koeffizientenvektor .............*/
REAL     x,                                       /* Auswertestelle ..................*/
REAL *   val,                                     /* Wert des Polynoms an x ..........*/
REAL *   dval                                     /* Wert der 1. Abl. an x ...........*/
);

int newmod (                                      /* Modifiziertes Newton Verfahren ............*/
REALFCT  fct,                                     /* Funktion ........................*/
REALFCT  fderv1,                                  /* 1. Ableitung ....................*/
REALFCT  fderv2,                                  /* 2. Ableitung ....................*/
REAL *   x,                                       /* Startwert / Loesung .............*/
REAL *   fval,                                    /* Funktionswert an x ..............*/
int *    iter,                                    /* Iterationszahl ..................*/
int *    mul                                      /* Vielfachheit der Nullst. ........*/
);

int pegasus (                                     /* Pegasus Verfahren .........................*/
REALFCT  fct,                                     /* Funktion ........................*/
REAL *   x1,                                      /* Startwert 1 .....................*/
REAL *   x2,                                      /* Startwert 2 / Loesung ...........*/
REAL *   f2,                                      /* Funktionswert an x2 .............*/
int *    iter                                     /* Iterationszahl ..................*/
);

int roots (                                       /* Pegasus, Anderson-Bjoerck-King Verfahren ..*/
int      method,                                  /* Verfahren .......................*/
REALFCT  fct,                                     /* Funktion ........................*/
int      quadex,                                  /* Quadratische Extrapolation.......*/
REAL *   x1,                                      /* Startwert 1 .....................*/
REAL *   x2,                                      /* Startwert 2 / Loesung ...........*/
REAL *   fx2,                                     /* Funktionswert an x2 .............*/
int *    iter                                     /* Iterationszahl ..................*/
);

/*--------------------------------------------------------------------*
 * P 3  Verfahren zur Loesung algebraischer Gleichungen ..............*
 *--------------------------------------------------------------------*/

int mueller (                                     /* Mueller Verfahren fuer reelle Polynome ....*/
int    n,                                         /* Polynomgrad .....................*/
REAL   a[],                                       /* Koeffizientenvektor .............*/
int    scaleit,                                   /* Skalieren .......................*/
REAL   zreal[],                                   /* Realteile Loesung ...............*/
REAL   zimag[]                                    /* Imaginaerteile Loesung ..........*/
);

void fmval (                                      /* (Komplexer) Polynomwert ...................*/
int      n,                                       /* Hoechster Koeffizient ...........*/
int      iu,                                      /* Niedrigster Koeffizient .........*/
REAL     zre[],                                   /* Koeffizienten ...................*/
REAL     zren,                                    /* Fuehrender Koeffizient ..........*/
REAL     xre,                                     /* Realteil x ......................*/
REAL     xim,                                     /* Imaginaerteil x .................*/
REAL *   fre,                                     /* Realteil Funktionswert ..........*/
REAL *   fim                                      /* Imaginaerteil Funktionswert .....*/
);

int bauhub (                                      /* Bauhuber Verfahren fuer komplexe Polynome .*/
int    real,                                      /* Koeffizienten sind reell ? ......*/
int    scale,                                     /* Skalieren ? .....................*/
int    n,                                         /* Polynomgrad .....................*/
REAL   ar[],                                      /* Realteile Koeffizienten .........*/
REAL   ai[],                                      /* Imaginaerteile Koeffizienten ....*/
REAL   rootr[],                                   /* Realteile Nullstellen ...........*/
REAL   rooti[],                                   /* Imaginaerteile Nullstellen ......*/
REAL   absf[]                                     /* Absolutbetraege der F-werte .....*/
);

/*--------------------------------------------------------------------*
 * P 4  Direkte Verfahren zur Loesung linearer Gleichungssysteme .....*
 *--------------------------------------------------------------------*/

int gauss (                                       /* Gauss Verfahren zur Loesung von lin. GLS ..*/
int      mod,                                     /* Modus: 0, 1, 2, 3 ...............*/
int      n,                                       /* Dimension der Matrix ............*/
REAL *   mat[],                                   /* Eingabematrix ...................*/
REAL *   lumat[],                                 /* LU Zerlegung ....................*/
int      perm[],                                  /* Zeilenvertauschungen ............*/
REAL     b[],                                     /* Rechte Seite des Systems ........*/
REAL     x[],                                     /* Loesung des Systems .............*/
int *    signd                                    /* Vorzeichen Determinante .........*/
);

int gaudec (                                      /* Gauss Zerlegung ...........................*/
int      n,                                       /* Dimension der Matrix ............*/
REAL *   mat[],                                   /* Eingabematrix ...................*/
REAL *   lumat[],                                 /* Zerlegungsmatrix ................*/
int      perm[],                                  /* Zeilenvertauschungen ............*/
int *    signd                                    /* Vorzeichen Determinante .........*/
);

int gausol (                                      /* Gauss Loesung .............................*/
int      n,                                       /* Dimension der Matrix ............*/
REAL *   lumat[],                                 /* Eingabematrix (LU) ..............*/
int      perm[],                                  /* Zeilenvertauschungen ............*/
REAL     b[],                                     /* Rechte Seite ....................*/
REAL     x[]                                      /* Loesung .........................*/
);

int gausoli (                                     /* Gauss Loesung mit Nachiteration ..........*/
int      n,                                       /* Dimension der Matrix ...........*/
REAL *   mat[],                                   /* Ausgangsmatrix .................*/
REAL *   lumat[],                                 /* Eingabematrix (LU) .............*/
int      perm[],                                  /* Zeilenvertauschungen ...........*/
REAL     b[],                                     /* Rechte Seite ...................*/
REAL     x[]                                      /* Loesung ........................*/
);

int mgauss (                                      /* Gauss Verfahren fuer mehrere rechte Seiten */
int       n,                                      /* Dimension der Matrix ............*/
int       k,                                      /* Anzahl rechter Seiten ...........*/
REAL *    mat[],                                  /* Eingabematrix ...................*/
REAL *    rmat[]                                  /* Rechte Seiten /Loesungen ........*/
);

REAL   det (                                      /* Determinante ..............................*/
int      n,                                       /* Dimension der Matrix ............*/
REAL *   mat[]                                    /* Eingabematrix ...................*/
);

int choly (                                       /* Cholesky Verfahren ........................*/
int       mod,                                    /* Modus: 0, 1, 2 ..................*/
int       n,                                      /* Dimension der Matrix ............*/
REAL *    mat[],                                  /* Eingabematrix ...................*/
REAL      b[],                                    /* Rechte Seite des Systems ........*/
REAL      x[]                                     /* Loesung .........................*/
);

int chodec (                                      /* Cholesky Zerlegung ........................*/
int      n,                                       /* Dimension der Matrix ............*/
REAL *   mat[]                                    /* Eingabematrix/LU Matrix .........*/
);

int chosol (                                      /* Cholesky Loesung ..........................*/
int      n,                                       /* Dimension der Matrix ............*/
REAL *   lmat[],                                  /* LU Matrix .......................*/
REAL     b[],                                     /* Rechte Seite des Systems ........*/
REAL     x[]                                      /* Loesung .........................*/
);

int pivot (                                       /* Bestimmung der Inversen (Austauschverf.) ..*/
int      n,                                       /* Dimension der Matrix ............*/
REAL *   mat[],                                   /* Eingabematrix ...................*/
REAL *   inv[],                                   /* Inverse .........................*/
REAL *   s,                                       /* Checksumme ......................*/
REAL *   cond                                     /* Konditionzahl ...................*/
);

int trdiag (                                      /* Tridiagonale Gleichungssysteme ............*/
int    n,                                         /* Dimension der Matrix ............*/
REAL   lower[],                                   /* Subdiagonale ....................*/
REAL   diag[],                                    /* Diagonale .......................*/
REAL   upper[],                                   /* Superdiagonale ..................*/
REAL   b[],                                       /* Rechte Seite / Loesung ..........*/
int    rep                                        /* rep = 0, 1 ......................*/
);

int tzdiag (                                      /* Zyklisch tridiagonale Gleichungssystem ....*/
int    n,                                         /* Dimension der Matrix ............*/
REAL   lower[],                                   /* Subdiagonale ....................*/
REAL   diag[],                                    /* Diagonale .......................*/
REAL   upper[],                                   /* Superdiagonale ..................*/
REAL   lowrow[],                                  /* Untere Zeile ....................*/
REAL   ricol[],                                   /* Rechte Spalte ...................*/
REAL   b[],                                       /* Rechte Seite / Loesung ..........*/
int    rep                                        /* rep = 0, 1 ......................*/
);

int diag5 (                                       /* 5 diagonale Gleichungssysteme .............*/
int    mod,                                       /* Modus: 0, 1, 2 ..................*/
int    n,                                         /* # Matrixzeilen ..................*/
REAL   ld2[],                                     /* 2. untere Subdiagonale ..........*/
REAL   ld1[],                                     /* 1. untere Subdiagonale ..........*/
REAL   d[],                                       /* Hauptdiagonale ..................*/
REAL   ud1[],                                     /* 1. obere Superdiagonale .........*/
REAL   ud2[],                                     /* 2. obere Superdiagonale .........*/
REAL   b[]                                        /* Rechte Seite/Loesung ............*/
);

int diag5dec (                                    /* Zerlegung des 5 diagonalen Systems ........*/
int    n,                                         /* # Matrixzeilen ..................*/
REAL   ld2[],                                     /* 2. untere Subdiagonale ..........*/
REAL   ld1[],                                     /* 1. untere Subdiagonale ..........*/
REAL   d[],                                       /* Hauptdiagonale ..................*/
REAL   ud1[],                                     /* 1. obere Superdiagonale .........*/
REAL   ud2[]                                      /* 2. obere Superdiagonale .........*/
);

int diag5sol (                                    /* Loesung des 5 diagonalen Systems ..........*/
int    n,                                         /* # Matrixzeilen ..................*/
REAL   ld2[],                                     /* 2. untere Subdiagonale ..........*/
REAL   ld1[],                                     /* 1. untere Subdiagonale ..........*/
REAL   d[],                                       /* Hauptdiagonale ..................*/
REAL   ud1[],                                     /* 1. obere Superdiagonale .........*/
REAL   ud2[],                                     /* 2. obere Superdiagonale .........*/
REAL   b[]                                        /* Rechte Seite / Loesung ..........*/
);

int diag5pd (                                     /* fuenfdiagonale streng regulaere Matrizen ..*/
int    mod,                                       /* Modus: 0, 1, 2 ..................*/
int    n,                                         /* # Matrixzeilen ..................*/
REAL   d[],                                       /* Hauptdiagonale ..................*/
REAL   ud1[],                                     /* 1. obere Superdiagonale .........*/
REAL   ud2[],                                     /* 2. obere Superdiagonale .........*/
REAL   b[]                                        /* Rechte Seite des Systems ........*/
);

int diag5pddec (                                  /* Zerlegung 5 diagonaler str. reg. Matrizen .*/
int    n,                                         /* # Matrixzeilen ..................*/
REAL   d[],                                       /* Hauptdiagonale ..................*/
REAL   ud1[],                                     /* 1. obere Superdiagonale .........*/
REAL   ud2[]                                      /* 2. obere Superdiagonale .........*/
);

int diag5pdsol (                                  /* Loesung 5 diagonaler str. reg. GLS ........*/
int    n,                                         /* # Matrixzeilen ..................*/
REAL   d[],                                       /* Hauptdiagonale ..................*/
REAL   ud1[],                                     /* 1. obere Superdiagonale .........*/
REAL   ud2[],                                     /* 2. obere Superdiagonale .........*/
REAL   b[]                                        /* Rechte Seite des Systems ........*/
);

int pack (                                        /* Zeile packen ..............................*/
int    n,                                         /* Dimension der Matrix ............*/
int    ld,                                        /* Anzahl Subdiagonalen ............*/
int    ud,                                        /* Anzahl Superdiagonalen ..........*/
int    no,                                        /* Zeilennummer ....................*/
REAL   row[],                                     /* Zeile ...........................*/
REAL   prow[]                                     /* Gepackte Zeile ..................*/
);

int unpack (                                      /* Zeile entpacken ...........................*/
int    n,                                         /* Dimension der Matrix ............*/
int    ld,                                        /* Anzahl Subdiagonalen ............*/
int    ud,                                        /* Anzahl Superdiagonalen ..........*/
int    no,                                        /* Zeilennummer ....................*/
REAL   prow[],                                    /* Gepackte Zeile ..................*/
REAL   row[]                                      /* Entpackte Zeile .................*/
);

int band (                                        /* Gleichungssysteme mit Bandmatrizen ........*/
int      mod,                                     /* Modus: 0, 1, 2 ..................*/
int      n,                                       /* # Zeilen ........................*/
int      ld,                                      /* # untere Diagonalen .............*/
int      ud,                                      /* # obere Diagonalen ..............*/
REAL *   pmat[],                                  /* gepackte Eingabematrix ..........*/
REAL     b[],                                     /* rechte Seite des Systems ........*/
int      perm[],                                  /* Zeilenvertauschungen ............*/
int *    signd                                    /* Vorzeichen Determinante .........*/
);

int banddec (                                     /* Zerlegung der Bandmatrix ..................*/
int      n,                                       /* # Zeilen ........................*/
int      ld,                                      /* # untere Diagonalen .............*/
int      ud,                                      /* # obere Diagonalen ..............*/
REAL *   pmat[],                                  /* gepackte Ein-/Ausgabematrix .....*/
int      perm[],                                  /* rechte Seite des Systems ........*/
int *    signd                                    /* Vorzeichen Determinante .........*/
);

int bandsol (                                     /* Loesung des Bandsystems ...................*/
int      n,                                       /* # Zeilen ........................*/
int      ld,                                      /* # untere Diagonalen .............*/
int      ud,                                      /* # obere Diagonalen ..............*/
REAL *   pmat[],                                  /* gepackte Eingabematrix ..........*/
REAL     b[],                                     /* rechte Seite des Systems ........*/
int      perm[]                                   /* Zeilenvertauschungen ............*/
);

int bando (                                       /* GLS mit Bandmatrizen (ohne Pivot) .........*/
int      mod,                                     /* Modus: 0, 1, 2 ..................*/
int      n,                                       /* # Zeilen ........................*/
int      ld,                                      /* # untere Diagonalen .............*/
int      ud,                                      /* # obere Diagonalen ..............*/
REAL *   pmat[],                                  /* gepackte Eingabematrix ..........*/
REAL     b[]                                      /* rechte Seite des Systems ........*/
);

int banodec (                                     /* Zerlegung der Bandmatrix ..................*/
int      n,                                       /* # Zeilen ........................*/
int      ld,                                      /* # untere Diagonalen .............*/
int      ud,                                      /* # obere Diagonalen ..............*/
REAL *   pmat[]                                   /* Ein-/Ausgabematrix ..............*/
);

int banosol (                                     /* Bandloesung ...............................*/
int      n,                                       /* # Zeilen ........................*/
int      ld,                                      /* # untere Diagonalen .............*/
int      ud,                                      /* # obere Diagonalen ..............*/
REAL *   pmat[],                                  /* Eingabematrix ...................*/
REAL     b[]                                      /* Rechte Seite / Loesung ..........*/
);

int house (                                       /* Householder Verfahren .....................*/
int      m,                                       /* # Zeilen ........................*/
int      n,                                       /* # Spalten .......................*/
REAL *   mat[],                                   /* Eingabematrix ...................*/
REAL     b[]                                      /* Rechte Seite, Loesung ...........*/
);

int mhouse (                                      /* Householder Verfahren (m. recht. Seiten) ..*/
int      m,                                       /* # Zeilen ........................*/
int      n,                                       /* # Spalten .......................*/
int      k,                                       /* # rechter Seiten ................*/
REAL *   mat[],                                   /* Eingabematrix ...................*/
REAL *   xmat[]                                   /* Rechte Seiten / Loesungsvektoren */
);

REAL   hcond (                                    /* Hadamardsche Konditionszahl ...............*/
int      n,                                       /* Dimension der Matrix ............*/
REAL *   mat[]                                    /* Eingabematrix ...................*/
);

REAL   ccond (                                    /* Konditionszahl nach Cline .................*/
int       n,                                      /* Dimension der Matrix ............*/
REAL *    mat[]                                   /* Eingabematrix ...................*/
);

REAL   fcond (                                    /* Konditionszahl nach Forsythe/Moler ........*/
int       n,                                      /* Dimension der Matrix ............*/
REAL *    mat[]                                   /* Eingabematrix ...................*/
);

/*--------------------------------------------------------------------*
 * P 5  Iterationsverfahren zur Loesung linearer Gleichungssysteme ...*
 *--------------------------------------------------------------------*/

int seidel (                                      /* Gauss Seidel Iterationsverfahren ..........*/
int      crit,                                    /* crit = 0, 1, 2, 3 ...............*/
int      n,                                       /* Dimension der Matrix ............*/
REAL *   mat[],                                   /* Eingabematrix ...................*/
REAL     b[],                                     /* Rechte Seite ....................*/
REAL     omega,                                   /* Relaxaktionskoeffizient .........*/
REAL     x[],                                     /* Loesung .........................*/
REAL     residu[],                                /* Residuen ........................*/
int *    iter                                     /* # Iterationen ...................*/
);

/*--------------------------------------------------------------------*
 * P 6  Systeme nichtlinearer Gleichungen ............................*
 *--------------------------------------------------------------------*/

int newt (                                        /* Mehrdimensionales Newton Verfahren .......*/
int      n,                                       /* Dimension des Systems ...........*/
REAL     x[],                                     /* Start-/Loesungsvektor ...........*/
FNFCT    fct,                                     /* Funktion ........................*/
JACOFCT  jaco,                                    /* Funktion zur Best. der Jacobi Mat*/
int      kmax,                                    /* Maximalzahl Daempfungsschritte ..*/
int      prim,                                    /* Maximalzahl Primitivschritte ....*/
char *   pfile,                                   /* Name der Protokolldatei .........*/
REAL     fvalue[],                                /* Funktionswert an Loesung ........*/
int *    iter,                                    /* Anzahl Iterationsschritte .......*/
REAL     eps                                      /* Fehlerschranke ..................*/
);

/*--------------------------------------------------------------------*
 * P 7  Eigenwerte und Eigenvektoren von Matrizen ....................*
 *--------------------------------------------------------------------*/

int mises (                                       /* von Mises Verfahren zur Eigenwertbest. ....*/
int      n,                                       /* Dimension der Matrix ............*/
REAL *   mat[],                                   /* Eingabematrix ...................*/
REAL     x[],                                     /* Eigenvektor .....................*/
REAL *   ew                                       /* Betragsgroesster Eigenwert ......*/
);

int eigen (                                       /* Alle Eigenwerte/Eigenvektoren von Matrizen */
int     vec,                                      /* Schalter fuer Eigenvektoren .....*/
int     ortho,                                    /* orthogonale Hessenbergreduktion? */
int     ev_norm,                                  /* Eigenvektoren normieren? ........*/
int     n,                                        /* Dimension der Matrix ............*/
REAL *  mat[],                                    /* Eingabematrix ...................*/
REAL *  eivec[],                                  /* Eigenvektoren ...................*/
REAL    valre[],                                  /* Realteile der Eigenwerte ........*/
REAL    valim[],                                  /* Imaginaerteile der Eigenwerte ...*/
int     cnt[]                                     /* Iterationzaehler ................*/
);
#endif

/* ------------------------- ENDE u_proto.h ------------------------- */
