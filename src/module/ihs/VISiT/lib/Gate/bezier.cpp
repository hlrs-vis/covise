/* ------------------------- MODUL bezier.c ------------------------- */

#include <Gate/include/basis.h>                   /* wegen REAL, ZERO, ONE, THREE, sqr, FABS, TWO, */
/*       FOUR, SQRT, MACH_EPS                    */
#include <Gate/include/bezier.h>                  /* wegen kubbez, valbez, bezier, rechvp, rechwp  */
#include <math.h>

int kubbez                                        /* Bezierpunkte einer Bezier-Spline-Kurve berechnen ...*/
(
REAL   *b[],                                      /* Gewichtspunkte ...........*/
REAL   *d[],                                      /* Bezierpunkte .............*/
int    anz_interpol,
double *laenge,                                   /* Bogenlaengen der Stuetzpunkte*/
double *laenge_plus,                              /* Bogenlaenge dimensionslos */
int    m,                                         /* Anzahl der Splinestuecke  */
int    dim                                        /* 2,3 fuer ebene, Raumkurve */
)                                                 /* Fehlercode ...............*/

/***********************************************************************
 * berechnet nach dem kubischen Bezier-Verfahren Bezier-Punkte einer    *
 * Kurve.                                                               *
 *                                                                      *
 * Eingabeparameter:                                                    *
 *                                                                      *
 *    REAL d[][3]           Koordinaten der Gewichtspunkte              *
 *    int  m                Anzahl der Kurvensegmente                   *
 *    int  dim              = 2: ebene Kurve                            *
 *                          = 3: Raumkurve                              *
 *                                                                      *
 * Ausgabeparameter:                                                    *
 *                                                                      *
 *    REAL b[][3]           Koordinaten der Bezier-Punkte               *
 *                                                                      *
 * Funktionswert:                                                       *
 *                                                                      *
 *   Fehlercode. Folgende Werte koennen auftreten:                      *
 *   = 0: alles in Ordnung                                              *
 *   = 1: nicht erlaubte Eingabeparameter:                              *
 *        m < 2  oder  dim < 2  oder  dim > 3                           *
 *                                                                      *
 * benutzte globale Namen:                                              *
 *                                                                      *
 *   REAL, TWO, THREE, FOUR, SIX                                        *
 *                                                                      *
 ***********************************************************************/

{
   int i, k;
   double x1, y1, z1;
   double x2, y2, z2;

   if (m < 2 || dim < 2 || dim > 3)
      return 1;

   for (i = 0; i < dim; i++)
   {
      for (k = 1; k < m; k++)
      {
         b[3*k-2][i] = (TWO*d[k-1][i] +  d[k][i]                ) / THREE;
         b[3*k]  [i] = (d[k-1][i] + FOUR*d[k][i] +     d[k+1][i]) / SIX;
         b[3*k+2][i] = (                 d[k][i] + TWO*d[k+1][i]) / THREE;
      }
      b[  2  ][i]   = (                 d[0][i] + TWO*d[ 1 ][i]) / THREE;
      b[3*m-2][i]   = (TWO*d[m-1][i] +  d[m][i]                ) / THREE;

      b[  0  ][i] = d[0][i];                      /* Randpunkte werden fuer natuerlichen */
      b[ 3*m ][i] = d[m][i];                      /* kubischen Bezier-Spline vorbesetzt  */
   }

   // Erweiterung zur Berechnung der Bogenlaenge

   laenge      [0] = 0;
   laenge_plus [0] = 0;

   valbez(0, 0, m, dim, b, anz_interpol, laenge_plus, &x1, &y1, &z1);

   for (i=1;i<(anz_interpol);i++)
   {
      valbez(((int)0),(double)i/(double)(anz_interpol-1),
         m, dim, b,anz_interpol,laenge_plus,&x2, &y2, &z2);

      laenge[i]=laenge[i-1] + pow((pow(x2-x1,2.)
         + pow(y2-y1,2.)
         + pow(z2-z1,2.)),0.5);
      x1=x2;
      y1=y2;
      z1=z2;

   }

   // Bogenlaengen auf 1. normieren

   for (i=1;i<(anz_interpol);i++)
   {
      laenge_plus[i] = laenge[i] / laenge[anz_interpol-1];
   }

   // Ende sicherheitshalber auf 1. setzen

   laenge_plus[anz_interpol-1]=1.;

   return 0;
}


/* ------------------------------------------------------------------ */

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
)                                                 /* Fehlercode                */

/* ================================================================== */
/*   v a l b e z  berechnet die kartesischen Koordinaten (x,y,z)      */
/*   eines Punktes auf der durch b gegebenen Bezierkurve, die durch   */
/*   t aus [0, 1] parametrisiert ist. t=0 ist der Anfangspunkt, t=1   */
/*   der Endpunkt der Kurve.                                          */
/*  ================================================================  */
/*   Eingabeparameter:                                                */
/*                                                                    */
/*    Name    Typ                Bedeutung                            */
/*   ---------------------------------------------------------------  */
/*    modus   int                Bei 1 wird t Bogenlaenge             */
/*    t       REAL               Kurvenparameter aus [0, 1]           */
/*    m       int                Anzahl der Kurvensegmente            */
/*    dim     int                = 2: ebene Kurve                     */
/*                               = 3: Raumkurve                       */
/*    b       REAL **            Bezierpunkte, Ausgabe von kubbez     */
/*                                                                    */
/*   Ausgabeparameter:                                                */
/*                                                                    */
/*    Name          Typ          Bedeutung                            */
/*   ---------------------------------------------------------------  */
/*    x, y, z       REAL         Kartesische Koordinaten des Punktes  */
/*                                                                    */
/*   Rueckgabewert:                                                   */
/*     = 0 : alles ok, Koordinaten bestimmt                           */
/*     = 1 : m < 2  oder  dim < 2  oder  dim > 3                      */
/*     = 2 : t nicht aus dem Intervall [0, 1]                         */
/*                                                                    */
/*    benutzte globale Namen:                                         */
/*      REAL, ZERO, ONE, sqr, THREE                                   */
/*                                                                    */
/* ================================================================== */

{
   int i;
   int  k3;
   double ueber;
   REAL tt, v;

   // Modifizierung für Bogenlaenge
   if (modus==1)
   {

      for(i=1;laenge_plus[i]<t;i++)  {}

      ueber= (laenge_plus[i] - t) / (laenge_plus[i] - laenge_plus[i-1]);

      t  = (double)(i - ueber) / (double)(anz_interpol-1);

      //printf("Parameter t: %8.4f\n",t);

   }

   if (m < 2 || dim < 2 || dim > 3)
      return 1;

   if (t < ZERO || t > ONE)
      return 2;

   tt = THREE * t * (REAL)m;
   k3 = (int)(tt / THREE) * 3;
   if (k3 == 3 * m)                               /* Splinestueck m???             */
      k3 -= 3;                                    /* gibt's nicht, voriges nehmen! */
   t = (tt - k3) / THREE;
   v = ONE - t;
   *z=0;

   *x   = sqr(v) * (v * b[k3][0]  +  THREE * t * b[k3+1][0]) +
      sqr(t) * (THREE * v * b[k3+2][0] + t * b[k3+3][0]);
   *y   = sqr(v) * (v * b[k3][1]  +  THREE * t * b[k3+1][1]) +
      sqr(t) * (THREE * v * b[k3+2][1] + t * b[k3+3][1]);
   if (dim == 3)
      *z = sqr(v) * (v * b[k3][2]  +  THREE * t * b[k3+1][2]) +
         sqr(t) * (THREE * v * b[k3+2][2] + t * b[k3+3][2]);

   return 0;
}


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
)                                                 /* Fehlercode                */

/* ================================================================== */
/*   v a l b e z  berechnet die kartesischen Koordinaten (x,y,z)      */
/*   eines Punktes auf der durch b gegebenen Bezierkurve, die durch   */
/*   t aus [0, 1] parametrisiert ist. t=0 ist der Anfangspunkt, t=1   */
/*   der Endpunkt der Kurve.                                          */
/*  ================================================================  */
/*   Eingabeparameter:                                                */
/*                                                                    */
/*    Name    Typ                Bedeutung                            */
/*   ---------------------------------------------------------------  */
/*    modus   int                Bei 1 wird t Bogenlaenge             */
/*    t       REAL               Kurvenparameter aus [0, 1]           */
/*    m       int                Anzahl der Kurvensegmente            */
/*    dim     int                = 2: ebene Kurve                     */
/*                               = 3: Raumkurve                       */
/*    b       REAL **            Bezierpunkte, Ausgabe von kubbez     */
/*                                                                    */
/*   Ausgabeparameter:                                                */
/*                                                                    */
/*    Name          Typ          Bedeutung                            */
/*   ---------------------------------------------------------------  */
/*    x, y, z       REAL         Kartesische Koordinaten des Punktes  */
/*                                                                    */
/*   Rueckgabewert:                                                   */
/*     = 0 : alles ok, Koordinaten bestimmt                           */
/*     = 1 : m < 2  oder  dim < 2  oder  dim > 3                      */
/*     = 2 : t nicht aus dem Intervall [0, 1]                         */
/*                                                                    */
/*    benutzte globale Namen:                                         */
/*      REAL, ZERO, ONE, sqr, THREE                                   */
/*                                                                    */
/* ================================================================== */

{
   int i;
   int  k3;
   double ueber;
   REAL tt, v;

   // Modifizierung für Bogenlaenge
   if (modus==1)
   {

      for(i=1;laenge_plus[i]<t;i++)  {}

      ueber= (laenge_plus[i] - t) / (laenge_plus[i] - laenge_plus[i-1]);

      t  = (double)(i - ueber) / (double)(anz_interpol-1);

      //printf("Parameter t: %8.4f\n",t);

   }

   if (m < 2 || dim < 2 || dim > 3)
      return 1;

   if (t < ZERO || t > ONE)
      return 2;

   tt = THREE * t * (REAL)m;
   k3 = (int)(tt / THREE) * 3;
   if (k3 == 3 * m)                               /* Splinestueck m???             */
      k3 -= 3;                                    /* gibt's nicht, voriges nehmen! */
   t = (tt - k3) / THREE;
   v = ONE - t;
   *dz=0;

   *dx   = -2.*(1.-t)   * ( (1.-t)*b[k3][0] + 3.*t*b[k3+1][0]  )  +
      (1.-t)*(1.-t) * (    -1.*b[k3][0] +   3.*b[k3+1][0]  )  +
      2.*t*  (  3.*(1.-t)*b[k3+2][0] + t*b[k3+3][0]  ) +
      t*t*   (        -3.*b[k3+2][0] +   b[k3+3][0]  );
   *dy   = -2.*(1.-t)   * ( (1.-t)*b[k3][1] + 3.*t*b[k3+1][1]  )  +
      (1.-t)*(1.-t) * (    -1.*b[k3][1] +   3.*b[k3+1][1]  )  +
      2.*t*  (  3.*(1.-t)*b[k3+2][1] + t*b[k3+3][1]  ) +
      t*t*   (        -3.*b[k3+2][1] +   b[k3+3][1]  );
   if (dim == 3)
      *dy   = -2.*(1.-t)   * ( (1.-t)*b[k3][2] + 3.*t*b[k3+1][2]  )  +
         (1.-t)*(1.-t) * (    -1.*b[k3][2] +   3.*b[k3+1][2]  )  +
         2.*t*  (  3.*(1.-t)*b[k3+2][2] + t*b[k3+3][2]  ) +
         t*t*   (        -3.*b[k3+2][2] +   b[k3+3][2]  );

   return 0;
}


/* ------------------------------------------------------------------ */

static void b_point (REAL*** b,
REAL*** d,
int     m,
int     n)
/***********************************************************************
 * errechnet die fuer eine Flaechenberechnung nach dem bikubischen      *
 * Bezierverfahren noch benoetigten, unbekannten Bezierpunkte.          *
 *                                                                      *
 * Eingabeparameter:                                                    *
 *                                                                      *
 *   REAL  b [3*m+1][3*n+1][3]  Feld von Zeigern:                       *
 *                              Koordinaten der Bezierpunkte            *
 *                              (Vorgabe gemaess Abb. 12.4);            *
 *   REAL  d [m+1][n+1][3]      Feld von Zeigern:                       *
 *                              Koordinaten der Gewichtspunkte          *
 *   int   m                    Anzahl der Pflaster in 1.Richtung       *
 *   int   n                    Anzahl der Pflaster in 2.Richtung       *
 *                                                                      *
 * Ausgabeparameter:                                                    *
 *                                                                      *
 *   REAL  b [3*m+1][3*n+1][3]  Koordinaten aller Bezierpunkte          *
 ***********************************************************************/
{
   int i, j, k;

   for (k=0; k<3; k++)
   {
      for (i=1; i<=m; i++)
         for (j=1; j<=n; j++)
            b [3*i-2][3*j-2][k] = (4.*d[i-1][j-1][k] + 2.*d[i-1][j][k] +
               2.*d[ i ][j-1][k] +    d[ i ][j][k])/9.;
      for (i=0; i<=m-1; i++)
         for (j=1; j<=n; j++)
            b [3*i+2][3*j-2][k] = (4.*d[i+1][j-1][k] + 2.*d[i][j-1][k] +
               2.*d[i+1][ j ][k] +    d[i][ j ][k])/9.;
      for (i=1; i<=m; i++)
         for (j=0; j<=n-1; j++)
            b [3*i-2][3*j+2][k] = (4.*d[i-1][j+1][k] + 2.*d[i-1][j][k] +
               2.*d[ i ][j+1][k] +    d[ i ][j][k])/9.;
      for (i=0; i<=m-1; i++)
         for (j=0; j<=n-1; j++)
            b [3*i+2][3*j+2][k] = (4.*d[i+1][j+1][k] + 2.*d[i][j+1][k] +
               2.*d[i+1][ j ][k] +    d[i][ j ][k])/9.;
      for (i=1; i<=m; i++)
         for (j=1; j<=n-1; j++)
            b [3*i-2][3*j][k] = (2.*d[i-1][j-1][k] + 8.*d[i-1][ j ][k] +
               d[ i ][j-1][k] + 2.*d[i-1][j+1][k] +
               4.*d[ i ][ j ][k] +    d[ i ][j+1][k])/18.;
      for (i=1; i<=m-1; i++)
         for (j=1; j<=n; j++)
            b [3*i][3*j-2][k] = (2.*d[i-1][j-1][k] + 8.*d[ i ][j-1][k] +
               d[i-1][ j ][k] + 2.*d[i+1][j-1][k] +
               4.*d[ i ][ j ][k] +    d[i+1][ j ][k])/18.;
      for (i=1; i<=m-1; i++)
         for (j=0; j<=n-1; j++)
            b [3*i][3*j+2][k] = (2.*d[i-1][j+1][k] + 8.*d[ i ][j+1][k] +
               d[i-1][ j ][k] + 2.*d[i+1][j+1][k] +
               4.*d[ i ][ j ][k] +    d[i+1][ j ][k])/18.;
      for (i=0; i<=m-1; i++)
         for (j=1; j<=n-1; j++)
            b [3*i+2][3*j][k] = (2.*d[i+1][j-1][k] + 8.*d[i+1][ j ][k] +
               d[ i ][j-1][k] + 2.*d[i+1][j+1][k] +
               4.*d[ i ][ j ][k] +    d[ i ][j+1][k])/18.;
      for (i=1; i<=m-1; i++)
         for (j=1; j<=n-1; j++)
            b [3*i][3*j][k] = (     d[i-1][j-1][k] + 4.*d[ i ][j-1][k] +
               d[i+1][j-1][k] + 4.*d[i-1][ j ][k] +
               16.*d[ i ][ j ][k] + 4.*d[i+1][ j ][k] +
               d[i-1][j+1][k] + 4.*d[ i ][j+1][k] +
               d[i+1][j+1][k]                    )/36.;
   }
}


static void intpol (REAL*   diff,
int     i,
int     j,
REAL*** b,
int     m,
int     n)
/***********************************************************************
 * fuehrt die Aenderungen (an der nach dem bikubischen Bezierverfahren  *
 * errechneten Spline-Flaeche) in den Interpolationsstellen durch.      *
 *                                                                      *
 * Eingabeparameter:                                                    *
 *                                                                      *
 *    REAL  diff [3]             Koordinaten des Differenzvektors, nach *
 *                               dem die Bezierflaeche veraendert wird  *
 *    int   i, j                 kennzeichnen das Pflaster, in dessen   *
 *                               Umgebung die Bezierflaeche veraendert  *
 *                               wird                                   *
 *    int   m                    Anzahl der Pflaster in 1.Richtung      *
 *    int   n                    Anzahl der Pflaster in 2.Richtung      *
 *    REAL  b [3*m+1][3*n+1][3]  Feld von Zeigern:                      *
 *                               Koordinaten der Bezierpunkte           *
 *                                                                      *
 * Ausgabeparameter:                                                    *
 *                                                                      *
 *    REAL  b [3*m+1][3*n+1][3]  Koordinaten der Bezierpunkte           *
 ***********************************************************************/
{
   static REAL gewicht[7][7] =
   {
      { 0.0625, 0.125, 0.25, 0.25, 0.25, 0.125, 0.0625 },
      {  0.125,  0.25,  0.5,  0.5,  0.5,  0.25,  0.125 },
      {   0.25,   0.5,  1.0,  1.0,  1.0,   0.5,   0.25 },
      {   0.25,   0.5,  1.0,  1.0,  1.0,   0.5,   0.25 },
      {   0.25,   0.5,  1.0,  1.0,  1.0,   0.5,   0.25 },
      {  0.125,  0.25,  0.5,  0.5,  0.5,  0.25,  0.125 },
      { 0.0625, 0.125, 0.25, 0.25, 0.25, 0.125, 0.0625 }
   };
   REAL        linker_rand [3][7],                /* gesicherte Randpunkte */
      rechter_rand[3][7],                         /* der Beziermatrix      */
      unterer_rand[3][7],
      oberer_rand [3][7],
      tmp;
   int         k1, k2, l,                         /* Laufvariablen         */
      i3, j3;                                     /* 3*i bzw. 3*j          */

   i3 = 3 * i;
   j3 = 3 * j;

   if (i == 1 || i == m-1 ||                      /* Werden Randpunkte zerstoert? */
      j == 1 || j == n-1)
      for (l = 0; l < 3; l++)                     /* alle 28         */
         for (k1 = -3; k1 <= 3; k1++)             /* moeglicherweise */
                                                  /* betroffenen     */
                     unterer_rand[l][3+k1] = b[i3+k1][0][l],
                                                  /* Randpunkte      */
               oberer_rand [l][3+k1] = b[i3+k1][3*n][l],
                                                  /* sichern         */
               linker_rand [l][3+k1] = b[0]   [j3+k1][l],
               rechter_rand[l][3+k1] = b[3*m] [j3+k1][l];

   for (l = 0; l < 3; l++)                        /* 49 Punkte   */
      for (tmp = diff[l], k1 = -3; k1 <= 3; k1++) /* verschieben */
         for (k2 = -3; k2 <= 3; k2++)
            b[i3+k1][j3+k2][l] += tmp * gewicht[3+k1][3+k2];

   if (i == 1 || i == m-1 ||                      /* Wurden Randpunkte zerstoert? */
      j == 1 || j == n-1)
      for (l = 0; l < 3; l++)                     /* alle 28         */
         for (k1 = -3; k1 <= 3; k1++)             /* moeglicherweise */
                                                  /* zerstoerten     */
                  b[i3+k1][0][l]    = unterer_rand[l][3+k1],
                                                  /* Randpunkte      */
               b[i3+k1][3*n][l]  = oberer_rand [l][3+k1],
                                                  /* restaurieren    */
               b[0]   [j3+k1][l] = linker_rand [l][3+k1],
               b[3*m] [j3+k1][l] = rechter_rand[l][3+k1];
}


int bezier (REAL*** b,
REAL*** d,
int     modified,
int     m,
int     n,
REAL    eps)

/***********************************************************************
 * realisiert das bikubische und das modifizierte bikubische            *
 * Bezierverfahren.                                                     *
 * Dabei werden aus den Eingabedaten Interpolationsstellen fuer eine    *
 * nach dem bikubischen Bezierverfahren zu bestimmende Spline-Flaeche   *
 * berechnet.                                                           *
 * Beim modifizierten bikubischen Bezierverfahren werden die gegebenen  *
 * Interpolationsstellen zunaechst als Gewichtspunkte aufgefasst, zu    *
 * welchen man sich Pseudo-Interpolationsstellen errechnet.             *
 * Diese werden so lange verschoben, bis sie mit den echten             *
 * Interpolationsstellen uebereinstimmen bis auf die Genauigkeit eps.   *
 *                                                                      *
 * Eingabeparameter:                                                    *
 *                                                                      *
 *    REAL  b [3*m+1][3*n+1][3]  Feld von Zeigern:                      *
 *                               Koordinaten der Bezierpunkte.          *
 *                               Diese Werte muessen angegeben sein:    *
 *                                 b [i][j][k] mit k=0(1)2 und          *
 *                                  i=0 (1) 3*m   und j=0,              *
 *                                  i=0           und j=0 (1) 3*n,      *
 *                                  i=0 (1) 3*m   und j=      3*n,      *
 *                                  i=      3*m   und j=0 (1) 3*n;      *
 *                               beim modifizierten Verfahren (modified)*
 *                               muss zusaetzlich angegeben werden:     *
 *                                  i=3 (3) 3*m-3 und j=3 (3) 3*n-3     *
 *    REAL  d [m+1][n+1][3]      Feld von Zeigern:                      *
 *                               modified  = 0: Koordinaten der         *
 *                                              Gewichtspunkte          *
 *                               modified != 0: leer                    *
 *    int   modified             modified  = 0: Bezierverfahren         *
 *                               modified != 0: modifiziertes Verfahren *
 *    int   m                    Anzahl der Pflaster in 1.Richtung      *
 *    int   n                    Anzahl der Pflaster in 2.Richtung      *
 *    REAL  eps                  modified = 1 : Genauigkeitsschranke    *
 *                                              fuer die Interpolation  *
 *                                                                      *
 * Ausgabeparameter:                                                    *
 *                                                                      *
 *  REAL    b [3*m+1][3*n+1][3]  Koordinaten der Bezierpunkte           *
 *                               b [k][i][j]                            *
 *                               mit i=0(1)3*m, j=0(1)3*n, k=0(1)2      *
 *                                                                      *
 *                                                                      *
 * Funktionsrueckgabewert:                                              *
 *                                                                      *
 *  = 0: kein Fehler                                                    *
 *  = 1: m < 2  oder  n < 2                                             *
 *  = 2: eps zu klein (nur modifiziertes Verfahren)                     *
 *                                                                      *
 * benutzte Unterprogramme:  intpol, b_point                            *
 *                                                                      *
 * benutzte Macros:  FABS                                               *
 ***********************************************************************/

{
   int  i, j, l;
   REAL diff [3];

   if (m < 2 || n < 2)
      return 1;

   if (modified)
   {

      int okay = 0;

      if (eps < (REAL)128.0 * MACH_EPS)
         return 2;

      for (l=0; l<3; l++)
         for (i=0; i<=m; i++)
            for (j=0; j<=n; j++)
               d [i][j][l] = b [3*i][3*j][l];
      b_point (b, d, m, n);

      while (!okay)
      {
         for (i=1; i<=m-1; i++)
            for (j=1; j<=n-1; j++)
         {
            for (l=0; l<3; l++)
               diff [l] = d [i][j][l] - b [3*i][3*j][l];
            intpol (diff, i, j, b, m, n);
         }
         for (okay=1,i=1; i<=m-1; i++)
            for (j=1; j<=n-1; j++)
               for (l=0; l<3; l++)
                  if (FABS (d [i][j][l] - b [3*i][3*j][l]) > eps)
                     okay = 0;
      }                                           /* while (!okay) */

   }                                              /* (modified) */

   else                                           /* (not modified) */
      b_point (b, d, m, n);

   return 0;
}


static void rechp (REAL*** b, int m, int n,

REAL vp, REAL wp, REAL* point)
/***********************************************************************
 * berechnet an der Schnittstelle zweier Parameterlinien einer Flaeche  *
 * die Raum-Koordinaten dieses Flaechenpunktes.                         *
 * ==================================================================== *
 *                                                                      *
 *   EINGABEPARAMETER:                                                  *
 *   -----------------                                                  *
 *                                                                      *
 *    Name    Typ/Laenge                Bedeutung                       *
 *   ------------------------------------------------------------------ *
 *    b       REAL  /[3*m+1][3*n+1][3]  Koordinaten der Bezierpunkte    *
 *                                      b ist ein Feld von Zeigern      *
 *    m       int/---                   Anzahl der Pflaster in 1.       *
 *                                      Richtung                        *
 *    n       int/---                   Anzahl der Pflaster in 2.       *
 *                                      Richtung                        *
 *    vp,wp   REAL  /---                definieren die Parameterli-     *
 *                                      nie, an deren Schnittstelle     *
 *                                      ein Punkt der Bezier-Flaeche    *
 *                                      berechnet werden soll           *
 *                                                                      *
 *                                                                      *
 *   AUSGABEPARAMETER:                                                  *
 *   -----------------                                                  *
 *                                                                      *
 *    Name    Typ/Laenge                Bedeutung                       *
 *   ------------------------------------------------------------------ *
 *    point   REAL  /[3]                Koordinaten des berechneten     *
 *                                      Punktes der Bezierflaeche       *
 *                                                                      *
 ***********************************************************************/
{
   int  i, j, k;
   REAL h, h1, h2, h3, h4, h5, h6, h7, h8, v, w, vv, ww;

   vv = vp * (3 * n);                ww = wp * (3 * m);
   i  = (int) (vv / 3.) * 3;         j  = (int) (ww / 3.) * 3;
   if (i >= 3*n) i = 3 * (n-1);      if (j >= 3*m) j = 3 * (m-1);
   v  = (vv - i) / 3.;               w  = (ww - j) / 3.;

   h  = 1 - v;                       h1 =      h * h * h;
   h2 = 3. * h * h * v;
   h3 = 3. * h * v * v;
   h4 =      v * v * v;

   h  = 1 - w;                       h5 =      h * h * h;
   h6 = 3. * h * h * w;
   h7 = 3. * h * w * w;
   h8 =      w * w * w;

   for (k=0; k<=2; k++)
   {
      point [k]  = (b [ j ][ i ][k] * h1 + b [ j ][i+1][k] * h2 +
         b [ j ][i+2][k] * h3 + b [ j ][i+3][k] * h4  ) * h5;
      point [k] += (b [j+1][ i ][k] * h1 + b [j+1][i+1][k] * h2 +
         b [j+1][i+2][k] * h3 + b [j+1][i+3][k] * h4  ) * h6;
   }
   for (k=0; k<=2; k++)
   {
      point [k] += (b [j+2][ i ][k] * h1 + b [j+2][i+1][k] * h2 +
         b [j+2][i+2][k] * h3 + b [j+2][i+3][k] * h4  ) * h7;
      point [k] += (b [j+3][ i ][k] * h1 + b [j+3][i+1][k] * h2 +
         b [j+3][i+2][k] * h3 + b [j+3][i+3][k] * h4  ) * h8;
   }
   return;
}


int rechvp (REAL*** b, int m, int n, REAL vp,
int num, REAL *points[])

/***********************************************************************
 * berechnet die Raum-Koordinaten der num Flaechenpunkte, die auf der   *
 * durch vp definierten Parameterlinie liegen                           *
.BE*)
* (vp=0, wenn i=0; vp=1, wenn i=3*n; d.h. vp legt einen Masstab        *
* an die (m x n)-Pflaster in Zaehlrichtung n).                         *
* ==================================================================== *
*                                                                      *
*   EINGABEPARAMETER:                                                  *
*   -----------------                                                  *
*                                                                      *
*    Name    Typ/Laenge                Bedeutung                       *
*   ------------------------------------------------------------------ *
*    b       REAL  /[3*m+1][3*n+1][3]  Koordinaten der Bezierpunkte    *
 *                                      b ist ein Feld von Zeigern      *
 *    m       int/---                   Anzahl der Pflaster in 1.       *
 *                                      Richtung                        *
 *    n       int/---                   Anzahl der Pflaster in 2.       *
 *                                      Richtung                        *
 *    vp      REAL  /---                definiert die Parameterli-      *
 *                                      nie, auf der Zwischenpunkte     *
 *                                      der Bezier-Flaeche berechnet    *
 *                                      werden sollen                   *
 *    num     int/---                   Anzahl der zu berechnenden      *
 *                                      Punkte                          *
 *                                                                      *
 *                                                                      *
 *   AUSGABEPARAMETER:                                                  *
 *   -----------------                                                  *
 *                                                                      *
 *    Name    Typ/Laenge                Bedeutung                       *
 *   ------------------------------------------------------------------ *
 *    points  REAL  /[num][3]           Koordinaten der berechneten     *
 *                                      Zwischenpunkte                  *
 *                                                                      *
 *                                                                      *
 *   FUNKTIONSWERT:                                                     *
 *   --------------                                                     *
 *                                                                      *
 *   = 0: in Ordnung                                                    *
 *   = 1: m < 2  oder  n < 2                                            *
 *   = 2: vp nicht aus [0,1]                                            *
 *   = 3: num < 2                                                       *
 *                                                                      *
 * ==================================================================== *
 *                                                                      *
 *   benutzte Unterprogramme:  rechp                                    *
 *   ------------------------                                           *
 *                                                                      *
 ***********************************************************************/
{
   int    i, k;
   REAL   step, h, point [3];

   if (m < 2 || n < 2)
      return 1;

   if (vp < ZERO || vp > ONE)
      return 2;

   if (num < 2)
      return 3;

   h = (REAL)(num - 1);
   for (i = 0; i <= num-1; i++)
   {
      step = i / h;
      rechp (b, m, n, vp, step, point);
      for (k = 0; k <= 2; k++)
         points [i][k] = point [k];
   }
   return 0;
}


int rechwp (REAL*** b, int m, int n, REAL wp,
int num, REAL *points[])

/***********************************************************************
 * berechnet die Raum-Koordinaten der num Flaechenpunkte, die auf der   *
 * durch wp definierten Paramterlinie liegen                            *
 * (wp=0, wenn j=0;  wp=1, wenn j=3*m; d.h. wp legt einen Masstab       *
 * an die (m x n)-Pflaster in Zaehlrichtung m).                         *
 * ==================================================================== *
 *                                                                      *
 *   EINGABEPARAMETER:                                                  *
 *   -----------------                                                  *
 *                                                                      *
 *    Name    Typ/Laenge                Bedeutung                       *
 *   ------------------------------------------------------------------ *
 *    b       REAL  /[3*m+1][3*n+1][3]  Koordinaten der Bezierpunkte    *
 *                                      b ist ein Feld von Zeigern      *
 *    m       int/---                   Anzahl der Pflaster in 1.       *
 *                                      Richtung                        *
 *    n       int/---                   Anzahl der Pflaster in 2.       *
 *                                      Richtung                        *
 *    wp      REAL  /---                definiert die Parameterli-      *
 *                                      nie, auf der Zwischenpunkte     *
 *                                      der Bezier-Flaeche berechnet    *
 *                                      werden sollen                   *
 *    num     int/---                   Anzahl der zu berechnenden      *
 *                                      Punkte                          *
 *                                                                      *
 *   AUSGABEPARAMETER:                                                  *
 *   -----------------                                                  *
 *                                                                      *
 *    Name    Typ/Laenge                Bedeutung                       *
 *   ------------------------------------------------------------------ *
 *    points  REAL  /[num][3]           Koordinaten der berechneten     *
 *                                      Zwischenpunkte                  *
 *                                                                      *
 *                                                                      *
 *   FUNKTIONSWERT:                                                     *
 *   --------------                                                     *
 *                                                                      *
 *   = 0: in Ordnung                                                    *
 *   = 1: m < 2  oder  n < 2                                            *
 *   = 2: wp nicht aus [0,1]                                            *
 *   = 3: num < 2                                                       *
 *                                                                      *
 * ==================================================================== *
 *                                                                      *
 *   benutzte Unterprogramme:  rechp                                    *
 *   ------------------------                                           *
 *                                                                      *
 ***********************************************************************/
{
   int i, k;
   REAL   step, h, point [3];

   if (m < 2 || n < 2)
      return 1;

   if (wp < ZERO || wp > ONE)
      return 2;

   if (num < 2)
      return 3;

   h = (REAL  ) (num - 1);
   for (i=0; i<=num-1; i++)
   {
      step = (REAL  ) (i) / h;
      rechp (b, m, n, step, wp, point);
      for (k=0; k<=2; k++)
         points [i][k] = point [k];
   }
   return 0;
}


/* -------------------------- ENDE bezier.c ------------------------- */
