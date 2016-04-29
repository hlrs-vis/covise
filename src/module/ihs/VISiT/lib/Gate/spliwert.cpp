/* ------------------------ MODUL spliwert.c ------------------------ */

/***********************************************************************
 *                                                                      *
 * Funktionen zur Auswertung von Polynomsplinefunktionen                *
 * -----------------------------------------------------                *
 *                                                                      *
 * Programmiersprache: ANSI C                                           *
 * Compiler:           Turbo C 2.0                                      *
 * Rechner:            IBM PS/2 70 mit 80387                            *
 * Bemerkung:          Umsetzung einer aequivalenten TP-Unit und eines  *
 *                     aequivalenten QuickBASIC-Moduls                  *
 * Autor:              Elmar Pohl (QuickBASIC)                          *
 * Bearbeiter:         Juergen Dietel, Rechenzentrum der RWTH Aachen    *
 * Datum:              DI 13. 8. 1991                                   *
 *                                                                      *
 ***********************************************************************/

#include <Gate/include/basis.h>                   /* wegen PI, sqr, intervall, FABS, POW,     */
/*       COS, SIN, MACH_EPS, REAL, TWO,     */
/*       THREE, ZERO, FIVE, FOUR, SIX       */
#include <Gate/include/spliwert.h>                /* wegen spwert, pspwert, hmtwert, pmtwert, */
/*       strwert                            */

/* ------------------------------------------------------------------ */
/*.BA*/

/*.BE*/
REAL spwert                                       /* Auswertung eines kubischen Polynomsplines .......*/
/*.BA*/
/*.IX{spwert}*/
/*.BE*/
(
int  n,                                           /* Anzahl der Splinestuecke ...............*/
REAL xwert,                                       /* Auswertungsstelle ......................*/
REAL a[],                                         /* Splinekoeffizienten von (x-x[i])^0 .....*/
REAL b[],                                         /* Splinekoeffizienten von (x-x[i])^1 .....*/
REAL c[],                                         /* Splinekoeffizienten von (x-x[i])^2 .....*/
REAL d[],                                         /* Splinekoeffizienten von (x-x[i])^3 .....*/
REAL x[],                                         /* Stuetzstellen ..........................*/
REAL ausg[]                                       /* 1., 2., 3. Ableitung des Splines .......*/
)                                                 /* Funktionswert des Splines ..............*/
/*.BA*/

/***********************************************************************
 * Funktions- und Ableitungswerte einer kubischen Polynomsplinefunktion *
 * berechnen                                                            *
.BE*)
*                                                                      *
* Eingabeparameter:                                                    *
* =================                                                    *
* n:       Index der letzten Stuetzstelle in x                         *
* xwert:   Stelle, an der Funktions- und Ableitungswerte berechnet     *
*          werden sollen                                               *
* a,b,c,d: [0..n-1]-Felder mit den Splinekoeffizienten                 *
* x:       [0..n]-Feld mit den Stuetzstellen                           *
*                                                                      *
* Ausgabeparameter:                                                    *
 * =================                                                    *
 * ausg: [0..2]-Feld mit den Ableitungswerten.                          *
 *       ausg[0] enthaelt die 1. Ableitung,                             *
 *       ausg[1] enthaelt die 2. Ableitung,                             *
 *       ausg[2] enthaelt die 3. Ableitung,                             *
 *       alle weiteren Ableitungen sind identisch 0.                    *
 *                                                                      *
 * Funktionswert:                                                       *
 * ==============                                                       *
 * Wert des Splines an der Stelle xwert                                 *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, intervall, TWO, THREE                                          *
.BA*)
***********************************************************************/
/*.BE*/

{
   static int i = 0;                              /* Nummer des Stuetzstellenintervalls von xwert */
   REAL       hilf1,                              /* Zwischenspeicher fuer haeufig verwendete     */
      hilf2,                                      /* Ausdruecke bei der Polynomauswertung         */
      hilf3;

   /* -- Im Falle eines wiederholten Aufrufs dieser Funktion muss i -- */
   /* -- durch intervall() nur dann neu bestimmt werden, wenn xwert -- */
   /* -- nicht mehr im selben Intervall liegt wie im vorigen Aufruf.-- */

   if (i>n) i=n;
   if (i<0) i=0;

   if (xwert < x[i] || xwert >= x[i + 1] || i>n)
      i = intervall(n, xwert, x);

   /*if ((xwert <  x[i]) && (i>0))    i = intervall(n, xwert, x);
   if ((xwert >= x[i+1]) && (i<n-1))  i = intervall(n, xwert, x);*/

   /* ------- das Splinepolynom nach dem Hornerschema auswerten ------ */

   xwert -= x[i];
   hilf1 = THREE * d[i];
   hilf2 = TWO   * c[i];
   hilf3 = TWO   * hilf1;
   ausg[0] = (hilf1 * xwert + hilf2) * xwert + b[i];
   ausg[1] = hilf3 * xwert + hilf2;
   ausg[2] = hilf3;

   return ((d[i] * xwert + c[i]) * xwert + b[i]) * xwert + a[i];
}


/* ------------------------------------------------------------------ */

/*.BA*/

/*.BE*/
void pspwert                                      /* Auswertung eines parametr. kub. Polynomsplines ......*/
/*.BA*/
/*.IX{pspwert}*/
/*.BE*/
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
REAL     *xabl,                                   /* 1. Ableitung des Splines .....*/
REAL   *yabl
)
/*.BA*/

/***********************************************************************
 * Funktions- und Ableitungswerte einer parametrischen kubischen        *
 * Splinefunktion berechnen                                             *
.BE*)
*                                                                      *
* Eingabeparameter:                                                    *
* =================                                                    *
* n:            Index des letzten Parameterwertes in t                 *
* twert:        Parameterwert, fuer den Funktions- und Ableitungswerte *
*               berechnet werden sollen                                *
* t:            [0..n]-Feld mit Parameterwerten der Wertepaare         *
*               (X[i],Y[i]), i = 0(1)n                                 *
* ax,bx,cx,dx:\                                                        *
* ay,by,cy,dy:/ [0..n-1]-Felder mit den Splinekoeffizienten            *
 *                                                                      *
 * Ausgabeparameter:                                                    *
 * =================                                                    *
 * sx:   Funktionswert der Splinekomponente SX fuer t = twert           *
 * sy:   Funktionswert der Splinekomponente SY fuer t = twert           *
 * ausp: [0..3,0..1]-Feld mit Funktions- und Ableitungswerten.          *
 *       Es ist ausp[0,0] = sx,   ausp[0,1] = sy                        *
 *       und ausp[i,0] ist die i-te Ableitung der Spline-               *
 *       komponente SX an der Stelle t = twert, i = 1(1)3.              *
 *       (Alle weiteren Ableitungen sind identisch 0.)                  *
 *       Entsprechend enthalten die ausp[i,1] die Ableitungen           *
 *       der Splinekomponente SY.                                       *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * spwert, REAL                                                         *
.BA*)
***********************************************************************/
/*.BE*/

{

   REAL ausg[4];

   *sx = spwert(n, twert, ax, bx, cx, dx, t, ausg);
   *xabl= ausg[0];

   *sy = spwert(n, twert, ay, by, cy, dy, t, ausg);
   *yabl= ausg[0];
}


/* ------------------------------------------------------------------ */

/*.BA*/

/*.BE*/
REAL hmtwert                                      /* Auswertung eines Hermite-Polynomsplines .......*/
/*.BA*/
/*.IX{hmtwert}*/
/*.BE*/
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
)                                                 /* Funktionswert des Splines .............*/
/*.BA*/

/***********************************************************************
 * Funktions- und Ableitungswerte einer Hermite-Polynomsplinefunktion   *
 * berechnen                                                            *
.BE*)
*                                                                      *
* Eingabeparameter:                                                    *
* =================                                                    *
* n:           Index der letzten Stuetzstelle in x                     *
* x0:          Stelle, an der Funktions- und Ableitungswerte berechnet *
*              werden sollen                                           *
* a,b,c,d,e,f: [0..n-1]-Felder mit den Splinekoeffizienten             *
* x:           [0..n]-Feld mit den Stuetzstellen                       *
*                                                                      *
* Ausgabeparameter:                                                    *
 * =================                                                    *
 * ausg: [0..4]-Feld mit den Ableitungswerten.                          *
 *       ausg[0] enthaelt die 1. Ableitung,                             *
 *       ausg[1] enthaelt die 2. Ableitung,                             *
 *       ausg[2] enthaelt die 3. Ableitung,                             *
 *       ausg[3] enthaelt die 4. Ableitung,                             *
 *       ausg[4] enthaelt die 5. Ableitung,                             *
 *       alle weiteren Ableitungen sind identisch 0.                    *
 *                                                                      *
 * Funktionswert:                                                       *
 * ==============                                                       *
 * Wert des Splines an der Stelle x0                                    *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, intervall, TWO, THREE, FIVE, FOUR, SIX                         *
.BA*)
***********************************************************************/
/*.BE*/

{
   int  i;                                        /* Nummer des Intervalls, in dem x0 liegt   */
   REAL B, C, D, E, F;                            /* Splinekoeffizienten fuer das Intervall i */

   i = intervall(n, x0, x);
   x0 -= x[i];
   B = b[i];
   C = c[i];
   D = d[i];
   E = e[i];
   F = f[i];
   ausg[0] = (((FIVE * F * x0 + FOUR * E) * x0 + THREE * D) * x0 +
      TWO * C) * x0 + B;
   ausg[1] = (((REAL)20.0 * F * x0 + (REAL)12.0 * E) * x0 + SIX * D) *
      x0 + TWO * C;
   ausg[2] = ((REAL)60.0 * F * x0 + (REAL)24.0 * E) * x0 + SIX * D;
   ausg[3] = (REAL)120.0 * F * x0 + (REAL)24.0 * E;
   ausg[4] = (REAL)120.0 * F;

   return ((((F * x0 + E) * x0 + D) * x0 + C) * x0 + B) * x0 + a[i];
}


/* ------------------------------------------------------------------ */

/*.BA*/

/*.BE*/
void pmtwert                                      /* Auswertung eines parametr. Hermite-Polynomsplines ...*/
/*.BA*/
/*.IX{pmtwert}*/
/*.BE*/
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
)
/*.BA*/

/***********************************************************************
 * Funktions- und Ableitungswerte einer parametrischen Hermite-Spline-  *
 * funktion 5. Grades berechnen, die von der Funktion parmit()          *
 * berechnet wurde                                                      *
.BE*)
*                                                                      *
* Eingabeparameter:                                                    *
* =================                                                    *
* n:                  Index des letzten Parameterwertes in t           *
* twert:              Parameterwert, fuer den Funktions- und           *
*                     Ableitungswerte berechnet werden sollen          *
* t:                  [0..n]-Feld mit Parameterwerten der Wertepaare   *
*                     (X[i],Y[i]), i = 0(1)n                           *
* ax,bx,cx,dx,ex,fx:\ [0..n-1]-Felder mit den Splinekoeffizienten      *
* ay,by,cy,dy,ey,fy:/                                                  *
 *                                                                      *
 * Ausgabeparameter:                                                    *
 * =================                                                    *
 * sx:   Funktionswert der Splinekomponente SX fuer t = twert           *
 * sy:   Funktionswert der Splinekomponente SY fuer t = twert           *
 * ausp: [0..5,0..1]-Feld mit Funktions- und Ableitungswerten.          *
 *       Es ist ausp[0,0] = sx,   ausp[0,1] = sy, und ausp[i,0] ist die *
 *       i. Ableitung der Splinekomponente SX an der Stelle t = twert,  *
 *       i = 1(1)5. (Alle weiteren Ableitungen sind identisch 0.)       *
 *       Entsprechend enthalten die ausp[i,1] die Ableitungen der       *
 *       Splinekomponente SY.                                           *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * hmtwert, REAL                                                        *
.BA*)
***********************************************************************/
/*.BE*/

{
   REAL ausg[6];
   int  i;

   *sx = hmtwert(n, twert, ax, bx, cx, dx, ex, fx, t, ausg);
   ausp[0][0] = *sx;
   for (i = 1; i < 6; i++)
      ausp[i][0] = ausg[i - 1];
   *sy = hmtwert(n, twert, ay, by, cy, dy, ey, fy, t, ausg);
   ausp[0][1] = *sy;
   for (i = 1; i < 6; i++)
      ausp[i][1] = ausg[i - 1];
}


/* ------------------------------------------------------------------ */

/*.BA*/

/*.BE*/
int strwert                                       /* Auswertung eines transf.-param. kub. Polynomsplines ..*/
/*.BA*/
/*.IX{strwert}*/
/*.BE*/
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
)                                                 /* Fehlercode ...........................*/
/*.BA*/

/***********************************************************************
 * eine transformiert-parametrische kubische Splinefunktion in der      *
 * Darstellung                                                          *
 *   s(phi) = a[i] + b[i](phi-phin[i]) + c[i](phi-phin[i])^2 +          *
 *                                     + d[i](phi-phin[i])^3            *
 * fuer phi aus [phin[i],phin[i+1]], i=0(1)n-1, auswerten.              *
.BE*)
* Berechnet werden der Funktionswert, 1., 2. und 3. Ableitung der      *
* Splinefunktion s(phi), die kartesischen Koordinaten (xk, yk) aus den *
* Polarkoordinaten (phi, s(phi)), die 1. Ableitung und die Kruemmung   *
* der Kurve K an der Stelle phi.                                       *
* Bemerkung: Diese Auswertungsfunktion eignet sich nicht zur Erzeugung *
*            einer Wertetabelle der Funktionswerte s(phi) oder der     *
*            Kurvenpunkte xk, yk.                                      *
*            Soll lediglich die Splinefunktion s(phi) ausgewertet      *
*            werden, sollte die Funktion spwert() benutzt werden.      *
*                                                                      *
 * Eingabeparameter:                                                    *
 * =================                                                    *
 * phi:     Stelle, an der die Splinefunktion ausgewertet werden soll   *
 *          (Winkel im Bogenmass)                                       *
 * n:       Index der letzten Stuetzstelle in phin                      *
 * phin:    [0..n]-Feld mit den Winkeln der Knoten in Polarkoordinaten- *
 *          darstellung                                                 *
 * a,b,c,d: [0..n-1]-Felder mit den Splinekoeffizienten                 *
 * phid:\   Angaben zur Drehung und Verschiebung des Koordinatensystems *
 * px:   >                                                              *
 * py:  /                                                               *
 *                                                                      *
 * Ausgabeparameter:                                                    *
 * =================                                                    *
 * ablei: [0..3]-Feld mit Funktions- und Ableitungswerten.              *
 *        ablei[0] enthaelt den Funktionswert s(phi),              (S)  *
 *        ablei[1] enthaelt die 1. Ableitung s'(phi),              (S1) *
 *        ablei[2] enthaelt die 2. Ableitung s''(phi),             (S2) *
 *        ablei[3] enthaelt die 3. Ableitung s'''(phi),                 *
 *        alle weiteren Ableitungen sind identisch 0.                   *
 * xk:\   kartesische Koordinaten der Kurve an der Stelle phi           *
 * yk:/                                                                 *
 * c1:    1. Ableitung der Kurve an der Stelle phi, berechnet nach:     *
 *                 c1 = (S1 * sin(rho) + S * cos(rho)) /                *
 *                      (S1 * cos(rho) - S * sin(rho))                  *
 *        mit rho = phi + phid                                          *
 * ckr:   Kruemmung der Kurve an der Stelle phi, berechnet nach:        *
 *        ckr = (2 * S1^2 - S * S2 + S^2) / ((S1^2 + S^2) ^ 1.5)        *
 *                                                                      *
 * Funktionswert:                                                       *
 * ==============                                                       *
 * fehler1 + 3 * fehler2.                                               *
 * Dabei ist fehler1 fuer c1 zustaendig und fehler2 fuer ckr.           *
 * fehler1 kann folgende Werte annehmen:                                *
 * 0: kein Fehler                                                       *
 * 1: Der Nenner in der Gleichung von c1 ist Null.                      *
 * 2: Der Betrag des Nenners in der Gleichung von c1 ist groesser als   *
 *    Null, jedoch kleiner oder gleich dem Vierfachen der Maschinen-    *
 *    genauigkeit und daher fuer weitere Berechnungen zu ungenau        *
 * Die gleichen Werte gelten fuer fehler2, nun aber bezueglich ckr.     *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * spwert, REAL, PI, MACH_EPS, sqr, FABS, POW, COS, SIN, TWO, ZERO,     *
 * FOUR                                                                 *
.BA*)
***********************************************************************/
/*.BE*/

{
   int  fehler1,
      fehler2,
      l;
   REAL fmasch,
      phix,
      rho,
      cosa,
      sina,
      hz,
      hn;

   fehler1 = fehler2 = 0;
   fmasch  = FOUR * MACH_EPS;

   if (phi < ZERO)                                /* phi an die Hilfsvariable phix    */
      l = (int)FABS(phi / TWO / PI)               /* zuweisen, wobei phix notfalls so */
         + 1,                                     /* umgerechnet wird, dass phix im   */
         phix = l * TWO * PI - phi;               /* Intervall   [0, 2 * pi]   liegt  */
   else if (phi > TWO * PI)
      l = (int)(phi / TWO * PI),
            phix = phi - l * TWO * PI;
   else
      phix = phi;

   /* --- den Funktionswert und die Ableitungen bei phix berechnen --- */
   ablei[0] = spwert(n, phix, a, b, c, d, phin, ablei + 1);

   rho  = phix + phid;                            /* die Kurvenkoordinaten xk, yk, die */
   cosa = COS(rho);                               /* 1. Ableitung und die Kurven-      */
   sina = SIN(rho);                               /* kruemmung berechnen               */
   *xk  = ablei[0] * cosa + px;
   *yk  = ablei[0] * sina + py;
   hz   = ablei[1] * sina + ablei[0] * cosa;
   hn   = ablei[1] * cosa - ablei[0] * sina;
   if (hn == ZERO)
      fehler1 = 1;
   else
   {
      if (FABS(hn) <= fmasch)
         fehler1 = 2;
      *c1 = hz / hn;
   }
   hz = TWO * sqr(ablei[1]) - ablei[0] * ablei[2] + sqr(ablei[0]);
   hn = POW((sqr(ablei[1]) + sqr(ablei[0])), (REAL)1.5);
   if (hn == ZERO)
      fehler2 = 1;
   else
   {
      if (FABS(hn) <= fmasch)
         fehler2 = 2;
      *ckr = hz / hn;
   }

   return fehler1 + 3 * fehler2;
}


/* ------------------------- ENDE spliwert.c ------------------------ */
