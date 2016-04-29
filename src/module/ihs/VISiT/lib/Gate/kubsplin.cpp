/* ------------------------ MODUL kubsplin.c ------------------------ */

/***********************************************************************
 *                                                                      *
 * Funktionen zur Berechnung interpolierender kubischer Polynomsplines  *
 * -------------------------------------------------------------------  *
 *                                                                      *
 * Programmiersprache: ANSI C                                           *
 * Compiler:           Turbo C 2.0                                      *
 * Rechner:            IBM PS/2 70 mit 80387                            *
 * Bemerkung:          Umsetzung eines aequivalenten QB-Moduls          *
 * Autoren:            Elmar Pohl (QuickBASIC), Dorothee Seesing (C)    *
 * Bearbeiter:         Juergen Dietel, Rechenzentrum der RWTH Aachen    *
 * Datum:              DI 2. 2. 1993                                    *
 *                                                                      *
 ***********************************************************************/

#include <Gate/include/basis.h>                   /* wegen sign, MAXROOT, sqr, NULL, SQRT, ONE, */
/*       FABS, PI, REAL, ACOS, ZERO, THREE,   */
/*       HALF, TWO                            */
#include <Gate/include/vmblock.h>                 /* wegen vmalloc, vmcomplete, vmfree, vminit, */
/*       VEKTOR                               */
#include <Gate/include/u_proto.h>                 /* wegen trdiag, tzdiag                       */
#include <Gate/include/kubsplin.h>                /* wegen spline, parspl, spltrans             */
/*.BA*/
/*.FE{C 10.1.2}
     {Berechnung der nichtparametrischen kubischen Splines}
     {Berechnung der nichtparametrischen kubischen Splines}*/

/*.BE*/
/* ------------------------------------------------------------------ */
/*.BA*/

/*.BE*/
int spline                                        /* nichtparametrischer kubischer Polynomspline .......*/
/*.BA*/
/*.IX{spline}*/
/*.BE*/
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
)                                                 /* Fehlercode ..........................*/
/*.BA*/

/***********************************************************************
 * zu den vorgegebenen Wertepaaren                                      *
 *                  (x[i], y[i]), i = 0(1)m-1                           *
 * die Koeffizienten eines nichtparametrischen interpolierenden         *
 * kubischen Polynomplines berechnen.                                   *
.BE*)
* Die Art der Randbedingung wird durch den Parameter marg_cond         *
* festgelegt. Die x[i] muessen streng monoton wachsen.                 *
* Bei wiederholten Aufrufen mit gleichen Stuetzstellen, aber verschie- *
* denen Stuetzwerten besteht die Moeglichkeit, die erneute Aufstellung *
* und Zerlegung der Matrix des Gleichungssystems zu vermeiden, indem   *
* man den Parameter save von Null verschieden waehlt und so die Be-    *
* schreibung der Zerlegungsmatrizen fuer den naechsten Aufruf rettet.  *
* Wichtig: Damit der Speicher fuer die Hilfsfelder wieder frei wird    *
* -------- und bei weiteren Aufrufen nicht mit falschen Zerlegungs-    *
*          matrizen gearbeitet wird, muss der letzte Aufruf einer      *
 *          zusammengehoerigen Aufruffolge mit save = 0 statt mit       *
 *          save = 1 ausgefuehrt werden!                                *
 *                                                                      *
 * Eingabeparameter:                                                    *
 * =================                                                    *
 * m:          Anzahl der Stuetzstellen (mindestens 3)                  *
 * x:          [0..m-1]-Vektor mit den x-Koordinaten der Wertepaare     *
 *             (wird nicht benoetigt, falls der vorige Aufruf mit       *
 *              save != 0 stattfand)                                    *
 * y:          [0..m-1]-Vektor mit den y-Koordinaten der Wertepaare     *
 * marg_cond:  = 0: not-a-knot-Bedingung (=> marg_0, marg_n ohne        *
 *                                           Bedeutung)                 *
 *             = 1: marg_0, marg_n sind 1. Ableitungen.                 *
 *             = 2: marg_0, marg_n sind 2. Ableitungen.                 *
 *                  (Fuer marg_0 = marg_n = 0 erhaelt man einen         *
 *                  natuerlichen Spline.)                               *
 *             = 3: marg_0, marg_n sind 3. Ableitungen.                 *
 *             = 4: periodischer Spline (=> marg_0, marg_n ohne         *
 *                                          Bedeutung)                  *
 * marg_0:     Randbedingung in x[0]                                    *
 * marg_n:     Randbedingung in x[m-1]                                  *
 * save:       Flagge, die anzeigt, ob der Speicher fuer die Hilfsfel-  *
 *             der mit den Zerlegungsmatrizen fuer den naechsten Aufruf *
 *             aufbewahrt werden soll. Im Normalfall ist save = 0 zu    *
 *             setzen. Wenn man mehrere Splinefunktionen mit denselben  *
 *             Stuetzstellen x[i], aber anderen y[i] berechnen will     *
 *             (z. B. bei parametrischen Splines), kann man ab dem      *
 *             zweiten Aufruf Rechenzeit sparen, indem man beim ersten  *
 *             Aufruf save = 1 setzt. Dann wird naemlich die neuerliche *
 *             Aufstellung und Zerlegung der Tridiagonalmatrix umgangen *
 *             (=> ca. 4*m Punktoperationen weniger).                   *
 *             Im letzten Aufruf muss man save = 0 waehlen, damit der   *
 *             von den Hilfsfeldern beanspruchte dynamische Speicher    *
 *             fuer andere Programmteile wieder verfuegbar wird.        *
 *                                                                      *
 * Ausgabeparameter:                                                    *
 * =================                                                    *
 * b: \  [0..m-2]-Vektoren mit den Splinekoeffizienten nach dem Ansatz  *
 * c:  >     s(x)  =  a[i] + b[i] * (x - x[i]) + c[i] * (x - x[i]) ^ 2  *
 * d: /                    + d[i] * (x - x[i]) ^ 3.                     *
 *       a entspricht y,                                                *
 *       c hat (wie a) noch ein zusaetzliches Element c[m-1].           *
 *                                                                      *
 * Funktionswert:                                                       *
 * ==============                                                       *
 * =  0: kein Fehler                                                    *
 * = -i: Monotoniefehler: x[i-1] >= x[i]                                *
 * =  1: falscher Wert fuer marg_cond                                   *
 * =  2: m < 3                                                          *
 * =  3: nicht genuegend Heapspeicher fuer die Hilfsfelder              *
 * =  4: marg_cond = 4: Eingabedaten nichtperiodisch                    *
 * >  4: Fehler in trdiag() oder tzdiag()                               *
 * Im Fehlerfall sind die Werte der Ausgabeparameter unbestimmt, und    *
 * der Speicher fuer die Hilfsfelder wird freigegeben.                  *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, vminit, vmalloc, vmcomplete, vmfree, VEKTOR, trdiag, tzdiag,   *
 * NULL, ZERO, THREE, HALF, TWO                                         *
.BA*)
***********************************************************************/
/*.BE*/

{

#define ciao(fehler)          /* dafuer sorgen, dass vor dem Beenden */\
{ \
/* von spline() aufgeraeumt wird       */\
vmfree(vmblock); /* Speicherplatz fuer die Hilfsfelder freigeben */\
vmblock = NULL;  /* und dies auch anzeigen                       */\
return fehler;   /* den Fehlercode an den Aufrufer weiterreichen */\
}

   static
      void *vmblock = NULL;                       /* Liste der dynamisch vereinbarten Vek- */
   /* toren. Der Wert NULL zeigt an, dass   */
   /* noch keine Hilfsvektoren aus eventu-  */
   /* ellen frueheren Aufrufen existieren,  */
   /* dass dies also der erste Aufruf einer */
   /* zusammengehoerenden Folge mit glei-   */
   /* chen Stuetzstellen ist.               */
   static
      REAL *h,                                    /* [0..m-2]-Vektor mit den Laengen der Stuetz-   */
   /* stellenintervalle                             */
      *lower,                                     /* [0..m-2]-Vektor mit der unteren Nebendiago-   */
   /* nale der Matrix, spaeter Zerlegungsmatrix     */
   /* von trdiag() bzw. tzdiag()                    */
      *diag,                                      /* [0..m-2]-Vektor mit der Hauptdiagonale der    */
   /* Matrix, spaeter Zerlegungsmatrix von          */
   /* trdiag() bzw. tzdiag()                        */
      *upper,                                     /* [0..m-2]-Vektor mit der oberen Nebendiago-    */
   /* nale der Matrix, spaeter Zerlegungsmatrix     */
   /* von trdiag() bzw. tzdiag()                    */
      *lowrow,                                    /* [0..m-4]-Vektor mit der unteren Zeile der     */
   /* Matrix, spaeter Zerlegungsmatrix von tzdiag() */
      *ricol;                                     /* [0..m-4]-Vektor mit der rechten Spalte der    */
   /* Matrix, spaeter Zerlegungsmatrix von tzdiag() */
   int  n,                                        /* m - 1, Index der letzten Stuetzstelle */
      fehler,                                     /* Fehlercode von trdiag() bzw. tzdiag() */
      i,                                          /* Laufvariable                          */
      erster_aufruf;                              /* Flagge, die anzeigt, dass gerade der  */
   /* erste Aufruf einer Folge stattfindet  */

   n = m - 1;

   if (n < 2)                                     /* zu kleinen Wert fuer n abfangen */
      ciao(2);

   if (marg_cond < 0 || marg_cond > 4)            /* falsches marg_cond abfangen */
      ciao(1);

   if (marg_cond == 4)                            /* periodischer Spline?       */
      if (y[n] != y[0])                           /* Periodizitaet ueberpruefen */
         ciao(4);

   /* 1. Aufruf: Speicher fuer die Hilfsfelder anfordern: 4 [0..n-1]-  */
   /* Vektoren (im periodischen Fall noch 2 [0..n-3]-Vektoren)         */

   if (vmblock == NULL)                           /* erster Aufruf einer Folge? */
   {
      erster_aufruf = 1;
#define MYALLOC(l)  (REAL *)vmalloc(vmblock, VEKTOR, (l), 0)
      vmblock = vminit();                         /* Speicherblock initialisieren */
      h     = MYALLOC(n);                         /* Speicher fuer die       */
      lower = MYALLOC(n);                         /* Hilfsvektoren anfordern */
      diag  = MYALLOC(n);
      upper = MYALLOC(n);
      if (marg_cond == 4)                         /* periodischer Spline mit  */
         if (n > 2)                               /* genuegend Stuetzstellen? */
            lowrow = MYALLOC(n - 2),              /* auch die zusaetzlichen   */
               ricol  = MYALLOC(n - 2);           /* Vektoren versorgen       */
#undef MYALLOC
   }
   else
      erster_aufruf = 0;
   if (! vmcomplete(vmblock))                     /* Ging eine der Speicheranforderungen */
      ciao(3);                                    /* fuer den Block schief?              */

   if (erster_aufruf)
      for (i = 0; i < n; i++)                     /* Schrittweiten berechnen und dabei die */
   {                                              /* Stuetzstellen auf Monotonie pruefen   */
      h[i] = x[i + 1] - x[i];                     /* Schrittweiten berechnen */
      if (h[i] <= ZERO)                           /* Stuetzstellen nicht monoton wachsend? */
         ciao(-(i + 1));
   }

   for (i = 0; i < n - 1; i++)                    /* das Gleichungssystem aufstellen */
   {
                                                  /* rechte Seite */
      c[i] = THREE * ((y[i + 2] - y[i + 1]) / h[i + 1]
         - (y[i + 1] - y[i])     / h[i]);
      if (erster_aufruf)
         diag[i] = TWO * (h[i] + h[i + 1]),       /* Hauptdiagonale   */
            lower[i + 1] = upper[i] = h[i + 1];   /* untere und obere */
   }                                              /* Nebendiagonale   */

   switch (marg_cond)                             /* je nach Randbedingung einige Koeffizienten */
   {                                              /* des Gleichungssystems korrigieren          */
      case 0:                                     /* not-a-knot-Bedingung?              */
         if (n == 2)                              /* nur drei Stuetzstellen?      */
         {                                        /* Da in diesem Fall das Gleichungssystem  */
            /* unterbestimmt ist, wird nur ein Polynom */
            /* 2. Grades berechnet.                    */
            c[0] /= THREE;                        /* rechte Seite    */
            if (erster_aufruf)
               diag[0] *= HALF;                   /* auch die Matrix */
         }
         else                                     /* mehr als drei Stuetzstellen? */
         {
                                                  /* rechte */
            c[0]     *= h[1]     / (h[0]     + h[1]);
                                                  /* Seite  */
            c[n - 2] *= h[n - 2] / (h[n - 1] + h[n - 2]);
            if (erster_aufruf)
               diag[0]      -= h[0],              /* auch die */
                  diag[n - 2]  -= h[n - 1],       /* Matrix   */
                  upper[0]     -= h[0],
                  lower[n - 2] -= h[n - 1];
         }
         break;

      case 1:                                     /* erste Randableitungen vorgegeben?  */
         c[0]     -= (REAL)1.5 * ((y[1] - y[0]) / h[0] - marg_0);
         c[n - 2] -= (REAL)1.5 * (marg_n - (y[n] - y[n - 1]) / h[n - 1]);
         if (erster_aufruf)
            diag[0]     -= HALF * h[0],           /* auch die Matrix */
               diag[n - 2] -= HALF * h[n - 1];    /* vorbesetzen     */
         break;

      case 2:                                     /* zweite Randableitungen vorgegeben? */
         c[0]     -= h[0]     * HALF * marg_0;
         c[n - 2] -= h[n - 1] * HALF * marg_n;
         break;

      case 3:                                     /* dritte Randableitungen vorgegeben? */
         c[0]        += HALF * marg_0 * h[0]     * h[0];
         c[n - 2]    -= HALF * marg_n * h[n - 1] * h[n - 1];
         if (erster_aufruf)
            diag[0]     += h[0],                  /* auch die Matrix */
               diag[n - 2] += h[n - 1];           /* vorbesetzen     */
         break;

      case 4:                                     /* periodischer Spline?               */
         c[n - 1] = THREE * ((y[1] - y[0])     / h[0] -
            (y[n] - y[n - 1]) / h[n - 1]);
         if (erster_aufruf)
            if (n > 2)
               diag[n - 1]  = TWO * (h[0] + h[n - 1]),
                  ricol[0] = lowrow[0] = h[0];
   }

   switch (n)                                     /* das Gleichungssystem loesen und damit  */
   {                                              /* die Splinekoeffizienten c[i] berechnen */
      case 2:                                     /* nur drei Stuetzstellen =>    */
         /* => Loesung direkt berechnen  */
         if (marg_cond == 4)                      /* periodischer Spline?         */
            c[1] = THREE * (y[0] - y[1]) / (x[2] - x[1]) / (x[1] - x[0]),
               c[2] = - c[1];
         else
            c[1] = c[0] / diag[0];
         break;

      default:                                    /* mehr als drei Stuetzstellen? */
         if (marg_cond == 4)                      /* periodischer Spline?         */
            fehler = tzdiag(n, lower, diag, upper, lowrow,
               ricol, c, !erster_aufruf);
         else                                     /* nichtperiodischer Spline? */
            fehler = trdiag(n - 1, lower, diag, upper, c, !erster_aufruf);

         if (fehler != 0)                         /* Fehler in tzdiag() oder in trdiag()? */
            ciao(fehler + 4);
         for (i = n; i != 0; i--)                 /* die Elemente des Loesungsvektors   */
            c[i] = c[i - 1];                      /* eine Position nach rechts schieben */
   }

   switch (marg_cond)                             /* in Abhaengigkeit von der Randbedingung den */
   {                                              /* ersten und letzten Wert von c korrigieren  */
      case 0:                                     /* not-a-knot-Bedingung?              */
         if (n == 2)                              /* nur drei Stuetzstellen?      */
            c[0] = c[2] = c[1];
         else                                     /* mehr als drei Stuetzstellen? */
            c[0] = c[1] + h[0] * (c[1] - c[2]) / h[1],
               c[n] = c[n - 1] + h[n - 1] *
               (c[n - 1] - c[n - 2]) / h[n - 2];
         break;

      case 1:                                     /* erste Randableitungen vorgegeben?  */
         c[0] =  (REAL)1.5 * ((y[1] - y[0]) / h[0] - marg_0);
         c[0] = (c[0] - c[1] * h[0] * HALF) / h[0];
         c[n] = (REAL)-1.5 * ((y[n] - y[n - 1]) / h[n - 1] - marg_n);
         c[n] = (c[n] - c[n - 1] * h[n - 1] * HALF) / h[n - 1];
         break;

      case 2:                                     /* zweite Randableitungen vorgegeben? */
         c[0] = marg_0 * HALF;
         c[n] = marg_n * HALF;
         break;

      case 3:                                     /* dritte Randableitungen vorgegeben? */
         c[0] = c[1]     - marg_0 * HALF * h[0];
         c[n] = c[n - 1] + marg_n * HALF * h[n - 1];
         break;

      case 4:                                     /* periodischer Spline?               */
         c[0] = c[n];

   }

   for (i = 0; i < n; i++)                        /* die restlichen      */
      b[i] = (y[i + 1] - y[i]) / h[i] - h[i] *    /* Splinekoeffizienten */
         (c[i + 1] + TWO * c[i]) / THREE,         /* b[i] und d[i]       */
         d[i] = (c[i + 1] - c[i]) / (THREE * h[i]);/* berechnen           */

   if (!save)                                     /* Hilfsfelder nicht aufbewahren */
      ciao(0);                                    /* (letzter Aufruf einer Folge)? */

   return 0;
#undef ciao
}


/*.BA*/

/*.FE{C 10.1.3}
     {Berechnung der parametrischen kubischen Splines}
     {Berechnung der parametrischen kubischen Splines}*/

/*.BE*/
/* ------------------------------------------------------------------ */
/*.BA*/

/*.BE*/
int parspl                                        /* parametrischer kubischer Polynomspline .......*/
/*.BA*/
/*.IX{parspl}*/
/*.BE*/
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
)                                                 /* Fehlercode ..........................*/
/*.BA*/

/***********************************************************************
 * zu den Wertepaaren                                                   *
 *                      (x[i], y[i]), i = 0(1)m-1,                      *
 * die Koeffizienten eines interpolierenden parametrischen kubischen    *
 * Splines berechnen.                                                   *
.BE*)
* Dabei kann die Art der Randbedingung durch den Parameter marg_cond   *
* vorgegeben werden. Die Parameterstuetzstellen  t  koennen entweder   *
* angegeben oder berechnet lassen werden.                              *
*                                                                      *
* Eingabeparameter:                                                    *
* =================                                                    *
* m:         Anzahl der Wertepaare (mindestens 3)                      *
* x: \       [0..m-1]-Vektoren mit den                                 *
* y: /       Wertepaaren                                               *
* marg_cond: Art der Randbedingung:                                    *
 *            = 0: not-a-knot-Bedingung                                 *
 *            = 1: Vorgabe der ersten Ableitung nach dem Parameter t,   *
 *                 und zwar                                             *
 *                 .                      .                             *
 *                 sx(t[0]) in marg_0[0], sy(t[0]) in marg_0[1]),       *
 *                 .                        .                           *
 *                 sx(t[m-1]) in marg_n[0], sy(t[m-1]) in marg_n[1])    *
 *            = 2: Vorgabe der zweiten Ableitung (Man erhaelt einen     *
 *                 natuerlichen Spline bei Vorgabe von marg_cond = 2    *
 *                 und Randbedingungen = 0.), und zwar                  *
 *                 ..                     ..                            *
 *                 sx(t[0]) in marg_0[0], sy(t[0]) in marg_0[1]),       *
 *                 ..                       ..                          *
 *                 sx(t[m-1]) in marg_n[0], sy(t[m-1]) in marg_n[1])    *
 *            = 3: periodischer Spline                                  *
 *            = 4: Vorgabe der Randableitungen dy/dx, und zwar          *
 *                 y'(x[0]) in marg_0[0] und y'(x[m-1]) in marg_n[0].   *
 *                 (marg_0[1] und marg_n[1] bleiben hier unbenutzt.)    *
 * marg_0:    [0..1]-Vektor mit den Randbedingungen in t[0] fuer        *
 *            marg_cond = 1, 2, 4 (ohne Bedeutung fuer marg_cond = 0    *
 *            und marg_cond = 3)                                        *
 * marg_n:    [0..1]-Vektor mit den Randbedingungen in t[m-1]           *
 * cond_t:    Vorgabe der Kurvenparameter t[i]:                         *
 *            =  0: Die Parameterwerte t[i] werden hier berechnet.      *
 *            != 0: Der Benutzer gibt die Werte selber vor.             *
 * t          [0..m-1]-Vektor mit den streng monoton steigenden         *
 *            Parameterstuetzstellen fuer cond_t != 0                   *
 *                                                                      *
 * Ausgabeparameter:                                                    *
 * =================                                                    *
 * t:     [0..m-1]-Vektor mit den Parameterwerten der Punktepaare fuer  *
 *        cond_t = 0                                                    *
 * bx: \  [0..m-2]-Vektoren mit den Splinekoeffizienten nach dem Ansatz *
 * cx:  >     sx(t)  =  ax[i] + bx[i] * (t - t[i]) +  cx[i]             *
 * dx: /                * (t - t[i]) ^ 2  + dx[i] * (t - t[i]) ^ 3.     *
 *        ax entspricht x,                                              *
 *        cx hat (wie ax) noch ein zusaetzliches Element cx[m-1].       *
 * by: \  [0..m-2]-Vektoren mit den Splinekoeffizienten nach dem Ansatz *
 * cy:  >     sy(t)  =  ay[i] + by[i] * (t - t[i]) +  cy[i]             *
 * dy: /                * (t - t[i]) ^ 2  + dy[i] * (t - t[i]) ^ 3.     *
 *        ay entspricht y,                                              *
 *        cy hat (wie ay) noch ein zusaetzliches Element cx[m-1].       *
 *                                                                      *
 * Funktionswert:                                                       *
 * ==============                                                       *
 * =  0: kein Fehler                                                    *
 * = -i: Zwei aufeinanderfolgende Punkte sind gleich:                   *
 *       (x[i-1], y[i-1]) = (x[i], y[i]).                               *
 * =  1: m < 3                                                          *
 * =  2: falscher Wert fuer die Art der Randbedingung in marg_cond      *
 * =  3: periodischer Spline: x[0] != x[m-1]                            *
 * =  4: periodischer Spline: y[0] != y[m-1]                            *
 * =  5: Die t[i] sind nicht streng monoton wachsend.                   *
 * >  5: Fehler in spline()                                             *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * spline, REAL, sign, MAXROOT, sqr, FABS, SQRT, ONE, ZERO              *
.BA*)
***********************************************************************/
/*.BE*/

{
   int  i,                                        /* Laufvariable                              */
      n,                                          /* m - 1, Index des letzten Stuetzpunkts     */
      mess = 0,                                   /* Art der Randbedingung fuer spline()       */
      fehler;                                     /* Fehlercode von spline() bzw. perspl()     */
   REAL deltx,                                    /* Hilfsvariable zur Berechnung der t[i]     */
      delty,                                      /* Hilfsvariable zur Berechnung der t[i]     */
      delt,                                       /* Hilfsvariable zur Berechnung der t[i]     */
      alfx = ZERO,                                /* linke Randbedingung von sx fuer spline()  */
      betx = ZERO,                                /* rechte Randbedingung von sx fuer spline() */
      alfy = ZERO,                                /* linke Randbedingung von sy fuer spline()  */
      bety = ZERO;                                /* rechte Randbedingung von sy fuer spline() */

   n = m - 1;

   if (n < 2)                                     /* weniger als drei Stuetzpunkte? */
      return 1;
   if (marg_cond < 0 || marg_cond > 4)            /* falsches marg_cond? */
      return 2;

   /* Falls t nicht vorgegeben wurde, werden die Werte nun berechnet:  */
   /* Es wird die chordale Parametrisierung angewandt, was bedeutet,   */
   /* dass als Abstand zwischen aufeinanderfolgenden Parameterwerten   */
   /* t[i] und t[i+1] die Laenge der Kurvensehne zwischen den beiden   */
   /* zugehoerigen Stuetzpunkten (x[i], y[i]) und (x[i+1], y[i+1]))    */
   /* gewaehlt wird. Dadurch gewinnt man mit geringem Aufwand grob     */
   /* angenaehert eine Parametrisierung nach der Bogenlaenge.          */

   if (cond_t == 0)                               /* Parameterwerte noch nicht vorhanden? */
      for (t[0] = ZERO, i = 1; i <= n; i++)
   {
      deltx = x[i] - x[i - 1];
      delty = y[i] - y[i - 1];
      delt  = deltx * deltx + delty * delty;
      if (delt <= ZERO)
         return -i;
      t[i] = t[i - 1] + SQRT(delt);
   }

   switch (marg_cond)                             /* je nach Art der Randbedingung die Eingabe-  */
   {                                              /* parameter mess, alfx, betx, alfy, bety fuer */
      /* spline() vorbesetzen                        */
      case 0:                                     /* not-a-knot-Spline? */
         mess = 0;
         break;
      case 1:                                     /* 1. Randableitungen nach t vorgegeben? */
      case 2:                                     /* 2. Randableitungen nach t vorgegeben? */
         mess = marg_cond;
         alfx = marg_0[0];
         alfy = marg_0[1];
         betx = marg_n[0];
         bety = marg_n[1];
         break;
      case 3:                                     /* periodischer Spline? */
         mess = 4;
         if (x[n] != x[0])                        /* ungeeignete Stuetzpunkte fuer */
            return 3;
         if (y[n] != y[0])                        /* einen periodischen Spline?    */
            return 4;
         alfx = betx = ZERO;                      /* vorsichtshalber wegen IBM C Set/2 1.0 */
         alfy = bety = ZERO;
         break;
      case 4:                                     /* 1. Randableitungen dy/dx vorgegeben? */
         mess = 1;                                /* fuer spline(): 1. Ableitungen vorgegeben */
         if (FABS(marg_0[0]) >= MAXROOT)          /* Ableitung zu gross? */
            alfx = ZERO,                          /* senkrechte Tangente */
               alfy = sign(ONE, y[1] - y[0]);
         else
            alfx = sign(SQRT(ONE / (ONE + sqr(marg_0[0]))), x[1] - x[0]),
               alfy = alfx * marg_0[0];
         if (FABS(marg_n[0]) >= MAXROOT)
            betx = ZERO,
               bety = sign(ONE, y[n] - y[n - 1]);
         else
            betx = sign(SQRT(ONE / (ONE + sqr(marg_n[0]))),x[n] - x[n - 1]),
               bety = betx * marg_n[0];
   }

   fehler = spline(n + 1, t, x, mess, alfx, betx, 1, bx, cx, dx);
   if (fehler < 0)
      return 5;
   else if (fehler > 0)
      return fehler + 5;

   fehler = spline(n + 1, t, y, mess, alfy, bety, 0, by, cy, dy);
   if (fehler != 0)
      return fehler + 20;

   return 0;
}


/* ------------------------------------------------------------------ */

/*.BA*/

/*.BE*/
int spltrans                                      /* transformiert-parametr. kub. Polynomspline .......*/
/*.BA*/
/*.IX{spltrans}*/
/*.BE*/
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
)                                                 /* Fehlercode ...........................*/
/*.BA*/

/***********************************************************************
 * die Koeffizienten einer transformiert-parametrischen interpolieren-  *
 * den kubischen Splinefunktion fuer eine geschlossene, ueberall glatte *
 * Kurve berechnen.                                                     *
.BE*)
* Eine transformiert-parametrische kubische Splinefunktion ist eine    *
* periodische kubische Splinefunktion wie in der Funktion spline(),    *
* jedoch in Polarkoordinatendarstellung. Dies ermoeglicht in vielen    *
* Faellen die Interpolation von Daten, deren Stuetzstellen nicht mono- *
* ton steigend angeordnet sind, ohne echte parametrische Splines (wie  *
* in der Funktion parspl()) berechnen zu muessen.                      *
* Hierzu transformiert die Funktion die eingegebenen Punkte zunaechst  *
* auf Polarkoordinaten (phin[i],a[i]), wobei phin[i] der Winkel und    *
* a[i] die Laenge des Ortsvektors von (x[i],y[i]) ist, i=0(1)m-1. Dies *
* muss so moeglich sein, dass die Winkelwerte phin[i] streng monoton   *
 * steigen, andernfalls ist das transformiert-parametrische Verfahren   *
 * nicht anwendbar. Dann wird eine periodische kubische Splinefunktion  *
 * mit den Winkeln phin[i] als Stuetzstellen berechnet, die die Vektor- *
 * laengen a[i] interpoliert. Um die Monotonie der phin[i] zu errei-    *
 * chen, kann es notwendig sein, den Koordinatenursprung auf einen      *
 * Punkt  P = (px, py)  zu verschieben und das Koordinatensystem um ei- *
 * nen Winkel phid zu drehen. (px, py) muss so in der durch die         *
 * (x[i], y[i])  beschriebenen Flaeche liegen, dass jeder von P ausge-  *
 * hende Polarstrahl die Randkurve der Flaeche nur einmal schneidet.    *
 * P kann sowohl vom Benutzer vorgegeben als auch von der Funktion be-  *
 * rechnet werden. Der hier berechnete Wert ist allerdings nur als Vor- *
 * schlagswert aufzufassen, der in unguenstigen Faellen nicht immer die *
 * Bedingungen erfuellt. Ausserdem muessen die (x[i],y[i]) in der Rei-  *
 * henfolge angeordnet sein, die sich ergibt, wenn man die Randkurve    *
 * der Flaeche, beginnend bei i = 1, im mathematisch positiven Sinn     *
 * durchlaeuft. Da die Kurve geschlossen ist, muss  x[m-1] = x[0]  und  *
 * y[m-1] = y[0]  sein.                                                 *
 *                                                                      *
 * Eingabeparameter:                                                    *
 * =================                                                    *
 * m:    Anzahl der Stuetzstellen (mindestens 3)                        *
 * x:    [0..m-1]-Vektor mit den Stuetzstellen                          *
 * y:    [0..m-1]-Vektor mit den zu interpolierenden Stuetzwerten       *
 * mv:   Marke fuer die Verschiebung des Koordinatenursprungs.          *
 *       mv > 0: Der Benutzer gibt die Koordinaten px, py vor.          *
 *       mv = 0: keine Verschiebung (d.h. px = py = 0)                  *
 *       mv < 0: px und py werden hier berechnet.                       *
 *               Es wird gesetzt:                                       *
 *               px = (xmax + xmin) / 2                                 *
 *               py = (ymax + ymin) / 2                                 *
 *               mit xmax = max(x[i]), xmin = min(x[i]),                *
 *                   ymax = max(y[i]), ymin = min(y[i]), i=0(1)m-1.     *
 *               Zur Beachtung: Hierdurch ist nicht notwendigerweise    *
 *               sichergestellt, dass P die oben genannten Bedingungen  *
 *               erfuellt. Falls die Funktion mit dem Fehlercode -3     *
 *               endet, muss P vom Benutzer vorgegeben werden.          *
 *                                                                      *
 * px: \ fuer mv > 0: vorgegebene Koordinaten des Punktes P             *
 * py: /                                                                *
 *                                                                      *
 * Ausgabeparameter:                                                    *
 * =================                                                    *
 * a: \  [0..m-1]-Vektoren mit Splinekoeffizienten in der Darstellung   *
 * b:  \     S(phi)  =  a[i] + b[i] * (phi - phin[i])                   *
 * c:  /                     + c[i] * (phi - phin[i]) ^ 2               *
 * d: /                      + d[i] * (phi - phin[i]) ^ 3               *
 *       fuer  phin[i] <= phi <= phin[i+1],   i=0(1)m-2 .               *
 *       Die a[i] sind die Vektorlaengen der (x[i],y[i]) in der Polar-  *
 *       koordinatendarstellung. b, c und d werden auch noch fuer       *
 *       Zwischenergebnisse missbraucht.                                *
 * phin: [0..m-1]-Vektor mit den Winkelkoordinaten der (x[i],y[i]) in   *
 *       der Polarkoordinatendarstellung.                               *
 *       Es ist phin[0]   = 0,                                          *
 *              phin[i]   = arctan((y[i] - py) / (x[i] - px)) - phid,   *
 *                          i=1(1)m-2                                   *
 *              phin[m-1] = 2 * Pi                                      *
 *                                                                      *
 * px: \ Koordinaten des Verschiebungspunktes P                         *
 * py: /                                                                *
 * phid: Winkel, um den das Koordinatensystem eventuell gedreht wurde.  *
 *       Es ist phid = arctan(y[0] / x[0]).                             *
 *                                                                      *
 * Funktionswert:                                                       *
 * ==============                                                       *
 *  0: kein Fehler                                                      *
 * -1: m < 3                                                            *
 * -3: Die phin[i] sind nicht streng monoton steigend.                  *
 * -4: x[m-1] != x[0]  oder  y[m-1] != y[0]                             *
 * >0: Fehler in spline()                                               *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * spline, REAL, PI, sqr, SQRT, ACOS, ZERO, TWO                         *
.BA*)
***********************************************************************/
/*.BE*/

{
   REAL xmin,                                     /* Minimum der x[i]                      */
      xmax,                                       /* Maximum der x[i]                      */
      ymin,                                       /* Minimum der y[i]                      */
      ymax,                                       /* Maximum der y[i]                      */
      sa,                                         /* sin(-phid)                            */
      ca;                                         /* cos(-phid)                            */
   int  n,                                        /* m - 1, Index der letzten Stuetzstelle */
      i;                                          /* Laufvariable                          */

   n = m - 1;

   /* ---------------- die Vorbedingungen ueberpruefen --------------- */
   if (n < 2)
      return -1;
   if (x[0] != x[n] || y[0] != y[n])
      return -4;

   /* ---------------- die Koordinaten transformieren ---------------- */
   if (mv == 0)                                   /* das Koordinatensystem  nicht verschieben? */
   {
      *px = *py = ZERO;
      for (i = 0; i <= n; i++)
         b[i] = x[i],
            c[i] = y[i];
   }
   else                                           /* den Koordinatenursprung nach (px, py) verschieben? */
   {
      if (mv < 0)                                 /* Sollen py und py berechnet werden? */
      {
         xmax = x[0];
         xmin = x[0];
         ymax = y[0];
         ymin = y[0];
         for (i = 1; i <= n; i++)
         {
            if (x[i] > xmax)
               xmax = x[i];
            if (x[i] < xmin)
               xmin = x[i];
            if (y[i] > ymax)
               ymax = y[i];
            if (y[i] < ymin)
               ymin = y[i];
         }
         *px = (xmax + xmin) / TWO;
         *py = (ymax + ymin) / TWO;
      }

      for (i = 0; i <= n; i++)                    /* die verschoben Punkte (x[i],y[i]) */
         b[i] = x[i] - *px,                       /* in (b[i],c[i]) aufgewahren        */
            c[i] = y[i] - *py;
   }

   /* ---- die transformierten Stuetzstellen berechnen:           ---- */
   /* ---- 1. die a[i] berechnen. Abbruch, wenn  a[i] = 0, d. h.  ---- */
   /* ----    wenn (px, py) mit einer Stuetzstelle zusammenfaellt ---- */
   for (i = 0; i <= n; i++)
   {
      a[i] = SQRT(sqr(b[i]) + sqr(c[i]));
      if (a[i] == ZERO)
         return -3;
   }

   /*------------------------------------------------------------------*/
   /* 2. die um alpha gedrehten Koordinaten X1, Y1 berechnen           */
   /*    nach den Gleichungen:                                         */
   /*                                                                  */
   /*  (X1)   ( cos(alpha)   -sin(alpha) ) (X)                         */
   /*  (  ) = (                          ) ( )                         */
   /*  (Y1)   ( sin(alpha)    cos(alpha) ) (Y)                         */
   /*                                                                  */
   /*  mit alpha = -phid                                               */
   /*------------------------------------------------------------------*/

   *phid = ACOS(b[0] / a[0]);
   if (c[0] < ZERO)
      *phid = TWO * PI - *phid;
   ca = b[0] / a[0];
   sa = -c[0] / a[0];
   for (i = 0; i <= n; i++)                       /* die gedrehten Koordinaten */
      d[i] = b[i] * ca - c[i] * sa,               /* (b[i],c[i]) in            */
         c[i] = b[i] * sa + c[i] * ca;            /* (d[i],c[i] ablegen        */

   /* ------ die Winkelkoordinaten phin[i] berechnen. Abbruch,  ------ */
   /* ------ wenn die Winkel nicht streng monoton steigend sind ------ */
   phin[0] = ZERO;
   for (i = 1; i < n; i++)
   {
      phin[i] = ACOS(d[i] / a[i]);
      if (c[i] < ZERO)
         phin[i] = TWO * PI - phin[i];
      if (phin[i] <= phin[i - 1])
         return -3;
   }
   phin[n] = TWO * PI;

   /* --------------- die Splinekoeffizienten berechnen -------------- */
   return spline(n + 1, phin, a, 4, ZERO, ZERO, 0, b, c, d);
}


/* ------------------------- ENDE kubsplin.c ------------------------ */
