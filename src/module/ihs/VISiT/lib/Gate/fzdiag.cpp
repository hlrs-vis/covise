/*.BA*/

/*.FE{C 4.11.1}{Systeme mit zyklisch tridiagonaler Matrix}
               {Systeme mit zyklisch tridiagonaler Matrix}*/

/*.BE*/
/* ------------------------- MODUL fzdiag.c ------------------------- */

#include <Gate/include/basis.h>
#include <Gate/include/u_proto.h>

/*.BA*/

/*.BE*/
int tzdiag                                        /* Zyklisch tridiagonale Gleichungssystem ....*/
/*.BA*/
/*.IX{tzdiag}*/
/*.BE*/
(
int   n,                                          /* Dimension der Matrix ............*/
REAL  lower[],                                    /* Subdiagonale ....................*/
REAL  diag[],                                     /* Diagonale .......................*/
REAL  upper[],                                    /* Superdiagonale ..................*/
REAL  lowrow[],                                   /* Untere Zeile ....................*/
REAL  ricol[],                                    /* Rechte Spalte ...................*/
REAL  b[],                                        /* Rechte Seite / Loesung ..........*/
int   rep                                         /* rep = 0, 1 ......................*/
)
/*.BA*/

/*====================================================================*
 *                                                                    *
 *  tzdiag bestimmt die Loesung x des linearen Gleichungssystems      *
 *  A * x = b mit zyklisch tridiagonaler n x n Koeffizienten-         *
 *  matrix A.                                                         *
.BE*)
 *  Sie ist in den 5 Vektoren lower, upper, diag, lowrow und ricol    *
 *  wie folgt abgespeichert:                                          *
 *                                                                    *
 *       ( diag[0]  upper[0]    0        0  .   . 0   ricol[0]   )    *
 *       ( lower[1] diag[1]   upper[1]   0      .     .   0      )    *
*       (   0      lower[2]  diag[2]  upper[2]   0       .      )    *
*  A =  (   .        0       lower[3]  .     .       .   .      )    *
*       (   .          .           .        .     .      0      )    *
*       (   .             .            .        .      .        )    *
*       (   0               .             .        . upper[n-2] )    *
 *       ( lowrow[0]  0 .  .    0        lower[n-1]   diag[n-1]  )    *
 *                                                                    *
 *  Speicherplatz fuer lowrow[1],..,lowrow[n-3] und ricol[1],...,     *
 *  ricol[n-3] muss zusaetzlich bereitgestellt werden, da dieser      *
 *  fuer die Aufnahme der Zerlegungsmatrix verfuegbar sein muss, die  *
 *  auf die 5 genannten Vektoren ueberspeichert wird.                 *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Anwendung:                                                       *
 *   =========                                                        *
 *      Vorwiegend fuer diagonaldominante zyklische Tridiagonalmatri- *
 *      zen wie sie bei der Spline-Interpolation auftreten.           *
 *      Fuer diagonaldominante Matrizen existiert immer eine LU-      *
 *      Zerlegung.                                                    *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Eingabeparameter:                                                *
 *   ================                                                 *
 *      n        Dimension der Matrix ( > 2 )  int n                  *
 *      lower    untere Nebendiagonale         REAL   lower[n]        *
 *      diag     Hauptdiagonale                REAL   diag[n]         *
 *      upper    obere Nebendiagonale          REAL   upper[n]        *
 *      b        rechte Seite des Systems      REAL   b[n]            *
 *      rep      = 0  erstmaliger Aufruf       int rep                *
 *               !=0  wiederholter Aufruf                             *
 *                    fuer gleiche Matrix,                            *
 *                    aber verschiedenes b.                           *
 *                                                                    *
 *   Ausgabeparameter:                                                *
 *   ================                                                 *
 *      b        Loesungsvektor des Systems,   REAL   b[n]            *
 *               die urspruengliche rechte Seite wird ueberspeichert  *
 *                                                                    *
 *      lower    ) enthalten bei rep = 0 die Zerlegung der Matrix;    *
 *      diag     ) die urspruenglichen Werte von lower u. diag werden *
 *      upper    ) ueberschrieben                                     *
 *      lowrow   )                             REAL   lowrow[n-2]     *
 *      ricol    )                             REAL   ricol[n-2]      *
 *                                                                    *
 *   Die Determinante der Matrix ist bei rep = 0 durch                *
 *      det A = diag[0] * ... * diag[n-1]     bestimmt.               *
 *                                                                    *
 *   Rueckgabewert:                                                   *
 *   =============                                                    *
 *      = 0      alles ok                                             *
 *      = 1      n < 3 gewaehlt oder ungueltige Eingabeparameter      *
 *      = 2      Die Zerlegungsmatrix existiert nicht                 *
 *                                                                    *
.BA*)
*====================================================================*/
/*.BE*/
{
   REAL     tmp;
   int i;

   if (n < 3) return (1);
   if (lower == NULL || diag == NULL || upper == NULL ||
      lowrow == NULL || ricol == NULL) return (1);

   if (rep == 0)                                  /*  Wenn rep = 0 ist,     */
   {                                              /*  Zerlegung der         */
      lower[0] = upper[n-1] = ZERO;               /*  Matrix berechnen.     */

      if (ABS (diag[0]) < MACH_EPS) return (2);
      /* Ist ein Diagonalelement  */
      tmp = ONE / diag[0];                        /* betragsmaessig kleiner   */
      upper[0] *= tmp;                            /* MACH_EPS, so ex. keine   */
      ricol[0] *= tmp;                            /* Zerlegung.               */

      for (i = 1; i < n-2; i++)
      {
         diag[i] -= lower[i] * upper[i-1];
         if (ABS(diag[i]) < MACH_EPS) return (2);
         tmp = ONE / diag[i];
         upper[i] *= tmp;
         ricol[i] = -lower[i] * ricol[i-1] * tmp;
      }

      diag[n-2] -= lower[n-2] * upper[n-3];
      if (ABS(diag[n-2]) < MACH_EPS) return (2);

      for (i = 1; i < n-2; i++)
         lowrow[i] = -lowrow[i-1] * upper[i-1];

      lower[n-1] -= lowrow[n-3] * upper[n-3];
      upper[n-2] = ( upper[n-2] - lower[n-2] * ricol[n-3] ) / diag[n-2];

      for (tmp = ZERO, i = 0; i < n-2; i++)
         tmp -= lowrow[i] * ricol[i];
      diag[n-1] += tmp - lower[n-1] * upper[n-2];

      if (ABS (diag[n-1]) < MACH_EPS) return (2);
   }                                              /* end if ( rep == 0 ) */

   b[0] /= diag[0];                               /* Vorwaertselemination    */
   for (i = 1; i < n - 1; i++)
      b[i] = ( b[i] - b[i-1] * lower[i] ) / diag[i];

   for (tmp = ZERO, i = 0; i < n - 2; i++)
      tmp -= lowrow[i] * b[i];

   b[n-1] = ( b[n-1] + tmp - lower[n-1] * b[n-2] ) / diag[n-1];

   b[n-2] -= b[n-1] * upper[n-2];                 /* Rueckwaertselimination  */
   for (i = n - 3; i >= 0; i--)
      b[i] -= upper[i] * b[i+1] + ricol[i] * b[n-1];

   return (0);
}


/* -------------------------- ENDE fzdiag.c ------------------------- */
