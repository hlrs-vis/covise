/*.BA*/

/*.FE{C 4.10.1}{Systeme mit tridiagonaler Matrix}
               {Systeme mit tridiagonaler Matrix}*/

/*.BE*/
/* ------------------------- MODUL ftdiag.c ------------------------- */

#include <Gate/include/basis.h>
#include <Gate/include/u_proto.h>

/*.BA*/

/*.BE*/
int trdiag                                        /* Tridiagonale Gleichungssysteme ............*/
/*.BA*/
/*.IX{trdiag}*/
/*.BE*/
(
int     n,                                        /* Dimension der Matrix ............*/
REAL    lower[],                                  /* Subdiagonale ....................*/
REAL    diag[],                                   /* Diagonale .......................*/
REAL    upper[],                                  /* Superdiagonale ..................*/
REAL    b[],                                      /* Rechte Seite / Loesung ..........*/
int     rep                                       /* rep = 0, 1 ......................*/
)
/*.BA*/

/*====================================================================*
 *                                                                    *
 *  trdiag bestimmt die Loesung x des linearen Gleichungssystems      *
 *  A * x = b mit tridiagonaler n x n Koeffizientenmatrix A.          *
.BE*)
 *  Sie ist in den 3 Vektoren lower, upper und diag wie folgt         *
 *  abgespeichert:                                                    *
 *                                                                    *
 *       ( diag[0]  upper[0]    0        0  .   .     .   0      )    *
 *       ( lower[1] diag[1]   upper[1]   0      .     .   .      )    *
 *       (   0      lower[2]  diag[2]  upper[2]   0       .      )    *
*  A =  (   .        0       lower[3]  .     .       .          )    *
*       (   .          .           .        .     .      0      )    *
*       (   .             .            .        .      .        )    *
*       (                   .             .        . upper[n-2] )    *
 *       (   0 .   .    .       0        lower[n-1]   diag[n-1]  )    *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Anwendung:                                                       *
 *   =========                                                        *
 *      Vorwiegend fuer diagonaldominante Tridiagonalmatrizen, wie    *
 *      sie bei der Spline-Interpolation auftreten.                   *
 *      Fuer diagonaldominante Matrizen existiert immer eine LU-      *
 *      Zerlegung; fuer nicht diagonaldominante Tridiagonalmatrizen   *
 *      sollte die Funktion band vorgezogen werden, da diese mit      *
 *      Spaltenpivotsuche arbeitet und daher numerisch stabiler ist.  *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Eingabeparameter:                                                *
 *   ================                                                 *
 *      n        Dimension der Matrix ( > 1 )  int n                  *
 *                                                                    *
 *      lower    untere Nebendiagonale         REAL   lower[n]        *
 *      diag     Hauptdiagonale                REAL   diag[n]         *
 *      upper    obere Nebendiagonale          REAL   upper[n]        *
 *                                                                    *
 *               bei rep != 0 enthalten lower, diag und upper die     *
 *               Dreieckzerlegung der Ausgangsmatrix.                 *
 *                                                                    *
 *      b        rechte Seite des Systems      REAL   b[n]            *
 *      rep      = 0  erstmaliger Aufruf       int rep                *
 *               !=0  wiederholter Aufruf                             *
 *                    fuer gleiche Matrix,                            *
 *                    aber verschiedenes b.                           *
 *                                                                    *
 *   Ausgabeparameter:                                                *
 *   ================                                                 *
 *      b        Loesungsvektor des Systems;   REAL   b[n]            *
 *               die urspruengliche rechte Seite wird ueberspeichert  *
 *                                                                    *
 *      lower    ) enthalten bei rep = 0 die Zerlegung der Matrix;    *
 *      diag     ) die urspruenglichen Werte von lower u. diag werden *
 *      upper    ) ueberschrieben                                     *
 *                                                                    *
 *   Die Determinante der Matrix ist bei rep = 0 durch                *
 *      det A = diag[0] * ... * diag[n-1] bestimmt.                   *
 *                                                                    *
 *   Rueckgabewert:                                                   *
 *   =============                                                    *
 *      = 0      alles ok                                             *
 *      = 1      n < 2 gewaehlt                                       *
 *      = 2      Die Dreieckzerlegung der Matrix existiert nicht      *
 *                                                                    *
.BA*)
*====================================================================*/
/*.BE*/
{
   int i;

   if (n < 2) return (1);                         /*  n mindestens 2        */

   if (lower == NULL || diag == NULL || upper == NULL ||
      b == NULL) return (1);

   /*  Wenn rep = 0 ist,     */
   /*  Dreieckzerlegung der  */
   if (rep == 0)                                  /*  Matrix u. det be-     */
   {                                              /*  stimmen               */
      for (i = 1; i < n; i++)
      {
         if (ABS(diag[i-1]) < MACH_EPS)           /*  Wenn ein diag[i] = 0  */
            return (2);                           /*  ist, ex. keine Zerle- */
         lower[i] /= diag[i-1];                   /*  gung.                 */
         diag[i] -= lower[i] * upper[i-1];
      }
   }

   if (ABS(diag[n-1]) < MACH_EPS) return (2);

   for (i = 1; i < n; i++)                        /*  Vorwaertselimination  */
      b[i] -= lower[i] * b[i-1];

   b[n-1] /= diag[n-1];                           /* Rueckwaertselimination */
   for (i = n-2; i >= 0; i--)
      b[i] = ( b[i] - upper[i] * b[i+1] ) / diag[i];

   return (0);
}


/* -------------------------- ENDE ftdiag.c ------------------------- */
