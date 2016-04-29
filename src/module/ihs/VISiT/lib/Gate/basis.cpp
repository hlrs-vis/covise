/* -------------------------- MODUL basis.c ------------------------- */

/***********************************************************************
 *                                                                      *
 * grundlegende Funktionen: Definitionsdatei                            *
 * -----------------------------------------                            *
 *                                                                      *
 * Programmiersprache: ANSI C                                           *
 * Compiler:           Turbo C 2.0                                      *
 * Rechner:            IBM PS/2 70 mit 80387                            *
 * Autor:              Juergen Dietel, Rechenzentrum der RWTH Aachen    *
 * Datum:              MI 12. 8. 1992                                   *
 *                                                                      *
 ***********************************************************************/

#include <Gate/include/basis.h>                   /* wegen NULL, freopen, stdout, fprintf, stderr,  */
/*       stdin, SQRT, EXP, sqrt, MACH_EPS,        */
/*       POSMAX, epsquad, maxroot, pi, ATAN, sqr, */
/*       umleiten, readln, intervall, horner,     */
/*       norm_max, skalprod, copy_vector, REAL,   */
/*       ONE, TWO, FOUR, ZERO, HALF, FABS,        */
/*       boolean, FOUR, basis, mach_eps, epsroot, */
/*       exp_1, posmin, sqrtlong, comdiv, comabs, */
/*       quadsolv, SetVec, CopyVec, ReadVec,      */
/*       WriteVec, SetMat, CopyMat, ReadMat,      */
/*       WriteMat, WriteHead, WriteEnd, LogError, */
/*       fgetc, stdin, SWAP                       */

/*--------------------------------------------------------------------*/

int basis(void)                                   /* Basis der Zahlendarstellung bestimmen */
/*.IX{basis}*/

/***********************************************************************
 * die Basis der Zahlendarstellung maschinenunabhaengig bestimmen,      *
 * falls nicht schon in frueheren Aufrufen geschehen, und als           *
 * Funktionswert zurueckgeben                                           *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, ONE, TWO                                                       *
 ***********************************************************************/

{
   REAL x,
      eins,
      b;

   x = eins = b = ONE;

   while ((x + eins) - x == eins)
      x *= TWO;
   while ((x + b) == x)
      b *= TWO;

   return (int)((x + b) - x);
}


/*--------------------------------------------------------------------*/

static int groesser1(REAL x)                      /* Hilfsfunktion fuer mach_eps() */

/***********************************************************************
 * Hilfsfunktion fuer mach_eps() (noetig, um gewisse Compileroptimie-   *
 * rungen zu umgehen): melden, ob die uebergebene Zahl x groesser als   *
 * Eins ist                                                             *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, ONE                                                            *
 ***********************************************************************/

{
   return x > ONE;
}


/*--------------------------------------------------------------------*/

REAL mach_eps(void)                               /* Maschinengenauigkeit bestimmen */
/*.IX{mach\unt eps}*/

/***********************************************************************
 * die Maschinengenauigkeit maschinenunabhaengig bestimmen, falls nicht *
 * schon in frueheren Aufrufen geschehen, und als Funktionswert         *
 * zurueckgeben                                                         *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, boolean, FALSE, ONE, HALF, TWO, TRUE                           *
 ***********************************************************************/

{
   static REAL    epsilon;
   static boolean schon_berechnet = FALSE;

   if (! schon_berechnet)
   {
      for (epsilon = ONE; groesser1(ONE + epsilon); )
         epsilon *= HALF;
      epsilon         *= TWO;
      schon_berechnet  = TRUE;
   }

   return epsilon;
}


/*--------------------------------------------------------------------*/

REAL epsroot(void)                                /* Wurzel aus der Maschinengenauigkeit bestimmen */
/*.IX{epsroot}*/

/***********************************************************************
 * die Wurzel aus der Maschinengenauigkeit berechnen, falls nicht schon *
 * in frueheren Aufrufen geschehen, und als Funktionswert zurueckgeben  *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, boolean, FALSE, TRUE, SQRT, MACH_EPS                           *
 ***********************************************************************/

{
   static REAL    save_mach_eps_root;
   static boolean schon_berechnet     = FALSE;

   if (! schon_berechnet)
      schon_berechnet    = TRUE,
         save_mach_eps_root = SQRT(MACH_EPS);

   return save_mach_eps_root;
}


/*--------------------------------------------------------------------*/

REAL epsquad(void)                                /* Quadrat der Maschinengenauigkeit bestimmen */
/*.IX{epsquad}*/

/***********************************************************************
 * das Quadrat der Maschinengenauigkeit berechnen, falls nicht schon    *
 * in frueheren Aufrufen geschehen, und als Funktionswert zurueckgeben  *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, boolean, FALSE, TRUE, MACH_EPS                                 *
 ***********************************************************************/

{
   static REAL    save_mach_eps_quad;
   static boolean schon_berechnet     = FALSE;

   if (! schon_berechnet)
      schon_berechnet    = TRUE,
         save_mach_eps_quad = MACH_EPS * MACH_EPS;

   return save_mach_eps_quad;
}


/*--------------------------------------------------------------------*/

REAL maxroot(void)                                /* Wurzel der groessten Maschinenzahl bestimmen */
/*.IX{maxroot}*/

/***********************************************************************
 * die Wurzel der groessten Maschinenzahl berechnen, falls nicht schon  *
 * in frueheren Aufrufen geschehen, und als Funktionswert zurueckgeben: *
 * Der Wert ist  2 ^ (MAX_EXP/2).                                       *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, boolean, FALSE, TRUE, SQRT, POSMAX                             *
 ***********************************************************************/

{
   static REAL       save_maxroot;
   static boolean    schon_berechnet = FALSE;
   REAL              faktor;
   unsigned long int n;

   if (! schon_berechnet)
   {
      save_maxroot = ONE;
      faktor       = TWO;
      for (n = MAX_EXP / 2; n > 1; n /= 2, faktor *= faktor)
         if (n % 2 != 0)
            save_maxroot *= faktor;
      save_maxroot    *= faktor;
      schon_berechnet  = TRUE;
   }

   return save_maxroot;
}


/*--------------------------------------------------------------------*/

REAL posmin(void)                                 /* kleinste positive Gleitkommazahl bestimmen */
/*.IX{posmin}*/

/***********************************************************************
 * die kleinste positive Gleitkommazahl berechnen, falls nicht schon    *
 * in frueheren Aufrufen geschehen, und als Funktionswert zurueckgeben. *
 * Der Algorithmus besteht darin, dass in y der Anfangswert Eins so     *
 * lange halbiert wird, bis er sich nicht mehr aendert oder zu Null     *
 * wird. Damit dabei keine Endlosschleife entsteht, wurde ein Zaehler   *
 * in die Iteration eingebaut, der nach 32767 Halbierungsversuchen auf  *
 * jeden Fall fuer den Abbruch der Schleife sorgt.                      *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, boolean, FALSE, ONE, TWO, ZERO, HALF, TRUE                     *
 ***********************************************************************/

{
   static REAL    y;                              /* nach Schleifenende: kleinste Gleitkommazahl   */
   REAL           x;                              /* in der Schleife:  2 * y (zum Vergleich mit y) */
   int            i;                              /* Zaehler zur Verhinderung einer Endlosschleife */
   static boolean schon_berechnet = FALSE;

   if (! schon_berechnet)
   {
      for (i = 0, x = ONE, y = TWO; x != ZERO && x != y && i < 32767; i++)
         y =  x,
            x *= HALF;
      schon_berechnet = TRUE;
   }

   return y;
}


/*--------------------------------------------------------------------*/

REAL pi(void)                                     /* die Kreiszahl pi bestimmen */
/*.IX{pi}*/

/***********************************************************************
 * die Kreiszahl PI berechnen, falls nicht schon in frueheren Aufrufen  *
 * geschehen, und als Funktionswert zurueckgeben                        *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, boolean, FALSE, TRUE, FOUR, ATAN                               *
 ***********************************************************************/

{
   static REAL    save_pi;
   static boolean schon_berechnet = FALSE;

   if (! schon_berechnet)
      schon_berechnet = TRUE,
         save_pi         = FOUR * ATAN(ONE);

   return save_pi;
}


/*--------------------------------------------------------------------*/

REAL exp_1(void)                                  /* die Eulersche Zahl bestimmen */
/*.IX{exp\unt 1}*/

/***********************************************************************
 * die Eulersche Zahl berechnen, falls nicht schon in frueheren         *
 * Aufrufen geschehen, und als Funktionswert zurueckgeben               *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, boolean, FALSE, TRUE, EXP, ONE                                 *
 ***********************************************************************/

{
   static REAL    save_exp_1;
   static boolean schon_berechnet = FALSE;

   if (! schon_berechnet)
      schon_berechnet = TRUE,
         save_exp_1      = EXP(ONE);

   return save_exp_1;
}


/*--------------------------------------------------------------------*/

REAL sqr(REAL x)                                  /* eine Gleitkommazahl quadrieren */
/*.IX{sqr}*/

/***********************************************************************
 * das Quadrat von x berechnen und als Funktionswert zurueckgeben       *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL                                                                 *
 ***********************************************************************/
{
   return x * x;
}


/*--------------------------------------------------------------------*/

void fehler_melden                                /* Fehlermeldung auf stdout und stderr schreiben */
/*.IX{fehler\unt melden}*/
(
char text[],                                      /* Fehlerbeschreibung .......*/
int  fehlernummer,                                /* Nummer des Fehlers .......*/
char dateiname[],                                 /* Ort des Fehlers: .........*/
int  zeilennummer                                 /* Dateiname, Zeilennummer ..*/
)

/***********************************************************************
 * eine Fehlermeldung ausgeben, zusammen mit dem Namen der Quelldatei   *
 * und der Zeilennummer, wo der Fehler bemerkt wurde, eventuell auch    *
 * (falls fehlernummer > 0) mit der Fehlernummer einer Funktion, die    *
 * kurz vorher aufgerufen worden ist                                    *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * sprintf, fprintf, stderr, printf                                     *
 ***********************************************************************/

{
   char meldung[200];

   if (fehlernummer == 0)
      sprintf(meldung, "\n%s, Zeile %d: %s!\n",
         dateiname, zeilennummer, text);
   else
      sprintf(meldung, "\n%s, Zeile %d: Fehler %d in %s!\n",
         dateiname, zeilennummer, fehlernummer, text);

   fprintf(stderr, "%s", meldung);
   printf("%s", meldung);
}


/*--------------------------------------------------------------------*/

int umleiten                                      /* stdin und stdout eventuell auf Datei umleiten */
/*.IX{umleiten}*/
(
int argc,                                         /* Argumentanzahl in der Kommandozeile ..*/
char *argv[]                                      /* Vektor der Argumente .................*/
)                                                 /* Fehlercode ...........................*/

/***********************************************************************
 * eine eventuelle Ein- bzw. Ausgabedatei der Standardeingabe (stdin)   *
 * bzw. der Standardausgabe (stdout) zuordnen                           *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * freopen, stdout, NULL, fprintf, stderr, stdin                        *
 ***********************************************************************/

{
   if (argc >= 3)                                 /* mindestens 2 Argumente? */
      if (freopen(argv[2], "w", stdout) == NULL)  /* Ausgabedatei oeffnen */
   {
      fprintf(stderr, "Fehler beim Oeffnen von %s!\n", argv[2]);
      return 1;
   }
   if (argc >= 2)                                 /* mindestens 1 Argument?  */
      if (freopen(argv[1], "r", stdin) == NULL)   /* Eingabedatei oeffnen */
   {
      fprintf(stderr, "Fehler beim Oeffnen von %s!\n", argv[1]);
      return 2;
   }

   return 0;
}


/*--------------------------------------------------------------------*/

void readln(void)                                 /* Zeilenrest in stdin ueberlesen */
/*.IX{readln}*/

/***********************************************************************
 * den Zeilenrest in der Standardeingabe einschliesslich des Zeilen-    *
 * trennzeichens ueberlesen                                             *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * fgetc, stdin, EOF                                                    *
 ***********************************************************************/

{
   int c;

   while ((c = fgetc(stdin)) != '\n' && c != EOF)
      ;
}


/*--------------------------------------------------------------------*/

void getline                                      /* eine Zeile Text von stdin lesen .............*/
/*.IX{getline}*/
(
char kette[],                                     /* Vektor mit dem gelesenen Text .......*/
int limit                                         /* maximale Laenge von kette ...........*/
)

/***********************************************************************
 * eine Zeile aus der Standardeingabe nach kette lesen, hoechstens je-  *
 * doch limit-1 Zeichen. Ein eventueller Zeilenrest (einschliesslich    *
 * des Zeilentrennzeichens) wird ueberlesen. limit ist die Hoechstzahl  *
 * an Zeichen, die in kette Platz finden. Da kette immer mit einem      *
 * Nullbyte abgeschlossen wird, werden also immer hoechstens limit-1    *
 * Zeichen aus der Eingabe in kette eingetragen.                        *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * fgetc, stdin, EOF                                                    *
 ***********************************************************************/

{
   int c;

   for (c = 0; --limit >= 1 && (c = fgetc(stdin)) != '\n' && c != EOF; )
      *kette++ = (char)c;
   *kette = '\0';                                 /* mit einem Nullbyte abschliessen */
   while (c != '\n' && c != EOF)                  /* einen eventuellen Zeilenrest    */
      c = fgetc(stdin);                           /* ueberlesen                      */
}


/*--------------------------------------------------------------------*/

int intervall                                     /* Intervallnummer einer Zerlegung suchen ...........*/
/*.IX{intervall}*/
(
int n,                                            /* Zahl der Teilintervalle - 1 ..........*/
REAL xwert,                                       /* Zahl, deren Intervall gesucht wird ...*/
REAL x[]                                          /* Grenzen der Teilintervalle ...........*/
)                                                 /* Index des gesuchten Teilintervalls ...*/

/***********************************************************************
 * Dies ist eine allgemein verwendbare Prozedur zum Suchen eines Inter- *
 * valls, das einen vorgegebenen Wert xwert enthaelt. Gegeben ist eine  *
 * streng monoton steigende Folge von Werten x[i], i=0(1)n. Gesucht ist *
 * der Index ix derart, dass x[ix] <= xwert < x[ix+1] gilt. Falls       *
 * xwert < x[0] bzw. xwert >= x[n-1], bekommt ix den Wert 0 bzw. n-1.   *
 * ix hat also den Wertebereich 0, 1, ..., n-1. intervall findet ix mit *
 * einem schnellen Binaersuchverfahren. Diese Aufgabe ist regelmaessig  *
 * zu loesen, wenn polynomiale Splinefunktionen auszuwerten sind. Die   *
 * x[i] sind dann die Stuetzstellen.                                    *
 *                                                                      *
 * Eingabeparameter:                                                    *
 * =================                                                    *
 * n:     Index der letzten Stuetzstelle in x                           *
 * xwert: Wert, dessen Intervall gesucht wird                           *
 * x:     [0..n]-Feld mit den Intervallgrenzen (siehe oben)             *
 *                                                                      *
 * Funktionswert:                                                       *
 * ==============                                                       *
 * gesuchter Index ix (siehe oben)                                      *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL                                                                 *
 ***********************************************************************/

{
   int ix,
      m;

   for (ix = 0; m = (ix + n) >> 1, m != ix; )
      if (xwert < x[m])
         n = m;
   else
      ix = m;

   return ix;
}


/*--------------------------------------------------------------------*/

REAL horner                                       /* Hornerschema zur Polynomauswertung .............*/
/*.IX{horner}*/
(
int n,                                            /* Polynomgrad .........*/
REAL a[],                                         /* Polynomkoeffizienten */
REAL x                                            /* Auswertungsstelle ...*/
)                                                 /* Polynomwert .........*/

/***********************************************************************
 * ein Polynom P in der Darstellung                                     *
 *       P(x)  =  a[0] + a[1] * x + a[2] * x^2 + ... + a[n] * x^n       *
 * nach dem Hornerschema auswerten                                      *
 *                                                                      *
 * Eingabeparameter:                                                    *
 * =================                                                    *
 * n: Grad des Polynoms                                                 *
 * a: [0..n]-Vektor mit den Koeffizienten des Polynoms                  *
 * x: Stelle, an der das Polynom auszuwerten ist                        *
 *                                                                      *
 * Funktionswert:                                                       *
 * ==============                                                       *
 * P(x)                                                                 *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL                                                                 *
 ***********************************************************************/

{
   REAL summe;

   for (summe = a[n], n--; n >= 0; n--)
      summe = summe * x + a[n];

   return summe;
}


/*--------------------------------------------------------------------*/

REAL norm_max                                     /* Maximumnorm eines REAL-Vektors bestimmen .......*/
/*.IX{norm\unt max}*/
(
REAL vektor[],                                    /* Eingabevektor ..........*/
int  n                                            /* Zahl der Vektorelemente */
)                                                 /* Maximumnorm ............*/

/***********************************************************************
 * die Maximumnorm des [0..n-1]-Vektors v berechnen und als             *
 * Funktionswert zurueckgeben                                           *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, FABS, ZERO                                                     *
 ***********************************************************************/

{
   REAL norm,                                     /* Vergleichswert fuer die Normberechnung */
      betrag;                                     /* Zwischenspeicher fuer den Betrag eines */
   /* Vektorelements                         */

   for (n--, norm = ZERO; n >= 0; n--, vektor++)
      if ((betrag = FABS(*vektor)) > norm)
         norm = betrag;

   return norm;
}


/* ------------------------------------------------------------------ */

REAL skalprod                                     /* Standardskalarprodukt zweier REAL-Vektoren */
/*.IX{skalprod}*/
(
REAL v[],                                         /* 1. Vektor ......................*/
REAL w[],                                         /* 2. Vektor ......................*/
int  n                                            /* Vektorlaenge ...................*/
)                                                 /* Skalarprodukt ..................*/

/***********************************************************************
 * das Skalarprodukt   v[0] * w[0] + ... + v[n-1] * w[n-1]  der beiden  *
 * [0..n-1]-Vektoren v und w berechnen                                  *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL, ZERO                                                           *
 ***********************************************************************/

{
   REAL skalarprodukt;

   for (skalarprodukt = ZERO; n-- != 0; )
      skalarprodukt += (*v++) * (*w++);

   return skalarprodukt;
}


/* ------------------------------------------------------------------ */

void copy_vector                                  /* einen REAL-Vektor kopieren ................*/
/*.IX{copy\unt vector}*/
(
REAL ziel[],                                      /* Zielvektor ...............*/
REAL quelle[],                                    /* Quellvektor ..............*/
int  n                                            /* Anzahl der Vektorelemente */
)

/***********************************************************************
 * n Elemente des Vektors quelle in den Vektor ziel kopieren            *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * REAL                                                                 *
 ***********************************************************************/

{
   for (n--; n >= 0; n--)
      *ziel++ = *quelle++;
}


/* -------------------- Albert Beckers Funktionen ------------------- */

static char Separator[] =
"--------------------------------------------------------------------";

long double sqrtlong (long double x)
/*.IX{sqrtlong}*/
/*====================================================================*
 *                                                                    *
 *  Doppelt genaue Quadratwurzel                                      *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Eingabeparameter:                                                *
 *   ================                                                 *
 *      x        long double x;                                       *
 *               Zahl, deren Wurzel zu ziehen ist                     *
 *                                                                    *
 *   Rueckgabewert:                                                   *
 *   =============                                                    *
 *               Doppelt genaue Quadratwurzel                         *
 *====================================================================*/
{
   long double y;
   long double yold;
   int i;

   y = (long double) sqrt ((double) (x));
   for (i = 0; i < 10; i++)
   {
      if (y == 0.0L) return 0.0L;
      yold = y;
      y = (y + x / y) * 0.5L;
      if (ABS (y - yold) <= ABS (y) * MACH_EPS) break;
   }
   return y;
}


int comdiv                                        /* Komplexe Division .........................*/
/*.IX{comdiv}*/
(
REAL   ar,                                        /* Realteil Zaehler ................*/
REAL   ai,                                        /* Imaginaerteil Zaehler ...........*/
REAL   br,                                        /* Realteil Nenner .................*/
REAL   bi,                                        /* Imaginaerteil Nenner ............*/
REAL * cr,                                        /* Realteil Quotient ...............*/
REAL * ci                                         /* Imaginaerteil Quotient ..........*/
)
/*====================================================================*
 *                                                                    *
 *  Komplexe Division c = a / b                                       *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Eingabeparameter:                                                *
 *   ================                                                 *
 *      ar,ai    REAL   ar, ai;                                       *
 *               Real-,Imaginaerteil des Dividenden                   *
 *      br,bi    REAL   br, bi;                                       *
 *               Real-,Imaginaerteil des Divisors                     *
 *                                                                    *
 *   Ausgabeparameter:                                                *
 *   ================                                                 *
 *      cr,ci    REAL   *cr, *ci;                                     *
 *               Real- u. Imaginaerteil des Divisionsergebnisses      *
 *                                                                    *
 *   Rueckgabewert:                                                   *
 *   =============                                                    *
 *      = 0      Ergebnis ok                                          *
 *      = 1      Division durch 0                                     *
 *                                                                    *
 *   Benutzte Macros: ABS                                             *
 *   ===============                                                  *
 *                                                                    *
 *====================================================================*/
{
   REAL tmp;

   if (br == ZERO && bi == ZERO) return (1);

   if (ABS (br) > ABS (bi))
   {
      tmp  = bi / br;
      br   = tmp * bi + br;
      *cr  = (ar + tmp * ai) / br;
      *ci  = (ai - tmp * ar) / br;
   }
   else
   {
      tmp  = br / bi;
      bi   = tmp * br + bi;
      *cr  = (tmp * ar + ai) / bi;
      *ci  = (tmp * ai - ar) / bi;
   }

   return (0);
}


REAL comabs                                       /* Komplexer Absolutbetrag ...................*/
/*.IX{comabs}*/
(
REAL  ar,                                         /* Realteil ........................*/
REAL  ai                                          /* Imaginaerteil ...................*/
)
/*====================================================================*
 *                                                                    *
 *  Komplexer Absolutbetrag von a                                     *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Eingabeparameter:                                                *
 *   ================                                                 *
 *      ar,ai    REAL   ar, ai;                                       *
 *               Real-,Imaginaerteil von a                            *
 *                                                                    *
 *   Rueckgabewert:                                                   *
 *   =============                                                    *
 *      Absolutbetrag von a                                           *
 *                                                                    *
 *   Benutzte Macros: SQRT, ABS, SWAP                                 *
 *   ===============                                                  *
 *                                                                    *
 *====================================================================*/
{
   if (ar == ZERO && ai == ZERO) return (ZERO);

   ar = ABS (ar);
   ai = ABS (ai);

   if (ai > ar)                                   /* Tausche ai und ar .........................*/
      SWAP (REAL, ai, ar)

         return ((ai == ZERO) ? (ar) : (ar * SQRT (ONE + ai / ar * ai / ar)));
}


void quadsolv                                     /* Komplexe quadratische Gleichung ...........*/
/*.IX{quadsolv}*/
(
REAL    ar,                                       /* Quadratischer Koeffizient .......*/
REAL    ai,
REAL    br,                                       /* Linearer Koeffizient ............*/
REAL    bi,
REAL    cr,                                       /* Konstanter Koeffizient ..........*/
REAL    ci,
REAL *  tr,                                       /* Loesung .........................*/
REAL *  ti
)
/*====================================================================*
 *                                                                    *
 *  Berechnung der betragsmaessig kleinsten Loesung der Gleichung     *
 *  a*t**2 + b*t + c = 0. a, b, c und t sind komplex.                 *
 *                                       2                            *
 *  Formel dazu: t = 2c / (-b +/- sqrt (b  - 4ac)).                   *
 *  Die Formel ist auch fuer a=0 gueltig!                             *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Eingabeparameter:                                                 *
 *  ================                                                  *
 *      ar, ai   a Faktor von t**2             REAL   ar, ai;         *
 *      br, bi   b Faktor von t                REAL   br, bi;         *
 *      cr, ci   c konstanter Term             REAL   cr, ci;         *
 *                                                                    *
 *  Ausgabeparameter:                                                 *
 *  ================                                                  *
 *      tr, ti  t komplexe Loesung             REAL   *tr, *ti;       *
 *                                                                    *
 *  Benutzte Macros: SQRT                                             *
 *  ===============                                                   *
 *                                                                    *
 *====================================================================*/
{
   REAL pr, pi, qr, qi, h;

   pr = br * br - bi * bi;
   pi = TWO * br * bi;                            /*  p = b * b             */

   qr = ar * cr - ai * ci;
   qi = ar * ci + ai * cr;                        /*  q = a * c             */

   pr = pr - (REAL)4.0 * qr;
   pi = pi - (REAL)4.0 * qi;                      /* p = b * b - 4 * a * c  */

   h  = SQRT (pr * pr + pi * pi);                 /* q = sqrt (p)           */

   qr = h + pr;
   if (qr > ZERO)
      qr = SQRT (qr * HALF);
   else
      qr = ZERO;

   qi = h - pr;
   if (qi > ZERO)
      qi = SQRT (qi * HALF);
   else
      qi = ZERO;

   if (pi < ZERO) qi = -qi;

   h = qr * br + qi * bi;                         /* p = -b +/- q, so dass Betrag p gross  */

   if (h > ZERO)
   {
      qr = -qr;
      qi = -qi;
   }

   pr = qr - br;
   pi = qi - bi;
   h = pr * pr + pi * pi;                         /* t = (2 * c) / p      */

   if (h == ZERO)
   {
      *tr = ZERO;
      *ti = ZERO;
   }
   else
   {
      *tr = TWO * (cr * pr + ci * pi) / h;
      *ti = TWO * (ci * pr - cr * pi) / h;
   }
}


void SetVec (int n, REAL x[], REAL val)
/*.IX{SetVec}*/
/*====================================================================*
 *                                                                    *
 *  Vektor der Laenge n mit konstanter Zahl vorbesetzen.              *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Eingabeparameter:                                                 *
 *  ================                                                  *
 *      n        int n;                                               *
 *               Dimension des Vektors                                *
 *      x        REAL x[];                                            *
 *               Eingabevektor.                                       *
 *      val      Zuzuweisender Wert.                                  *
 *                                                                    *
 *   Ausgabeparameter:                                                *
 *   ================                                                 *
 *      x        Mit val vorbesetzter Vektor                          *
 *                                                                    *
 *====================================================================*/
{
   int i;

   for (i = 0; i < n; i++)
      x[i] = val;
}


void CopyVec (int n, REAL source[], REAL dest[])
/*.IX{CopyVec}*/
/*====================================================================*
 *                                                                    *
 *  Vektor source der Laenge n auf Vektor dest kopieren.              *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Eingabeparameter:                                                 *
 *  ================                                                  *
 *                                                                    *
 *      n        int n;                                               *
 *               Dimension des Vektors                                *
 *      source   REAL source[];                                       *
 *               Eingabevektor.                                       *
 *      dest     REAL dest[];                                         *
 *               Vektor auf den kopiert wird.                         *
 *                                                                    *
 *   Ausgabeparameter:                                                *
 *   ================                                                 *
 *      dest     Gleiche Inhalte wie source                           *
 *                                                                    *
 *   Achtung: Es wird fuer dest kein Speicher allokiert.              *
 *                                                                    *
 *====================================================================*/
{
   int i;

   for (i = 0; i < n; i++)
      dest[i] = source[i];
}


int ReadVec (int n, REAL x[])
/*.IX{ReadVec}*/
/*====================================================================*
 *                                                                    *
 *  Vektor x der Laenge n von stdin einlesen.                         *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Eingabeparameter:                                                 *
 *  ================                                                  *
 *                                                                    *
 *      n        int n;                                               *
 *               Dimension des Vektors                                *
 *      x        REAL x[];                                            *
 *               Eingabevektor.                                       *
 *                                                                    *
 *   Ausgabeparameter:                                                *
 *   ================                                                 *
 *      x        Eingelesene Inhalte                                  *
 *                                                                    *
 *   Achtung: Es wird fuer x kein Speicher allokiert.                 *
 *                                                                    *
 *====================================================================*/
{
   int i;
   double tmp;

   for (i = 0; i < n; i++)
   {
      if (scanf (FORMAT_IN, &tmp) <= 0) return (-1);
      x[i] = (REAL) tmp;
   }

   return (0);
}


int WriteVec (int n, REAL x[])
/*.IX{WriteVec}*/
/*====================================================================*
 *                                                                    *
 *  Vektor x der Laenge n auf stdout ausgeben.                        *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Eingabeparameter:                                                 *
 *  ================                                                  *
 *                                                                    *
 *      n        int n;                                               *
 *               Dimension des Vektors                                *
 *      x        REAL x[];                                            *
 *               Eingabevektor.                                       *
 *                                                                    *
 *   Rueckgabewert:                                                   *
 *   =============                                                    *
 *      =  0     Alles ausgegeben.                                    *
 *      = -1     Fehler beim Schreiben auf stdout                     *
 *                                                                    *
 *====================================================================*/
{
   int i;

   for (i = 0; i < n; i++)
      if (printf (FORMAT_126LF, x[i]) <= 0) return (-1);
   if (printf ("\n") <= 0) return (-1);

   return 0;
}


void SetMat (int m, int n, REAL * a[], REAL val)
/*.IX{SetMat}*/
/*====================================================================*
 *                                                                    *
 *  m x n Matrix mit konstanter Zahl val vorbesetzen.                 *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Eingabeparameter:                                                 *
 *  ================                                                  *
 *      m        int m; ( m > 0 )                                     *
 *               Zeilenzahl der Matrix.                               *
 *      n        int n; ( n > 0 )                                     *
 *               Spaltenzahl der Matrix.                              *
 *      a        REAL * a[];                                          *
 *               EingabeMatrix.                                       *
 *      val      Zuzuweisender Wert.                                  *
 *                                                                    *
 *   Ausgabeparameter:                                                *
 *   ================                                                 *
 *      a        Mit val vorbesetzte Matrix.                          *
 *                                                                    *
 *====================================================================*/
{
   int i, j;

   for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
         a[i][j] = val;
}


void CopyMat (int m, int n, REAL * source[], REAL * dest[])
/*.IX{CopyMat}*/
/*====================================================================*
 *                                                                    *
 *  m x n Matrix source auf m x n Matrix dest kopieren.               *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Eingabeparameter:                                                 *
 *  ================                                                  *
 *      m        int m; ( m > 0 )                                     *
 *               Zeilenzahl der Matrix.                               *
 *      n        int n; ( n > 0 )                                     *
 *               Spaltenzahl der Matrix.                              *
 *      source   REAL * source[];                                     *
 *               Eingabematrix.                                       *
 *      dest     REAL * dest[];                                       *
 *               Matrix, auf die kopiert wird.                        *
 *                                                                    *
 *   Ausgabeparameter:                                                *
 *   ================                                                 *
 *      dest     Gleiche Inhalte wie source                           *
 *                                                                    *
 *   Achtung: Es wird fuer dest kein Speicher allokiert.              *
 *                                                                    *
 *====================================================================*/
{
   int i, j;

   for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
         dest[i][j] = source[i][j];
}


int ReadMat (int m, int n, REAL * a[])
/*.IX{ReadMat}*/
/*====================================================================*
 *                                                                    *
 *  m x n Matrix von stdin einlesen.                                  *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Eingabeparameter:                                                 *
 *  ================                                                  *
 *      m        int m; ( m > 0 )                                     *
 *               Zeilenzahl der Matrix.                               *
 *      n        int n; ( n > 0 )                                     *
 *               Spaltenzahl der Matrix.                              *
 *      a        REAL * a[];                                          *
 *               Eingabematrix.                                       *
 *                                                                    *
 *   Ausgabeparameter:                                                *
 *   ================                                                 *
 *      a        Enthaelt die eingelesenen Inhalte                    *
 *                                                                    *
 *   Achtung: Es wird fuer a kein Speicher allokiert.                 *
 *                                                                    *
 *====================================================================*/
{
   int i, j;
   double x;

   for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
   {
      if (scanf (FORMAT_IN, &x) <= 0) return (-1);
      a[i][j] = (REAL) x;
   }

   return (0);
}


int WriteMat (int m, int n, REAL * a[])
/*.IX{WriteMat}*/
/*====================================================================*
 *                                                                    *
 *  m x n Matrix auf stdout ausgeben.                                 *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Eingabeparameter:                                                 *
 *  ================                                                  *
 *      m        int m; ( m > 0 )                                     *
 *               Zeilenzahl der Matrix.                               *
 *      n        int n; ( n > 0 )                                     *
 *               Spaltenzahl der Matrix.                              *
 *      a        REAL * a[];                                          *
 *               Auszugebende Matrix.                                 *
 *                                                                    *
 *   Rueckgabewert:                                                   *
 *   =============                                                    *
 *      =  0      Alles ausgegeben.                                   *
 *      = -1      Fehler beim Schreiben auf stdout                    *
 *                                                                    *
 *====================================================================*/
{
   int i, j;

   if (printf ("\n") <= 0) return (-1);

   for (i = 0; i < m; i++)
   {
      for (j = 0; j < n; j++)
         if (printf (FORMAT_126LF, a[i][j]) <= 0) return (-1);

      if (printf ("\n") <= 0) return (-1);
   }
   if (printf ("\n") <= 0) return (-1);

   return (0);
}


int WriteHead (char * string)
/*.IX{WriteHead}*/
/*====================================================================*
 *                                                                    *
 *  Header mit Text string auf stdout ausgeben                        *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Eingabeparameter:                                                 *
 *  ================                                                  *
 *      string   char *string;                                        *
 *               Headertext (0 terminiert)                            *
 *                                                                    *
 *   Rueckgabewert:                                                   *
 *   =============                                                    *
 *      =  0      Alles ausgegeben.                                   *
 *      = -1      Fehler beim Schreiben auf stdout                    *
 *      = -2      Ungueltiger Headertext                              *
 *                                                                    *
 *====================================================================*/
{
   if (string == NULL) return (-2);

   if (printf ("\n%s\n%s\n%s\n\n", Separator, string, Separator) <= 0)
      return (-1);

   return 0;
}


int WriteEnd ()
/*.IX{WriteEnd}*/
/*====================================================================*
 *                                                                    *
 *  Begrenzer auf stdout ausgeben                                     *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *   Rueckgabewert:                                                   *
 *   =============                                                    *
 *      =  0      Alles ausgegeben.                                   *
 *      = -1      Fehler beim Schreiben auf stdout                    *
 *                                                                    *
 *====================================================================*/
{
   if (printf ("\n%s\n\n", Separator) <= 0) return (-1);
   return 0;
}


void LogError (char * string, int rc, char * file, int line)
/*.IX{LogError}*/
/*====================================================================*
 *                                                                    *
 *  Fehler auf stdout ausgeben                                        *
 *                                                                    *
 *====================================================================*
 *                                                                    *
 *  Eingabeparameter:                                                 *
 *  ================                                                  *
 *      string   char *string;                                        *
 *               Fehlertext (0 terminiert)                            *
 *      rc       int rc;                                              *
 *               Fehler Code                                          *
 *      file     char *file;                                          *
 *               Name der C Datei, in der der Fehler auftrat          *
 *      line     int line;                                            *
 *               Zeilennummer der C Datei, in der der Fehler auftrat  *
 *                                                                    *
 *====================================================================*/
{
   if (string == NULL)
   {
      printf ("Unknown ERROR in file %s at line %d\n", file, line);
      WriteEnd ();
      return;
   }

   if (rc == 0)
      printf ("ERROR: %s, File %s, Line %d\n", string, file, line);
   else
      printf ("ERROR: %s, rc = %d, File %s, Line %d\n",
         string, rc, file, line);

   WriteEnd ();
   return;
}


/* -------------------------- ENDE basis.c -------------------------- */
