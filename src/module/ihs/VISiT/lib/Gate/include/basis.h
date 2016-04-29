/*.KA{C 0}{Hilfsvereinbarungen}{Hilfsvereinbarungen}*/
/*.FE{C 0.1}
     {Grundlegende Deklarationen und Definitionen}
     {Grundlegende Deklarationen und Definitionen}*/

/* ---------------------- DEKLARATIONEN basis.h --------------------- */

/***********************************************************************
 *                                                                      *
 * grundlegende Funktionen: Deklarationsdatei (mit Typen und Makros)    *
 * -----------------------------------------------------------------    *
 *                                                                      *
 * Programmiersprache: ANSI C                                           *
 * Autor:              Juergen Dietel, Rechenzentrum der RWTH Aachen    *
 * Datum:              MI 30. 9. 1992                                   *
 *                                                                      *
 ***********************************************************************/

/***********************************************************************
 * vorsorgen fuer den Fall, dass diese Deklarationsdatei mehrfach in    *
 * einen Quelltext aufgenommen wird                                     *
 ***********************************************************************/

#ifndef BASIS_H_INCLUDED
#define BASIS_H_INCLUDED

/***********************************************************************
 * einige andere hier benoetigte Deklarationsdateien aufnehmen.         *
 * Bei den Standarddeklarationsdateien geschieht das vor allem deshalb, *
 * weil man dann in den C-Modulen nur diese Datei einfuegen muss, um    *
 * alle gewuenschten Standardnamen zur Verfuegung zu haben, und weil    *
 * sich dann Anpassungen bei Verwendung von Nichtstandardcompilern      *
 * (hoffentlich) auf diese Datei beschraenken.                          *
 ***********************************************************************/

#include <stdio.h>                                /* wegen NULL, printf, scanf, fprintf, stderr,  */
/*       freopen, stdin, stdout, fopen, fclose, */
/*       fclose, fseek, SEEK_END, SEEK_SET,     */
/*       ftell, fwrite, fread, size_t, getchar  */
#include <stdlib.h>                               /* wegen abs                                    */
#include <math.h>                                 /* wegen fabs, sqrt, pow, exp, sin, cos, log,   */
/*       atan, acos                             */
#include <float.h>                                /* wegen DBL_EPSILON, DBL_MAX                   */
#ifdef sun                                        /* fuer GNU CC unter SunOS                      */
#include <unistd.h>                               /* wegen SEEK_END                               */
#else
#ifdef amigados                                   /* fuer GNU CC auf einem Amiga                  */
#include <unistd.h>                               /* wegen SEEK_END                               */
#endif
#endif
#ifndef SEEK_END                                  /* SEEK_END immer noch nicht definiert (z. B    */
/* bei GNU CC 2.1 fuer i386-MS-DOS-Rechner)?    */
#include <unistd.h>                               /* wegen SEEK_END                               */
#endif

/***********************************************************************
 * die gewuenschte Genauigkeit fuer die Gleitkommarechnung einstellen:  *
 * Falls das Makro FLOAT definiert ist, wird mit einfacher Genauigkeit  *
 * (Datentyp float) gearbeitet, bei LDOUBLE mit der hoechstmoeglichen   *
 * Genauigkeit (Datentyp long double), sonst mit doppelter Genauigkeit  *
 * (Datentyp double). LDOUBLE bringt aber nur mit solchen Compilern     *
 * etwas, bei denen long double genauer ist als double (z. B. Turbo C,  *
 * aber nicht QuickC).                                                  *
 * Zur Ein- bzw. Ausgabe von Gleitkommazahlen sollten die Makros LZS    *
 * (Laengenzeichen fuer scanf()) bzw. LZP (Laengenzeichen fuer          *
 * printf()) verwendet werden, also zwei verschiedene, da laut          *
 * ANSI C-Standard bei der Ausgabe von double-Werten im Gegensatz zur   *
 * Eingabe das Laengenzeichen "l" weggelassen werden soll.              *
 * wichtig: Falls der benutzte Compiler die Makros FLT_MAX, LDBL_MAX,   *
 * ======== DBL_MAX bzw. FLT_MAX_EXP, LDBL_MAX_EXP, DBL_MAX_EXP         *
 *          (i. a. in float.h zu finden) nicht kennt, muessen an den    *
 *          durch !!!! markierten Stellen passende Werte eingesetzt     *
 *          werden!                                                     *
 ***********************************************************************/

#ifdef FLOAT                                      /* einfache Genauigkeit? ............*/

typedef float     REAL;                           /* Standardgleitkommatyp float ......*/
/*.IX{REAL}*/
typedef double    LONG_REAL;                      /* genauerer Gleitkommatyp ..........*/
/*.IX{LONG\unt REAL}*/

#ifdef FLT_EPSILON                                /* ANSI C-Compiler? .................*/
#define MACH_EPS  (REAL)FLT_EPSILON
/*.IX{MACH\unt EPS}*/
#else                                             /* kein ANSI C-Compiler? ............*/
#define MACH_EPS  mach_eps()                      /* die Maschinengenauigkeit .........*/
#endif

#ifdef FLT_MAX_EXP                                /* ANSI C-Compiler? .................*/
#define MAX_EXP   FLT_MAX_EXP                     /* Binaerexponent von POSMAX ........*/
/*.IX{MAX\unt EXP}*/
#else                                             /* kein ANSI C-Compiler? ............*/
#define MAX_EXP   128                             /* muss angepasst werden!!!! ........*/
#endif

#ifdef FLT_MAX                                    /* ANSI C-Compiler? ................ */
#define POSMAX    (REAL)FLT_MAX                   /* groesste Gleitkommazahl ......... */
/*.IX{POS\unt MAX}*/
#else                                             /* kein ANSI C-Compiler? ........... */
#define POSMAX    1e38f                           /* muss angepasst werden!!!! ....... */
#endif

#ifdef FLT_MIN                                    /* ANSI C-Compiler? .................*/
#define POSMIN    (REAL)FLT_MIN                   /* kleinste positive Gleitkommazahl .*/
/*.IX{POS\unt MIN}*/
#else                                             /* kein ANSI C-Compiler? ............*/
#define POSMIN    posmin()
#endif

#define LZS       ""                              /* Laengenzeichen fuer formatierte   */
/*.IX{LZS}*/
/* Eingabe von Gleitkommazahlen      */
#define LZP       ""                              /* Laengenzeichen fuer formatierte   */
/*.IX{LZP}*/
/* Ausgabe von Gleitkommazahlen      */

#else
#ifdef LDOUBLE                                    /* hoechste Genauigkeit? ............*/

typedef long double  REAL;                        /* Standardgleitkommatyp long double */
/*.IX{REAL}*/
typedef long double  LONG_REAL;                   /* "genauerer" Gleitkommatyp ........*/
/*.IX{LONG\unt REAL}*/
#define LONG_DOUBLE_USED

#ifdef LDBL_EPSILON                               /* ANSI C-Compiler? .................*/
#define MACH_EPS  (REAL)LDBL_EPSILON
/*.IX{MACH\unt EPS}*/
#else                                             /* kein ANSI C-Compiler? ............*/
#define MACH_EPS  mach_eps()                      /* die Maschinengenauigkeit .........*/
#endif

#ifdef LDBL_MAX_EXP                               /* ANSI C-Compiler? .................*/
#define MAX_EXP   LDBL_MAX_EXP                    /* Binaerexponent von POSMAX ........*/
/*.IX{MAX\unt EXP}*/
#else                                             /* kein ANSI C-Compiler? ............*/
#define MAX_EXP   1023                            /* muss angepasst werden!!!! ........*/
#endif

#ifdef LDBL_MAX                                   /* ANSI C-Compiler? .................*/
#define POSMAX    (REAL)LDBL_MAX                  /* groesste Gleitkommazahl ..........*/
/*.IX{POS\unt MAX}*/
#else                                             /* kein ANSI C-Compiler? ............*/
#define POSMAX    1e100l                          /* muss angepasst werden!!!! ........*/
#endif

#ifdef LDBL_MIN                                   /* ANSI C-Compiler? .................*/
#define POSMIN    (REAL)LDBL_MIN                  /* kleinste positive Gleitkommazahl .*/
/*.IX{POS\unt MIN}*/
#else                                             /* kein ANSI C-Compiler? ............*/
#define POSMIN    posmin()
#endif

#define LZS       "L"                             /* Laengenzeichen fuer formatierte   */
/*.IX{LZS}*/
/* Eingabe von Gleitkommazahlen      */
#define LZP       "L"                             /* Laengenzeichen fuer formatierte   */
/*.IX{LZP}*/
/* Ausgabe von Gleitkommazahlen      */

#else                                             /* doppelte Genauigkeit? ............*/

typedef double       REAL;                        /* Standardgleitkommatyp double .....*/
/*.IX{REAL}*/
typedef long double  LONG_REAL;                   /* genauerer Gleitkommatyp ..........*/
/*.IX{LONG\unt REAL}*/

#ifdef DBL_EPSILON                                /* ANSI C-Compiler? .................*/
#define MACH_EPS  (REAL)DBL_EPSILON
/*.IX{MACH\unt EPS}*/
#else                                             /* kein ANSI C-Compiler? ............*/
#define MACH_EPS  mach_eps()                      /* die Maschinengenauigkeit .........*/
#endif

#ifdef DBL_MAX_EXP                                /* ANSI C-Compiler? .................*/
#define MAX_EXP   DBL_MAX_EXP                     /* Binaerexponent von POSMAX ........*/
/*.IX{MAX\unt EXP}*/
#else                                             /* kein ANSI C-Compiler? ............*/
#define MAX_EXP   1023                            /* muss angepasst werden!!!! ........*/
#endif

#ifdef DBL_MAX                                    /* ANSI C-Compiler? .................*/
#define POSMAX    (REAL)DBL_MAX                   /* groesste Gleitkommazahl ..........*/
/*.IX{POS\unt MAX}*/
#else                                             /* kein ANSI C-Compiler? ............*/
#define POSMAX    1e100                           /* muss angepasst werden!!!! ........*/
#endif

#ifdef DBL_MIN                                    /* ANSI C-Compiler? .................*/
#ifdef __BORLANDC__
#if __BORLANDC__ <= 0x0200                        /* Borland C++ 2.0 fuer DOS? ........*/
/* weil der Wert           */
#define POSMIN    2.2250738585072017E-308         /* 2.2250738585072014E-308 */
#else                                             /* aus `float.h' als Null  */
/* betrachtet wird!        */
#define POSMIN    DBL_MIN                         /* kleinste positive Gleitkommazahl .*/
#endif
#else
#define POSMIN    DBL_MIN                         /* kleinste positive Gleitkommazahl .*/
#endif
/*.IX{POS\unt MIN}*/
#else                                             /* kein ANSI C-Compiler? ............*/
#define POSMIN    posmin()
#endif

#define LZS       "l"                             /* Laengenzeichen fuer formatierte   */
/*.IX{LZS}*/
/* Eingabe von Gleitkommazahlen      */
#define LZP       ""                              /* Laengenzeichen fuer formatierte   */
/*.IX{LZP}*/
/* Ausgabe von Gleitkommazahlen      */
#endif
#endif

/***********************************************************************
 * einige wichtige Datentypen vereinbaren                               *
 ***********************************************************************/

typedef enum {FALSE, TRUE}
boolean;
/*.IX{FALSE}*/
/*.IX{TRUE}*/
/*.IX{boolean}*/

/* Funktionszeigertypen fuer die Approximation in Kapitel 8 ..........*/
typedef REAL (*ansatzfnk) (int i, REAL x);
/*.IX{ansatzfnk}*/
typedef REAL (*approxfnk) (REAL c[], REAL x);
/*.IX{approxfnk}*/
typedef void (*ableitfnk) (REAL x, REAL c[], REAL *d);
/*.IX{ableitfnk}*/

/* Typ der Funktion, die die rechte Seite der expliziten       .......*/
/* gewoehnlichen Differentialgleichung  y' = f(x,y)  auswertet .......*/
typedef REAL (*dglfnk)(REAL x, REAL y);
/*.IX{dglfnk}*/

/* Typ der Funktion, die die rechte Seite des           ..............*/
/* Differentialgleichungssystems  y' = f(x,y) auswertet ..............*/
typedef void (*dglsysfnk)(REAL x, REAL y[], REAL f[]);
/*.IX{dglsysfnk}*/

/* Typ der Funktion, die den Wert der Randbedingung  r(ya, yb) .......*/
/* eines Zwei-Punkt-Randwertproblems 1. Ordnung berechnet      .......*/
typedef void (*rndbedfnk)(REAL ya[], REAL yb[], REAL r[]);
/*.IX{rndbedfnk}*/

/* Aufzaehlungstyp zur Klassifizierung von Fehlercodes, die von den ..*/
/* meisten Funktionen zurueckgegeben werden, die ein numerisches    ..*/
/* Verfahren verwirklichen                                          ..*/
typedef enum { KEIN_FEHLER, WARNUNG, UNBEKANNT, FATAL }
fehler_t;
/*.IX{KEIN\unt FEHLER}*/
/*.IX{WARNUNG}*/
/*.IX{UNBEKANNT}*/
/*.IX{FATAL}*/
/*.IX{fehler\unt t}*/

typedef REAL abl_mat1[4][2];                      /* werden zur Auswertung von Spline- */
/*.IX{abl\unt mat1}*/
typedef REAL abl_mat2[6][2];                      /* funktionen in spliwert benoetigt  */
/*.IX{abl\unt mat2}*/

typedef REAL mat4x4[4][4];                        /* Typ fuer bikubische Splines       */

/*--------------------------------------------------------------------*
 * Typvereinbarungen von Albert Becker                                *
 *--------------------------------------------------------------------*/

/* Reelle Funktionen ................................................*/
typedef REAL (* REALFCT)  (REAL);
/*.IX{REALFCT}*/

/* Reelle mehrdimensionale Funktionen ...............................*/
typedef int (* FNFCT)  (int, REAL [], REAL []);
/*.IX{FNFCT}*/

/* Funktionen zur Bestimmung der Jacobi Matrix ......................*/
typedef int (* JACOFCT)  (int, REAL [], REAL * []);
/*.IX{JACOFCT}*/

/***********************************************************************
 * einige wichtige Makros vereinbaren                                   *
 * Hinweis: Borland C++ bietet ab der Version 3.0 erstmals auch zum Typ *
 *          long double passende Gleitkommafunktionen (z. B. expl()     *
 *          statt exp(), sinl() statt sin()). Da Borland C++ 3.0 aber   *
 *          kein Makro zu definieren scheint, das eine Unterscheidung   *
 *          von Borland C++ 2.0 erlaubt, muss man das selbst in die     *
 *          Hand nehmen: Falls man unter Borland C++ 3.0 mit            *
 *          long double arbeiten will, sollte man vor der Uebersetzung  *
 *          das Makro BC3 definieren. Dann werden automatisch die neuen *
 *          genaueren Gleitkommafunktionen verwendet.                   *
 ***********************************************************************/

#define BASIS     basis()                         /* die Basis der Zahlendarstellung  */
/*.IX{BASIS}*/
#define EPSROOT   epsroot()                       /* die Wurzel aus MACH_EPS          */
/*.IX{EPSROOT}*/
#define EPSQUAD   epsquad()                       /* das Quadrat von MACH_EPS         */
/*.IX{EPSQUAD}*/
#define MAXROOT   maxroot()                       /* die Wurzel aus der               */
/*.IX{MAXROOT}*/
/* groessten Gleitkommazahl         */
#ifndef PI
#define PI        pi()                            /* die Kreiszahl                    */
/*.IX{PI}*/
#endif
#define EXP_1     exp_1()                         /* die Eulersche Zahl               */
/*.IX{EXP\unt 1}*/

#define ZERO      (REAL)0.0                       /* Namen fuer haeufig vorkommende   */
/*.IX{ZERO}*/
#define ONE       (REAL)1.0                       /* Gleitkommakonstanten vereinbaren */
/*.IX{ONE}*/
#define TWO       (REAL)2.0
/*.IX{TWO}*/
#define THREE     (REAL)3.0
/*.IX{THREE}*/
#define FOUR      (REAL)4.0
/*.IX{FOUR}*/
#define FIVE      (REAL)5.0
/*.IX{FIVE}*/
#define SIX       (REAL)6.0
/*.IX{SIX}*/
#define EIGHT     (REAL)8.0
/*.IX{EIGHT}*/
#define NINE      (REAL)9.0
/*.IX{NINE}*/
#define TEN       (REAL)10.0
/*.IX{TEN}*/
#define HALF      (REAL)0.5
/*.IX{HALF}*/

#ifdef __BORLANDC__
#if __BORLANDC__ >= 0x0400                        /* ein von BC2 unterscheidbares BC, */
/* das long double-Funktionen kennt */
/* (mindestens Borland C++ 3.1 oder */
/* Borland C++ 1.00 fuer OS/2)?     */
#define BC3                                       /* BC3 automatisch definieren       */
#endif
#endif

#ifdef _MSC_VER
#if _MSC_VER     >= 0x0258                        /* ein von QC2 unterscheidbares MC, */
/* das long double-Funktionen kennt */
/* (mindestens Microsoft C 6.00A)?  */
#define MC6                                       /* MC6 automatisch definieren       */
#endif
#endif

#if defined(LDOUBLE) &&                     /* Borland C++ 3.0 oder  */(defined(BC3) || defined(MC6))
/* Microsoft C 6.0 mit    */
/* hoechster Genauigkeit? */
#define FABS(x)    fabsl((x))                     /* die long double-Ver-   */
/*.IX{FABS}*/
#define SQRT(x)    sqrtl((x))                     /* sionen der wichtigsten */
/*.IX{SQRT}*/
#define POW(x, y)  powl((x), (y))                 /* Gleitkommafunktionen   */
/*.IX{POW}*/
#define SIN(x)     sinl((x))                      /* verwenden              */
/*.IX{SIN}*/
#define COS(x)     cosl((x))
/*.IX{COS}*/
#define EXP(x)     expl((x))
/*.IX{EXP}*/
#define LOG(x)     logl((x))
/*.IX{LOG}*/
#define ATAN(x)    atanl((x))
/*.IX{ATAN}*/
#define ACOS(x)    acosl((x))
/*.IX{ACOS}*/
#define COSH(x)    coshl((x))
/*.IX{COSH}*/

#else                                             /* geringere Genauigkeit  */
/* oder kein BC3 und      */
/* kein MC6?              */
#define FABS(x)    (REAL)fabs((double)(x))        /* Namen fuer wichtige    */
#ifdef LONG_DOUBLE_USED                           /* Gleitkommafunktionen   */
#define SQRT(x)    sqrtlong((x))                  /* vereinbaren, die eine  */
/*.IX{SQRT}*/
#else                                             /* Benutzung mit jeder    */
#define SQRT(x)    (REAL)sqrt((double)(x))        /* der drei moeglichen    */
#endif                                            /* Genauigkeiten erlauben */
#define POW(x, y)  (REAL)pow((double)(x), \
/*.IX{POW}*/ \
(double)(y))
#define SIN(x)     (REAL)sin((double)(x))
/*.IX{SIN}*/
#define COS(x)     (REAL)cos((double)(x))
/*.IX{COS}*/
#define EXP(x)     (REAL)exp((double)(x))
/*.IX{EXP}*/
#define LOG(x)     (REAL)log((double)(x))
/*.IX{LOG}*/
#define ATAN(x)    (REAL)atan((double)(x))
/*.IX{ATAN}*/
#define ACOS(x)    (REAL)acos((double)(x))
/*.IX{ACOS}*/
#define COSH(x)    (REAL)cosh((double)(x))
/*.IX{COSH}*/
#endif

#undef sign
#undef min
#undef max
#define sign(x, y) (((y) < ZERO) ? -FABS(x) :     /* |x| mal Vor-  */    FABS(x))
/*.IX{sign}*/ 
/* zeichen von y */

#define min(a, b)        (((a) < (b)) ? (a) : (b))
/*.IX{min}*/
#define max(a, b)        (((a) > (b)) ? (a) : (b))
/*.IX{max}*/
#define SWAP(typ, a, b)  {typ temp; temp = a; a = b; b = temp;}
/* zwei Objekte beliebi- */
/*.IX{SWAP}*/ \
/* gen Typs vertauschen  */

/* ------------------ Makros von Albert Becker ---------------------- */
#define ABS(X) (((X) >= ZERO) ? (X) : -(X))       /* Absolutbetrag von X */
/*.IX{ABS}*/
#define SIGN(X,Y) /*.IX{SIGN}*/ /* Vorzeichen von   */   (((Y) < ZERO) ? -ABS(X) : ABS(X))
                                                  /* Y mal ABS(X)     */
#define SQR(X) ((X) * (X))                        /* Quadrat von X       */
/*.IX{SQR}*/

#define FORMAT_IN      "%lg"                      /* Input Format fuer REAL  */
/*.IX{FORMAT\unt IN}*/
#define FORMAT_LF      "% " LZP "f "                /* Format l fuer REAL      */
/*.IX{FORMAT\unt LF}*/
#define FORMAT_126LF   "% 12.6" LZP "f "            /* Format 12.6f fuer REAL  */
/*.IX{FORMAT\unt 126LF}*/
#define FORMAT_2010LF  "% 20.10" LZP "f "           /* Format 20.10f fuer REAL */
/*.IX{FORMAT\unt 2010LF}*/
#define FORMAT_2016LF  "% 20.16" LZP "f "           /* Format 20.16f fuer REAL */
/*.IX{FORMAT\unt 2016LF}*/
#define FORMAT_LE      "% " LZP "e "                /* Format e fuer REAL      */
/*.IX{FORMAT\unt LE}*/
#define FORMAT_2016LE  "% 20.16" LZP "e "           /* Format 20.16e fuer REAL */
/*.IX{FORMAT\unt 2016LE}*/

/***********************************************************************
 * alle in basis.c definierten externen Funktionen deklarieren          *
 ***********************************************************************/

int basis(void);                                  /* Basis der Zahlendarstellung bestimmen */

REAL mach_eps(void);                              /* Maschinengenauigkeit bestimmen */

REAL epsroot(void);                               /* Wurzel aus der Maschinengenauigkeit bestimmen */

REAL epsquad(void);                               /* Quadrat der Maschinengenauigkeit bestimmen */

REAL maxroot(void);                               /* Wurzel der groessten Maschinenzahl bestimmen */

REAL posmin(void);                                /* kleinste positive Gleitkommazahl bestimmen */

REAL pi(void);                                    /* die Kreiszahl pi bestimmen */

REAL exp_1(void);                                 /* die Eulersche Zahl bestimmen */

REAL sqr(REAL x);                                 /* eine Gleitkommazahl quadrieren */

void fehler_melden                                /* Fehlermeldung auf stdout und stderr schreiben */
(
char text[],                                      /* Fehlerbeschreibung .......*/
int  fehlernummer,                                /* Nummer des Fehlers .......*/
char dateiname[],                                 /* Ort des Fehlers: .........*/
int  zeilennummer                                 /* Dateiname, Zeilennummer ..*/
);

int umleiten                                      /* stdin und stdout eventuell auf Datei umleiten */
(
int argc,                                         /* Argumentanzahl in der Kommandozeile ..*/
char *argv[]                                      /* Vektor der Argumente .................*/
);                                                /* Fehlercode ...........................*/

void readln(void);                                /* Zeilenrest in stdin ueberlesen */

void getline                                      /* eine Zeile Text von stdin lesen .............*/
(
char kette[],                                     /* Vektor mit dem gelesenen Text .......*/
int limit                                         /* maximale Laenge von kette ...........*/
);

int intervall                                     /* Intervallnummer einer Zerlegung suchen ...........*/
(
int n,                                            /* Zahl der Teilintervalle - 1 ..........*/
REAL xwert,                                       /* Zahl, deren Intervall gesucht wird ...*/
REAL x[]                                          /* Grenzen der Teilintervalle ...........*/
);                                                /* Index des gesuchten Teilintervalls ...*/

REAL horner                                       /* Hornerschema zur Polynomauswertung .............*/
(
int n,                                            /* Polynomgrad .........*/
REAL a[],                                         /* Polynomkoeffizienten */
REAL x                                            /* Auswertungsstelle ...*/
);                                                /* Polynomwert .........*/

REAL norm_max                                     /* Maximumnorm eines REAL-Vektors bestimmen .......*/
(
REAL vektor[],                                    /* Eingabevektor ..........*/
int  n                                            /* Zahl der Vektorelemente */
);                                                /* Maximumnorm ............*/

REAL skalprod                                     /* Standardskalarprodukt zweier REAL-Vektoren */
(
REAL v[],                                         /* 1. Vektor ......................*/
REAL w[],                                         /* 2. Vektor ......................*/
int  n                                            /* Vektorlaenge ...................*/
);                                                /* Skalarprodukt ..................*/

void copy_vector                                  /* einen REAL-Vektor kopieren ................*/
(
REAL ziel[],                                      /* Zielvektor ...............*/
REAL quelle[],                                    /* Quellvektor ..............*/
int  n                                            /* Anzahl der Vektorelemente */
);

/*--------------------------------------------------------------------*
 * Basisfunktionen Kapitel 1 (von Albert Becker) .....................*
 *--------------------------------------------------------------------*/

long double sqrtlong  (long double x);

int comdiv (                                      /* Komplexe Division .........................*/
REAL     ar,                                      /* Realteil Zaehler ................*/
REAL     ai,                                      /* Imaginaerteil Zaehler ...........*/
REAL     br,                                      /* Realteil Nenner .................*/
REAL     bi,                                      /* Imaginaerteil Nenner ............*/
REAL   * cr,                                      /* Realteil Quotient ...............*/
REAL   * ci                                       /* Imaginaerteil Quotient ..........*/
);

REAL  comabs (                                    /* Komplexer Absolutbetrag ...................*/
REAL   ar,                                        /* Realteil ........................*/
REAL   ai                                         /* Imaginaerteil ...................*/
);

void quadsolv (                                   /* Komplexe quadratische Gleichung ...........*/
REAL     ar,                                      /* Quadratischer Koeffizient .......*/
REAL     ai,
REAL     br,                                      /* Linearer Koeffizient ............*/
REAL     bi,
REAL     cr,                                      /* Konstanter Koeffizient ..........*/
REAL     ci,
REAL   * tr,                                      /* Loesung .........................*/
REAL   * ti
);

void SetVec                                       /* Vektor vorbesetzen ........................*/
(int n, REAL x[], REAL val);

void CopyVec                                      /* Vektor kopieren ...........................*/
(int n, REAL source[], REAL dest[]);

int ReadVec                                       /* Vektor von stdin einlesen .................*/
(int n, REAL x[]);

int WriteVec                                      /* Vektor auf stdout ausgeben ................*/
(int n, REAL x[]);

void SetMat                                       /* Matrix vorbesetzen ........................*/
(int m, int n, REAL * a[], REAL val);

void CopyMat                                      /* Matrix kopieren ...........................*/
(int m, int n, REAL * source[], REAL * dest[]);

int ReadMat                                       /* Matrix von stdin einlesen .................*/
(int m, int n, REAL * a[]);

int WriteMat                                      /* Matrix auf stdout ausgeben ................*/
(int m, int n, REAL * mat[]);

int WriteHead  (char *s);                         /* Header auf stdout schreiben .....*/

int WriteEnd  (void);                             /* Separator auf stdout schreiben ..*/

void LogError                                     /* Error auf stdout ausgeben .................*/
(char *s, int rc, char *file, int line);
#endif

/* -------------------------- ENDE basis.h -------------------------- */
