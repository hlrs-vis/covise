/* ------------------------- MODUL vmblock.c ------------------------ */

/***********************************************************************
 *                                                                      *
 * Verwaltung eines Satzes von dynamischen Vektoren und Matrizen        *
 * -------------------------------------------------------------        *
 *                                                                      *
 * Idee:   In vielen Unterprogrammen der Numerikbibliothek werden immer *
 *         wieder dynamisch vereinbarte Vektoren und Matrizen           *
 *         benoetigt. Dabei tritt jedoch manchmal das Problem auf, dass *
 *         nur fuer einen Teil der Vektoren und Matrizen Speicher       *
 *         vorhanden ist, so dass der schon belegte Speicher            *
 *         zurueckgegeben und auf den Speichermangel geeignet reagiert  *
 *         werden muss. Dies kostet viel Muehe und stellt eine haeufige *
 *         Fehlerquelle dar, wenn man es jedesmal neu formulieren muss. *
 *         Zur Vereinfachung dieser Arbeit wurde daher dieses C-Modul   *
 *         geschrieben. Es verwaltet alle zusammengehoerigen            *
 *         Speicheranforderungen fuer Vektoren und Matrizen in einer    *
 *         einfach verketteten Liste. Dazu werden dem Benutzer folgende *
 *         vier Funktionen zur Verfuegung gestellt:                     *
 *                                                                      *
 *         - vminit(),    das eine leere Vektor-Matrix-Liste erzeugt    *
 *                        und einen typlosen Listenanfangszeiger        *
 *                        liefert, mit dem alle weiteren Funktionen     *
 *                        arbeiten,                                     *
 *                                                                      *
 *         - vmalloc(),   das Speicher fuer einen neuen Vektor oder     *
 *                        eine neue Matrix anfordert, die Adresse in    *
 *                        die Liste einfuegt und zurueckliefert,        *
 *                                                                      *
 *         - vmcomplete() zur nachtraeglichen Pruefung, ob alle         *
 *                        bisher in der Liste vorgenommenen             *
 *                        Speicheranforderungen zum Erfolg fuehrten,    *
 *                        und                                           *
 *                                                                      *
 *         - vmfree(),    das den von einer Vektor-Matrix-Liste         *
 *                        beanspruchten Speicher vollstaendig freigibt. *
 *                                                                      *
 *         Ausserdem werden noch die sieben Makros                      *
 *                                                                      *
 *         - VEKTOR  (fuer REAL-Vektoren),                              *
 *         - VVEKTOR (fuer beliebige Vektoren),                         *
 *         - MATRIX  (fuer REAL-Matrizen),                              *
 *         - IMATRIX (fuer int-Matrizen),                               *
 *         - MMATRIX (fuer Matrizen von 4x4-Matrizen),                  *
 *         - UMATRIX (fuer untere REAL-Dreiecksmatrizen) und            *
 *         - PMATRIX (fuer Punktmatrizen im R3)                         *
 *                                                                      *
 *         exportiert, mit denen der Benutzer beim Aufruf von vmalloc() *
 *         den Typ der anzufordernden Datenstruktur waehlen kann.       *
 *                                                                      *
 *         Achtung: 1. Der von einer Vektor-Matrix-Liste                *
 *                     beanspruchte Speicher darf nur durch vmfree()    *
 *                     freigegeben werden!                              *
 *                  2. vmfree() gibt immer nur den gesamten schon       *
 *                     angeforderten Speicher frei, der zu einer Liste  *
 *                     gehoert, laesst sich also nicht auf einzelne     *
 *                     Vektoren oder Matrizen der Liste anwenden!       *
 *                                                                      *
 * Aufruf: Der Benutzer vereinbart einen typlosen Zeiger, der           *
 *         zuallererst durch einen Aufruf von vminit() initialisiert    *
 *         werden muss und von da an den einzigen gueltigen Zugang zur  *
 *         Speicherliste darstellt. Ueber diesen Zeiger koennen nun mit *
 *         Hilfe von vmalloc() Vektoren und Matrizen dynamisch angelegt *
 *         werden. Wurden alle Speicheranforderungen getaetigt, sollte  *
 *         man mit vmcomplete() pruefen, ob sie auch gelungen sind, und *
 *         dann entsprechend reagieren. Wenn die zur Liste gehoerenden  *
 *         Vektoren und Matrizen nicht mehr benoetigt werden, empfiehlt *
 *         es sich, denn davon beanspruchten Speicher durch Aufruf von  *
 *         vmfree() der Allgemeinheit wieder zur Verfuegung zu stellen. *
 *         Beispiel:                                                    *
 *             ...                                                      *
 *             void *vmblock;    /+ Anfang der Vektor-Matrix-Liste +/   *
 *             REAL *vektor1;    /+ REAL-Vektor mit n Elementen    +/   *
 *             int  *vektor2;    /+ int-Vektor mit n Elementen     +/   *
 *             REAL **matrix1;   /+ Matrix mit m Zeilen, n Spalten +/   *
 *             int  **matrix2;   /+ Matrix mit m Zeilen, n Spalten +/   *
 *             mat4x4 **mmat;    /+ Matrix mit m*n Elementen vom   +/   *
 *                               /+ Typ `mat4x4' (16 REAL-Zahlen)  +/   *
 *             REAL **umatrix;   /+ untere (n,n)-Dreiecksmatrix    +/   *
 *             REAL ***pmatrix;  /+ Matrix mit m*n Punkten im R3   +/   *
 *             ...                                                      *
 *             vmblock = vminit();                                      *
 *             vektor1 = (REAL *)vmalloc(vmblock, VEKTOR,  n, 0);       *
 *             vektor2 = (int *) vmalloc(vmblock, VVEKTOR, n,           *
 *                                       sizeof(int));                  *
 *             ...                                                      *
 *             matrix1 = (REAL **)  vmalloc(vmblock, MATRIX,  m, n);    *
 *             matrix2 = (int  **)  vmalloc(vmblock, IMATRIX, m, n);    *
 *             mmat    = (mat4x4 **)vmalloc(vmblock, MMATRIX, m, n);    *
 *             umatrix = (REAL ***) vmalloc(vmblock, UMATRIX, m, 0);    *
 *             pmatrix = (REAL ***) vmalloc(vmblock, PMATRIX, m, n);    *
 *             ...                                                      *
 *             if (! vmcomplete(vmblock))  /+ teilweise misslungen? +/  *
 *             {                                                        *
 *               vmfree(vmblock);          /+ Block ganz freigeben  +/  *
 *               return 99;                /+ Fehler melden         +/  *
 *             }                                                        *
 *             ...                                                      *
 *             vmfree(vmblock);                                         *
 *             ...                                                      *
 *                                                                      *
 * Programmiersprache: ANSI C                                           *
 * Compiler:           Borland C++ 2.0                                  *
 * Rechner:            IBM PS/2 70 mit 80387                            *
 * Autor:              Juergen Dietel, Rechenzentrum der RWTH Aachen    *
 * Datum:              DO 10. 9. 1992                                   *
 *                                                                      *
 ***********************************************************************/

#include <Gate/include/basis.h>                   /* wegen size_t, NULL, malloc, free, calloc,   */
/*       boolean, FALSE, TRUE, REAL, mat4x4    */
#include <Gate/include/vmblock.h>                 /* wegen vmalloc, vmcomplete, vmfree, vminit,  */
/*       VEKTOR, VVEKTOR, MATRIX, IMATRIX,     */
/*       MMATRIX, UMATRIX, PMATRIX             */

/*--------------------------------------------------------------------*/

typedef struct VML                                /* Element einer Vektor-Matrix-Liste      */
{
   void       *vmzeiger;                          /* Zeiger auf den Vektor bzw. die Matrix  */
   int        typ;                                /* Typ des Zeigers: Vektor oder Matrix    */
   /* (moegliche Werte: VEKTOR, VVEKTOR,     */
   /*                   MATRIX, IMATRIX,     */
   /*                   MMATRIX, UMATRIX,    */
   /*                   PMATRIX)             */
   size_t     groesse;                            /* im Ankerelement die Flagge, die eine   */
   /* misslungene Speicheranforderung        */
   /* anzeigt, sonst ungenutzt ausser bei    */
   /* Matrizen, wo `groesse' fuer die        */
   /* Zeilenanzahl "missbraucht" wird        */
   size_t     spalten;                            /* Spaltenanzahl bei Punktmatrizen, sonst */
   /* ungenutzt                              */
   struct VML *naechst;                           /* Zeiger auf das naechste Listenelement  */
} vmltyp;
/*.IX{vmltyp}*/

#define VMALLOC  (vmltyp *)malloc(sizeof(vmltyp)) /* Speicher fuer */
/*.IX{VMALLOC}*/
                                                  /* ein neues     */
                                                  /* Listenelement */
                                                  /* anfordern     */

#define LISTE    ((vmltyp *)vmblock)              /* zur Abkuerzung        */
/*.IX{LISTE}*/
/* der Schreibweise      */
#define MAGIC    410                              /* soll ein gueltiges    */
/*.IX{MAGIC}*/
/* Ankerelement anzeigen */

/*--------------------------------------------------------------------*/

void *vminit                                      /* eine leere Vektor-Matrix-Liste erzeugen ......*/
/*.IX{vminit}*/
(
void
)                                                 /* Adresse der Liste .................*/

/***********************************************************************
 * eine leere Vektor-Matrix-Liste erzeugen. Diese besteht aus einem     *
 * Ankerelement, das nur dazu benoetigt wird, die Speichermangelflagge  *
 * und einen magischen Wert fuer Plausibilitaetskontrollen aufzunehmen, *
 * der es ermoeglicht, eine nicht initialisierte Liste mit hoher        *
 * Wahrscheinlichkeit als ungueltig zu erkennen.                        *
 * Als Funktionswert wird die Adresse des Ankers zurueckgegeben, im     *
 * Fehlerfall natuerlich NULL. Um in den nachfolgenden Aufrufen von     *
 * vmalloc(), vmcomplete() und vmfree() pruefen zu koennen, ob der      *
 * uebergebene typlose Zeiger wirklich auf eine Vektor-Matrix-Liste     *
 * zeigt, wird die Komponente `typ' des Ankerelements dazu missbraucht, *
 * einen magischen Wert aufzunehmen, der ein gueltiges Ankerelement     *
 * anzeigen soll.                                                       *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * vmltyp, VMALLOC, MAGIC, NULL, malloc                                 *
 ***********************************************************************/

{
   vmltyp *liste;                                 /* Zeiger auf das Ankerelement der Liste */

   if ((liste = VMALLOC) == NULL)                 /* Speicher fuer den Anker anfordern */
      return NULL;                                /* misslungen? => Fehler melden      */
   liste->vmzeiger = NULL;                        /* damit vmfree() sich nicht vertut  */
   liste->typ      = MAGIC;                       /* einen gueltigen Anker anzeigen    */
   liste->groesse  = 0;                           /* noch kein Speichermangel          */
   liste->naechst  = NULL;                        /* noch kein Nachfolger              */

   return (void *)liste;
}


/*--------------------------------------------------------------------*/

static void matfree                               /* Speicher einer dynamischen Matrix freigeben ..*/
/*.IX{matfree}*/
(
void   **matrix,                                  /* [0..m-1,0..]-Matrix ...............*/
size_t m                                          /* Zeilenanzahl der Matrix ...........*/
)

/***********************************************************************
 * eine wie in matmalloc() erzeugte Matrix mit m Zeilen freigeben       *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * size_t, NULL, free                                                   *
 ***********************************************************************/

{
#ifdef FAST_ALLOC
   void *tmp;                                     /* kleinste Zeilenadresse             */
#endif
   if (matrix != NULL)                            /* Matrix vorhanden?                  */
   {
#ifndef FAST_ALLOC                          /* sichere, aber teuere Allokation?   */
      while (m != 0)                              /* den Speicher der Matrixelemente    */
         free(matrix[--m]);                       /* zeilenweise freigeben              */
#else                                       /* sparsamere Allokationsmethode?     */
      /* (setzt linearen Adressraum voraus) */
      for (tmp = matrix[0]; m != 0; )             /* den Zeiger mit der kleinsten  */
         if (matrix[--m] < tmp)                   /* Adresse suchen (noetig wegen  */
            tmp = matrix[m];                      /* moeglicher Vertauschung!)     */
      free(tmp);                                  /* die Speicherflaeche fuer alle      */
      /* Matrixelemente in einem Stueck     */
      /* freigeben                          */
#endif
      free(matrix);                               /* alle Zeilenzeiger freigeben        */
   }
}


/*--------------------------------------------------------------------*/

/***********************************************************************
 * Speicherplatz fuer eine rechteckige [0..m-1,0..n-1]-Matrix mit       *
 * Elementen vom Typ `typ' anfordern und die Anfangsadresse der Matrix  *
 * in `mat' ablegen, falls die Anforderung zum Erfolg fuehrte, sonst in *
 * `mat' NULL eintragen. Dabei wird fuer jede der m Zeilen der Matrix   *
 * ein eigener Zeiger angelegt, der auf n Matrixelemente verweist. Bei  *
 * Speichermangel wird die schon reservierte Teilmatrix freigegeben.    *
 * Falls vor der Uebersetzung das Makro FAST_ALLOC definiert wurde,     *
 * wird zwar auch noch mit m Zeilenzeigern gearbeitet, aber (nach einer *
 * Idee von Albert Becker) der Speicher fuer die m*n Matrixelemente in  *
 * einem Stueck angefordert und dann den einzelnen Zeilenzeigern        *
 * Adressen in dieser Speicherflaeche zugeordnet. Passend dazu gibt es  *
 * auch in matfree() einen FAST_ALLOC-Teil, wo man darauf achten muss,  *
 * dass die Zeilenzeiger seit dem Anlegen der Matrix vertauscht worden  *
 * sein koennten.                                                       *
 * Falls eine untere Dreiecksmatrix angelegt werden soll (umat != 0),   *
 * wird der Wert n ignoriert, da die Matrix quadratisch ist, und es     *
 * wird nur Speicher fuer m*(m+1)/2 REAL-Zahlen belegt (abgesehen von   *
 * den Zeilenzeigern).                                                  *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * size_t, NULL, calloc, matfree                                        *
 ***********************************************************************/

#ifndef FAST_ALLOC                                /* sichere, aber teuere Allokation?   */
#define matmalloc(mat, m, n, typ, umat) \
/*.IX{matmalloc}*/ \
{ \
   size_t j,                               /* laufender Zeilenindex */ \
      k;                               /* Elemente in Zeile j   */ \
   if ((mat = (typ **)calloc((m), sizeof(typ *))) != NULL) \
      for (j = 0; j < (m); j++) \
   { \
      k = (umat) ? (j + 1) : (n); \
      if ((((typ **)mat)[j] = (typ *)calloc(k, sizeof(typ))) == NULL) \
      { \
         matfree((void **)(mat), j); \
         mat = NULL; \
         break; \
      } \
   } \
}


#else                                             /* sparsamere Allokationsmethode?     */
/* (setzt linearen Adressraum voraus) */
#define matmalloc(mat, m, n, typ, umat) \
/*.IX{matmalloc}*/ \
{ \
   typ    *tmp;  /* Adresse des zusammenhaengenden Speicherbereichs, */ \
   /* der alle Matrixelemente enthaelt                 */ \
   size_t j,     /* laufender Zeilenindex                            */ \
      k,     /* Index fuer `tmp' auf die j. Zeile (Wert: j*n)    */ \
      l;     /* Groesse der Speicherflaeche: volle (m*n Elemente)*/ \
   /* oder untere (m*(m+1)/2 Elemente) Dreiecksmatrix  */ \
   if ((mat = (typ **)calloc((m), sizeof(typ *))) != NULL) \
   { \
      l = (umat) ? (((m) * ((m) + 1)) / 2) : ((m) * (n)); \
      if ((tmp = (typ *)calloc(l, sizeof(typ))) != NULL) \
         for (j = k = 0; j < (m); j++) \
            ((typ **)mat)[j]  = tmp + k, \
               k                += (umat) ? (j + 1) : (n); \
            else \
            { \
               free(mat); \
               mat = NULL; \
            } \
   } \
}
#endif

/*--------------------------------------------------------------------*/

static void pmatfree                              /* Speicher einer Punktmatrix freigeben ........*/
/*.IX{pmatfree}*/
(
void   ***matrix,                                 /* [0..m-1,0..n-1]-Matrix von Punkten */
size_t m,                                         /* Zeilenanzahl der Matrix ...........*/
size_t n                                          /* Spaltenanzahl der Matrix ..........*/
)

/***********************************************************************
 * eine wie in pmatmalloc() erzeugte Matrix mit m Zeilen und n Spalten  *
 * freigeben                                                            *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * size_t, NULL, free, matfree                                          *
 ***********************************************************************/

{
   if (matrix != NULL)                            /* Matrix vorhanden?               */
   {
      while (m != 0)                              /* den Speicher der Matrixelemente */
         matfree(matrix[--m], n);                 /* zeilenweise freigeben           */
      free(matrix);                               /* die Zeilenzeiger freigeben      */
   }
}


/*--------------------------------------------------------------------*/

static REAL ***pmatmalloc                         /* Speicher f. eine Punktmatrix anfordern */
/*.IX{pmatmalloc}*/
(
size_t m,                                         /* Zeilenanzahl der Matrix ...........*/
size_t n                                          /* Spaltenanzahl der Matrix ..........*/
)                                                 /* Adresse der Matrix ................*/

/***********************************************************************
 * Speicherplatz fuer eine [0..m-1,0..n-1,0..2]-Matrix mit Elementen    *
 * vom Typ REAL anfordern und ihre Anfangsadresse als Funktionswert     *
 * zurueckgeben, falls die Anforderung zum Erfolg fuehrte, sonst NULL.  *
 * Dabei wird fuer jede Zeile der Matrix ein eigener Zeiger angelegt.   *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * size_t, REAL, NULL, calloc, pmatfree, matmalloc                      *
 ***********************************************************************/

{
   REAL   ***matrix;                              /* Zeiger auf die Zeilenvektoren */
   size_t i;                                      /* laufender Zeilenindex         */

   matrix = (REAL ***)                            /* fuer jede der m Zeilen */
      calloc(m, sizeof(*matrix));                 /* einen Zeiger           */

   if (matrix == NULL)                            /* nicht genug Speicher?  */
      return NULL;                                /* Speichermangel melden  */

   for (i = 0; i < m; i++)                        /* fuer jeden Zeilenzeiger eine */
   {                                              /* (n,3)-REAL-Matrix anfordern  */
      matmalloc(matrix[i], n, 3, REAL, 0);
      if (matrix[i] == NULL)                      /* nicht genug Speicher?  */
      {
         pmatfree((void ***)matrix, i, 3);        /* reservierte Teilmatrix */
         /* freigeben              */
         return NULL;                             /* Speichermangel melden  */
      }
   }

   return matrix;
}


/*--------------------------------------------------------------------*/

void *vmalloc                                     /* dynamischen Vektor bzw. Matrix erzeugen ......*/
/*.IX{vmalloc}*/
(
void   *vmblock,                                  /* Adresse einer Vektor-Matrix-Liste  */
int    typ,                                       /* Art des Vektors/der Matrix ........*/
size_t zeilen,                                    /* Elementanzahl/Zeilenanzahl ........*/
size_t spalten                                    /* Spaltenanzahl/Elementgroesse ......*/
)                                                 /* Adresse des geschaffenen Objekts ..*/

/***********************************************************************
 * ein durch `typ' bestimmtes Element (Vektor oder Matrix), dessen      *
 * Groesse durch `zeilen' und `spalten' festgelegt wird, erzeugen und   *
 * vorne in die bei `vmblock' beginnende einfach verkettete Liste       *
 * einfuegen. Die Adresse des neuen Vektors bzw. der neuen Matrix wird  *
 * als Funktionswert zurueckgegeben. Bei einem REAL-Vektor (Typ VEKTOR) *
 * enthaelt der Parameter `zeilen' die Laenge, `spalten' wird nicht     *
 * benutzt. Beim Typ VVEKTOR (variabler Vektor) muss in `spalten' die   *
 * Groesse eines einzelnen Vektorelements stehen. Bei einer vollen      *
 * Matrix (Typ MATRIX, IMATRIX, MMATRIX oder PMATRIX) enthaelt `zeilen' *
 * die Zeilen- und `spalten' die Spaltenanzahl der Matrix. Bei einer    *
 * (quadratischen) unteren Dreiecksmatrix (Typ UMATRIX) enthaelt        *
 * `zeilen' die Zeilen- bzw. Spaltenanzahl der Matrix.                  *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * vmltyp, VMALLOC, LISTE, MAGIC, matmalloc, pmatmalloc, REAL, VEKTOR,  *
 * VVEKTOR, MATRIX, IMATRIX, MMATRIX, UMATRIX, PMATRIX, NULL, size_t,   *
 * malloc, calloc, mat4x4, matmalloc                                    *
 ***********************************************************************/

{
   vmltyp *element;                               /* Zeiger auf das neue Listenelement */

   if (LISTE      == NULL ||                      /* ungueltige Liste oder         */
      LISTE->typ != MAGIC)                        /* ungueltiges Ankerelement?     */
      return NULL;                                /* Fehler melden                 */

   if ((element = VMALLOC) == NULL)               /* neues Listenelement anfordern */
   {                                              /* misslungen? =>                */
      LISTE->groesse = 1;                         /* Speichermangel anzeigen       */
      return NULL;                                /* Fehler melden                 */
   }

   switch (typ)                                   /* Speicher fuer die gewuenschte Datenstruktur */
   {                                              /* anfordern (Vektor oder Matrix) und ihre     */
      /* Adresse in das neue Listenelement eintragen */

      case VEKTOR:                                /* ---------- REAL-Vektor?       ---------- */
         element->vmzeiger = calloc(zeilen, sizeof(REAL));
         break;

      case VVEKTOR:                               /* ---------- beliebiger Vektor? ---------- */
         element->vmzeiger = calloc(zeilen, spalten);
         break;

      case MATRIX:                                /* ---------- REAL-Matrix?       ---------- */
         matmalloc(element->vmzeiger, zeilen, spalten, REAL, 0);
         element->groesse  = zeilen;              /* fuer vmfree() unter groesse */
         break;                                   /* die Zeilenanzahl eintragen  */

      case IMATRIX:                               /* ---------- int-Matrix?        ---------- */
         matmalloc(element->vmzeiger, zeilen, spalten, int, 0);
         element->groesse  = zeilen;              /* fuer vmfree() unter groesse */
         break;                                   /* die Zeilenanzahl eintragen  */

      case MMATRIX:                               /* ---------- mat4x4-Matrix?     ---------- */
         matmalloc(element->vmzeiger, zeilen, spalten, mat4x4, 0);
         element->groesse  = zeilen;              /* fuer vmfree() unter groesse */
         break;                                   /* die Zeilenanzahl eintragen  */

      case UMATRIX:                               /* ---------- untere Dreiecksmatrix? ------ */
         matmalloc(element->vmzeiger, zeilen, 0, mat4x4, 1);
         element->groesse  = zeilen;              /* fuer vmfree() unter groesse */
         break;                                   /* die Zeilenanzahl eintragen  */

      case PMATRIX:                               /* ---------- Punktmatrix?       ---------- */
         element->vmzeiger = (void *)pmatmalloc(zeilen, spalten);
         element->groesse  = zeilen;              /* fuer vmfree() unter groesse */
         element->spalten  = spalten;             /* und spalten die Zeilen- und */
         break;                                   /* Spaltenanzahl eintragen     */

      default:                                    /* ---- ungueltiger Datenstrukturtyp? ----  */
         element->vmzeiger = NULL;                /* Nullzeiger eintragen        */
   }

   if (element->vmzeiger == NULL)                 /* kein Speicher da fuers Objekt? */
      LISTE->groesse = 1;                         /* Das merken wir uns.            */

   element->typ = typ;                            /* Datenstrukturtyp im         */
   /* Listenelement notieren      */
   element->naechst = LISTE->naechst;             /* neues Element einfuegen vor */
   /* dem ersten Element und ...  */

   LISTE->naechst = element;                      /* ... hinter dem Ankerelement */

   return element->vmzeiger;                      /* neue Vektor/Matrix-Adresse  */
}                                                 /* zurueckgeben                */


/*--------------------------------------------------------------------*/

boolean vmcomplete                                /* Vektor-Matrix-Liste auf Speichermangel testen */
/*.IX{vmcomplete}*/
(
void *vmblock                                     /* Adresse der Liste .................*/
)                                                 /* kein Speichermangel? ..............*/

/***********************************************************************
 * Hier wird einfach nur der negierte Wert der Flagge im Ankerelement   *
 * der bei `vmblock' beginnenden Liste zurueckgegeben und somit         *
 * gemeldet, dass alle Speicheranforderungen in der Liste gelungen sind *
 * (TRUE) oder nicht (FALSE).                                           *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * LISTE                                                                *
 ***********************************************************************/

{
   return LISTE->groesse ? FALSE : TRUE;
}


/*--------------------------------------------------------------------*/

void vmfree                                       /* Speicher einer Vektor-Matrix-Liste freigeben  */
/*.IX{vmfree}*/
(
void *vmblock                                     /* Adresse der Liste .................*/
)

/***********************************************************************
 * saemtlichen dynamischen Speicher der bei `vmblock' beginnenden Liste *
 * freigeben                                                            *
 *                                                                      *
 * benutzte globale Namen:                                              *
 * =======================                                              *
 * vmltyp, LISTE, MAGIC, matfree, pmatfree, VEKTOR, VVEKTOR, MATRIX,    *
 * IMATRIX, MMATRIX, UMATRIX, PMATRIX, NULL, free                       *
 ***********************************************************************/

{
   vmltyp *hilf;                                  /* Zwischenspeicher fuer einen Zeigerwert */

   if (LISTE == NULL)                             /* ungueltige Liste?         */
      return;                                     /* nichts tun                */

   if (LISTE->typ != MAGIC)                       /* ungueltiges Ankerelement? */
      return;                                     /* nichts tun                */

   for ( ; LISTE != NULL; vmblock = (void *)hilf)
   {

      switch (LISTE->typ)
      {
         case VEKTOR:
         case VVEKTOR: if (LISTE->vmzeiger != NULL)
         free(LISTE->vmzeiger);
         break;
         case MATRIX:
         case IMATRIX:
         case MMATRIX:
         case UMATRIX: matfree((void **)LISTE->vmzeiger,
            LISTE->groesse);
         break;
         case PMATRIX: pmatfree((void ***)LISTE->vmzeiger,
            LISTE->groesse, LISTE->spalten);
      }

      hilf = LISTE->naechst;                      /* Nachfolgerzeiger retten   */
      free(LISTE);                                /* Listenelement freigeben   */
   }
}


/* ------------------------- ENDE vmblock.c ------------------------- */
