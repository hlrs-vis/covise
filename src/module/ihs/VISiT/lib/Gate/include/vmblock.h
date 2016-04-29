
/*.FE{C 0.2}{Dynamische Vektoren und Matrizen}
            {Dynamische Vektoren und Matrizen}*/

/* --------------------- DEKLARATIONEN vmblock.h -------------------- */

#ifndef VMBLOCK_H_INCLUDED
#define VMBLOCK_H_INCLUDED

/***********************************************************************
 * symbolische Namen, mit denen der Benutzer beim Aufruf von vmalloc()  *
 * den Typ der anzufordernden dynamischen Datenstruktur waehlen kann    *
 ***********************************************************************/

#define VEKTOR   0                                /* fuer einen REAL-Vektor            */
/*.IX{VEKTOR}*/
#define VVEKTOR  1                                /* fuer einen Vektor mit Elementen   */
/*.IX{VVEKTOR}*/
/* von angegebener Groesse           */
#define MATRIX   2                                /* fuer eine REAL-Matrix             */
/*.IX{MATRIX}*/
#define IMATRIX  3                                /* fuer eine int-Matrix              */
/*.IX{IMATRIX}*/
#define MMATRIX  4                                /* fuer eine Matrix von 4x4-Matrizen */
/*.IX{PMATRIX}*/
/* (mit Elementen vom Typ `mat4x4')  */
#define UMATRIX  5                                /* fuer eine untere Dreiecksmatrix   */
/*.IX{UMATRIX}*/
#define PMATRIX  6                                /* fuer eine Punktmatrix im R3       */
/*.IX{PMATRIX}*/

/***********************************************************************
 * Operationen auf der Vektor-Matrix-Liste                              *
 ***********************************************************************/

void *vminit                                      /* eine leere Vektor-Matrix-Liste erzeugen ......*/
(
void
);                                                /* Adresse der Liste .................*/

void *vmalloc                                     /* dynamischen Vektor bzw. Matrix erzeugen ......*/
(
void   *vmblock,                                  /* Adresse einer Vektor-Matrix-Liste  */
int    typ,                                       /* Art des Vektors/der Matrix ........*/
size_t zeilen,                                    /* Elementanzahl/Zeilenanzahl ........*/
size_t spalten                                    /* Spaltenanzahl/Elementgroesse ......*/
);                                                /* Adresse des geschaffenen Objekts ..*/

boolean vmcomplete                                /* Vektor-Matrix-Liste auf Speichermangel testen */
(
void *vmblock                                     /* Adresse der Liste .................*/
);                                                /* kein Speichermangel? ..............*/

void vmfree                                       /* Speicher einer Vektor-Matrix-Liste freigeben  */
(
void *vmblock                                     /* Adresse der Liste .................*/
);
#endif

/* ------------------------- ENDE vmblock.h ------------------------- */
