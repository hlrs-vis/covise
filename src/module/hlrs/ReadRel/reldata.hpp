/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __RELDATA__
#define __RELDATA__

#include <string.h>
#include <time.h>
#include "usefullroutines.h"

#ifndef MAXINT
#define MAXINT 0xffffffff
#endif

#define VON 0
#define NACH 1

enum ELEMENTTYPE
{
    ELEM_NODE,
    ELEM_EDGE,
    ELEM_FACE,
    ELEM_CELL,
    ELEM_NONE,
    ELEM_ALL,
    ANZ_ELEM
};
enum SMOOTHTYP
{
    SM_NONE,
    SM_NORMAL,
    SM_DOUBLE,
    SM_QUALITY,
    SM_GRID,
    ANZ_SMOOTH
};
enum DATATYP
{
    DATA_X = 1,
    DATA_Y = 2,
    DATA_Z = 3,
    DATA_VAL = 11,
    DATA_IGNORE = 0,
    DATA_INDEX = 13,
    DATA_THICKNESS = 15
};
enum VECTOR
{
    X = 0,
    Y = 1,
    Z = 2,
    B = 3
};
enum VERBOSE
{
    VER_NORMAL,
    VER_MAX,
    VER_DEBUG,
    VER_NONE,
    VER_DATAOUT,
    ANZ_VERBOSE
};
enum FORMAT
{
    FMT_ANSYS,
    FMT_ANSYS_SKRIPT,
    FMT_VRML,
    FMT_STL,
    FMT_BINSTL,
    FMT_GAMBIT,
    FMT_ASCII,
    FMT_POVRAY,
    ANZ_FORMAT
};
enum SORTTYPE
{
    SORT_MERGE,
    SORT_BUBBLE,
    SORT_CLOCKWISE,
    SORT_COUNTERCLOCKWISE
};

// Elementflags
enum FLAG
{
    FLAG_USED = 0x01, // Wird dieses Element von einem höheren benutzt?
    FLAG_DELETE = 0x02, // Dieses Element kann gelöscht werden?
    FLAG_NORMAL = 0x04, // Dieses Element besitzt einen Normalen-Datensatz
    FLAG_DISCON = 0x08, // Dieses Element ist disconnected worden und hat somit seine Verbindung nach UNTEN verlohren
    FLAG_OPT = 0x10, // Dieses Element ist optimiert worden
    FLAG_AREA = 0x20 // Dieses Element gehört in ein Gebiet (GETAREA)
};

//=========================================================================================
// Erlaeuterungen
//=========================================================================================
//
// Der Datensatz ist folgendermassen aufgebaut:
// Es gibt vier verschiedene Ebenen, die aus Elementen aufgebaut sind. Jedes Element
// Besteht aus diversen organisatorischen Daten (z.B. id, flags,typ=ELEMENTTYPE,etc) und
// einigen Listen.
// Die .element-Liste besteht aus einer Anzahl (e_anz) von Daten des Typs PTR_ELEMENT
// (pointer to element), die wiederum auf ein anderes Element zeigt und dessen id speichert.
// Diese Verbindung könnte man mit "Besteht aus" oder "ist SUB-Element von mir" bezeichnen.
// Die Liste .data besteht aus einer Anzahl (d_anz) Daten, die jeweils einen typ und einen
// Wert besitzen. Der typ bezeichnet dabei die Art des wertes und wird aus dem String
// "! property_numbers= (...)" erzeugt.
// Die .up-Liste ist (im Gegensatz zu den Arrays von .element und .data) eine Kette vom Typ
// ECHAIN (element chain), dem allgemeinen Listentyp von ELEMENT (wird auch von diversen Methoden
// verwendet!). Die up-Liste könnte man ab besten mit "ist SUPER_Element von mir" bezeichnen.
// Die aufgeführten Elemente sind solche, die das aktuelle Ellement beinhalten. Als
// Umkehrschluss könnte man sagen: Ein Element ohne up-Liste ist entweder ein Element höchster
// Ordnung (Ordnungsreihenfolge von hoch->tief: Zelle, Fläche,Kante,Knoten), oder ein unbenutzes
// Element. Unbenutzte Elemente sollten das flag FLAG_USED mittels ChangeFlag() löschen!
// Die Verpointerungen zwischen den Elementen haben sowohl Vorteile, als auch Nachteile:
// Vorteil ist der schnelle Zugriff ohne Suchen auf alle Elemente in allen Richtungen, Nachteil die
// Unübersichtlichkeit und die Probleme beim kopieren von Elementen (Die Pointer der anderen
// Elemente werden beim kopieren mittels CopyElement() nicht mitverändert!). Es gibt zwei Methoden
// die die Arbeit mit diesen Elementen vereinfachen: Disconnect() und Reconnect().
// Wird eine Elementschicht verändert (z.B. kopiert), sollte sie mit Disconnect von den anderen
// Ebenen gelöst werden (dabe werden alle Pointer auf die Elementschicht gelöscht), und nach der
// Operation mit Reconnect wieder hergestellt werden.

struct PTR_ELEM
{
    int id;
    struct ELEMENT *ptr;
}; // Array von Elementpointern + Richtung

struct PTR_DATA
{
    int typ;
    double value;
}; // Array von Daten

struct ECHAIN
{
    int id; // ist element->id, somit ID des gepointeten Elements
    ECHAIN *next;
    ELEMENT *element;
};

struct ELEMENT
{
    int id; // Index des Elements
    unsigned int flags;
    int e_anz; // Anzahl der verwalteten Elemente
    int d_anz; // Anzahl der verwalteten Daten
    ELEMENTTYPE typ; // Typ dieses Elements
    PTR_ELEM *element; // Array der niedrigeren Elemente
    PTR_DATA *data; // Array der Daten
    ECHAIN *up; // Liste von Elementen einer Ordnung höer, die dieses Element benutzen
};

// Test: für Deletedoubleelement(ELEM_NODE)

struct NODELIST
{
    int idx;
    double x, y, z;
    NODELIST *next;
};

class RELDATA
{
    //  protected:
public:
    // Variablen
    int nextnodeid; // zentrale ID-Vergabe. Wird von ReadElement gesetzt
    int nextedgeid; // und dann mit der GetNextID() angesteuert
    int nextfaceid; // Alle Elemente mit CreateElement erstellen!
    int nextcellid;

    int startnodes; // Für Kompressionsberechnung
    int startedges;
    int startfaces;
    int startcells;

    int anz_eknoten; // oder -1 für undefiniert
    int anz_ekanten; // oder -1 für undefiniert
    int anz_eflaechen; // oder -1 für undefiniert
    int anz_ezellen; // oder -1 für undefiniert

    struct ELEMENT *eknoten;
    struct ELEMENT *ekante;
    struct ELEMENT *eflaeche;
    struct ELEMENT *ezelle;

    // Routinen für das Datenfeld
    int Reconnect(ELEMENTTYPE); // Erstellt die Verbindungen zwischen den Elementen
    int Disconnect(ELEMENTTYPE); // Löst die Verbindung zwischen den Elementen
    int CreateUplinks(ELEMENT *); // Trägt uplinks in alle von diesem Elem. gepointerten Elemente ein
    int *GetNodes(ELEMENT *, ECHAIN *neukanten = NULL); // liefert eine id-Liste der Knoten des Elements zurück
    ECHAIN *GetFaces(ELEMENT *); // liefert eine ECHAIN-Liste der Flächen des Elements zurück
    ELEMENT *GetElementPtr(int, ELEMENTTYPE); // Liefert den Pointer auf ein Element des angegeben Index'
    ELEMENT *CopyElement(ELEMENT *, ELEMENT *); // Kopiert die Flächeninformationen
    int ClearElement(ELEMENT *); // Löscht alle Pointerdaten aus einem Element
    // Merge wandert nach extern!
    //    int Merge(ECHAIN*,ELEMENTTYPE);             // Fügt einem Elementtyp die angegebenen Elemente hinzu

    // Funktionen im Raum
    int CalculatePhongNormals(void); // Nötig für POV_Ray Ausgabe: Erstellt Vectoren an Randpunkten
    int BreakDownFaces(int); // zerlegt eine Fläche in kleinere
    int SurfaceSmooth(SMOOTHTYP, int); // Oberflächenglättung
    int CoarseFaces();
    int Extrude(int, double, BOOL killfaces = FALSE);
    int OptimizeFaces(void);
    ECHAIN *GetNeighbours(ELEMENT *, int);
    double *GetVector(ELEMENT *); // liefert den Vector einer Kante und seinen Betrag
    double *GetAngles(ELEMENT *); // liefert die inneren Winkel einer Fläche als Liste
    double *GetFaceNormal(ELEMENT *); // liefert die nichtnormierte Normale einer Fläche
    double GetFaceAngle(ELEMENT *, ELEMENT *); // liefert den Winkel zwischen Normalen von zwei Flächen
    double *GetCenterPoint(ELEMENT *); // Liefert den geometrischen Mittelpunkt einer Fläche
    double *GetBalancePoint(ELEMENT *); // Liefert den Schwerpunkt einer Fläche

    // Service Routinen
    int GetFlagUsage(int, ELEMENTTYPE); // liefert die Zahl der Elemente zurück, die diese Flag-Kombination benutzen
    ECHAIN *GetArea(ELEMENT *, ELEMENT *, double, int);
    int GetNextID(ELEMENTTYPE); // vergibt einen neuen ID für einen Elementtyp;
    int GetEdgeID(int, int, ECHAIN *neukanten = NULL);
    int GetFaceID(int *, int, ECHAIN *neuflaechen = NULL, ECHAIN *neukanten = NULL);
    int GetDataNameOrder(char *, int **); // Liefert eine int Tabelle mit Reihenfolge
    int GetDataType(int); // liefert den Variablen-Typ zu einem Property
    int GetDataCount(char *, int *, int *, int *);
    ELEMENTTYPE GetPrevElementType(ELEMENTTYPE);
    ELEMENTTYPE GetNextElementType(ELEMENTTYPE);
    int ChangeFlag(ELEMENT *, unsigned int, BOOL, BOOL forlower = FALSE); // Ändert das Flag bei einem oder allen unteren Elementen
    int ChangeFlag(ELEMENTTYPE, unsigned int, BOOL, BOOL forlower = FALSE); // Ändert das Flag bei einer Elementebene sonst wie oben
    int GetMaxID(ELEMENTTYPE);

    // löschen und erstellen
    int *CreateArea(ELEMENT *, double, const double, const double); // Liefert eine Kantenliste, die eine bestimmte Fläche umschiesst
    ECHAIN *CreateCell(int *, int *, int);
    ELEMENT *CreateElement(ELEMENTTYPE, BOOL newid = TRUE); // Erstellt ein Element mit neuem ID etc
    int RefineMesh(double); // Verfeinert das Gitter mit einem Gradientenverfahren
    int DeleteUnused(ELEMENTTYPE);
    int DeleteEChain(ECHAIN *, BOOL);
    BOOL DeleteElement(ELEMENT *);
    int DeleteDoubleElement(ELEMENTTYPE); // Löscht mehrfach vorhandene Elemente, eins bleibt übrig
    int KillDoubleElement(ELEMENTTYPE); // Löscht alle mehrfach vorhandene Elemente, keins bleibt übrig
    int *DeleteDoubleIDs(int *, int *); // Löscht doppelte IDs in einem iD-Feld
    int *KillDoubleIDs(int *, int *); // löscht alle IDs die mindestens doppelt vorkommen

    // Suchen
    int ElementBinSearch(int, int, int, ELEMENTTYPE); // Binsearch für Elemente
    int BinSearch(int, int, int, int *); // Binsearch für IntegerArrays

    // sortieren
    ECHAIN *SortEChain(ECHAIN *);
    clock_t SortElement(ELEMENTTYPE, SORTTYPE); // Sortiert ein Elementfeld nach dem ID
    ELEMENT *MergeSort(ELEMENT *, int); // Mergesort, über SortElement ansteuern!
    int BubbleSort(ELEMENTTYPE); // Bubbelsort, über SortElement ansteuern!
    int BubbleSort(int *, int, BOOL ignorevz = FALSE); // Bubbelsort, integer Version, Frei!
    int *MergeSort(int *, int, BOOL ignoresgn = FALSE); // Mergesort für Integer, frei ansteuerbar
    NODELIST *MergeSort(NODELIST *, int, int); // Mergesort für NODELISTs, nach X,Y,Z möglich
    int SortEdges(int *, int, SORTTYPE, ECHAIN *newedge = NULL); // Gegen oder mit dem Uhrzeigersinn
    int ReIndex(ELEMENTTYPE);

    // IO Routinen
    int ReadElementData(char *, ELEMENTTYPE, int *, int); // Testversion

    // Testroutinen (Beta)
    BOOL DoWeNeed(int, int *, int); // Sucht einen ID in einem ID-Feld
    int FindMultipleEdges2(void);
    int Errorscan(void);
    int MarkUsedElements();
    int MeltSmallFaces(void);
    int GetAreaSize(ELEMENT *);
    //    ECHAIN *CutGeometryBy(int,double);
    int MeltSmallTriangles(void);
    int SwapEdges(void);
    int LineCheck(void);

    RELDATA(); // Konstruktor
    ~RELDATA(); // Destruktor
    int ReadData(char *); // öffnet einen Datensatz
    int PrintStatement(void); // Gibt eine Gesamtübersicht am Bildschirm aus

    // Testroutinen extern
    ECHAIN *CutGeometryBy(int, double);
    int Merge(ECHAIN *, ELEMENTTYPE); // Fügt einem Elementtyp die angegebenen Elemente hinzu

    //    int WriteAs(char *, FORMAT);                // Schreibt daten ins gewünschte Format
};

/*
 Removed code

//    int anz_knoten;                             // oder -1 für undefiniert
//    int anz_kanten;                             // oder -1 für undefiniert
//    int anz_flaechen;                           // oder -1 für undefiniert
//    int anz_zellen;                             // oder -1 für undefiniert

//    struct KNOTEN *knoten;                      // Tabelle mit Knoten
//    struct KANTE *kante;                        // Tabelle mit Kanten
//    struct FLAECHE *flaeche;                    // Tabelle mit Flächen

//    KNOTEN* GetNodePtr(int);                    // Liefert den Pointer auf den Knoten des angegeben Index'


//    KANTE* GetEdgePtr(int);                     // Liefert den Pointer auf den Kanten des angegeben Index'
//    int NodeBinSearch(int,int,int);             // Binsearch für Knoten
//    int EdgeBinSearch(int,int,int);             // Binsearch für Kanten
//    KNOTEN*  CopyStruct(KNOTEN*, KNOTEN*);      // Kopiert die Knoteninformation
//    KANTE*   CopyStruct(KANTE*, KANTE*);        // Kopiert die Kanteninformation
//    FLAECHE* CopyStruct(FLAECHE*, FLAECHE*);    // Kopiert die Flächeninformationen
//    int ReadElementData(char *, ELEMENTTYPE);   // Liest die Daten einer Elementgruppe
//    int FindMultipleEdges(void);
//    int GetFaceID2(int *,int,ECHAIN *neuflaechen=NULL);


//    int ReadNodeData(char *);                   // liest die Knotendaten aus dem File
//    int ReadEdgeData(char *);                   // liest die Kantendaten aus dem File
//    int ReadFaceData(char *);                   // liest die Flächendaten aus dem File


struct KNOTEN
{
  int id;                         // logische Nummer des Knotens
  int anz;                        // Anzahl der Elemente je Knoten (x,y,z,val,...)
  int *types;                     // Typen der Knoten, Reihenfolge wie in data
  double *data;                   // Werte (achtung: bisher alle double) FIXME
};

struct KANTE
{
  int id;
  int anz;
  int *sign;
  struct KNOTEN **knoten;
};

struct FLAECHE
{
  int id;
  int anz;
  int *sign;
  struct KANTE **kanten;
};  



*/

#endif
