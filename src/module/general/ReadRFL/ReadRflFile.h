/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __READRFL_HPP__
#define __READRFL_HPP__

#include "binheader.h"
#include <util/coviseCompat.h>

struct ELEMENT
{
    int material; // Materialnummer
    int type; // ETY-Nummer
    int real; // RC Nummer
    int section; // Nummer der zugehörigen Section
    int coord; // Element KS
    int death; // Death Flag: 1=alive, 0=death
    int solidmodel; // Solid Model Referenz
    int shape; // Formidentifier
    int num; // Elementnummer
    int *nodes; // Array mit den zugehörigen Knoten
    // Jetzt noch Daten aus Geometrieheader
    int anznodes; // länge des nodes-Arrays
};

struct ETY
{
    int id; // Referenz Nummer des Elements
    int routine; // Element routine Nummer
    int keyops[12]; // nicht näher definierte Optionen
    int dofpernode; // Anzahl der DOFs pro node
    int nodes; // Anzahl der Knoten für dieses Element
    int nodeforce; // Anzahl der Knoten mit Kräften
    int nodestress; // Anzahl der Knoten mit Schubspannungen
};

struct NODE
{
    double id; // Da muss man später nix mehr konvertieren!
    double x, y, z;
    double thxy, thyz, thzx;
};

struct DOFROOT
{
    int dataset; // Nummer des Dataset aus dem die Ergebnisse stammen
    int typ; // Typ nach File definition
    bool exdof; // Extension oder normal
    char name[20]; // Klartextname
    int anz; // Anzahl Elemente in data
    double *data; // Pointer auf daten
    DOFROOT *next; // Pointer auf nächste Liste oder NULL
};

struct GEOMETRYHEADER
{
    int maxety; // Max. Element -Typ Nummer
    int maxrl; // Max. RealConstant Nummer
    int nodes; // Anzahl der definierten Knoten im Modell
    int elements; // Anzahl definierter Elemente im Modell
    int maxcoord; // Max. Kooordinatensystem-Referenz
    int ptr_ety; // Offset zur Tabelle der element typen
    int ptr_rel; // Offset zur Tabelle der RealConstants
    int ptr_nodes; // Offset zu den Punktkoordinaten
    int ptr_sys; // Offset zum Koordinatensystem-Index
    int ptr_eid; // Offset zu der Tabelle der Element-Indices
    int ptr_mas; // Offset zur diagonal mass matrix
    int coordsize; // Anzahl der Daten, die ein KS beschreiben (24)
    int elemsize; // Max. Anzahl von Punkten, die ein Element haben kann
    int etysize; // Anzahl der items (?) die  einen Elementtyp beschreiben
    int rcsize; // Anzahl der items, die einen RC definieren
};

struct SOLUTIONHEADER
{
    int numelements; // Anzahl Elemente
    int numnodes; // Anzahl der Knoten
    unsigned int mask; // Bitmask
    int loadstep; // eben jener
    int iteration; // Iterationsnummer
    int sumiteration; // cummulative Iterationsnummer
    int numreact; // Anzahl der Reaktionskräfte
    int maxesz; // nicht dokumentiert
    int nummasters; // Anzahl der Master (?)
    int ptr_nodalsol; // Offset zur nodal solution
    int ptr_elemsol; // Offset zur element solution
    int ptr_react; // Offset zu den Reaktionskräften
    int ptr_masters; // Offset zu den Masters(?)
    int ptr_bc; // Offset zu den Boundary conditions
    int extrapolate; // 0=move, 1=extra unless non-linear, 2=extra always
    int mode; // Mode number of harminic loads
    int symmetry; // für harmonische Last ;)
    int complex; // 0=real, 1=complex
    int numdofs; // Anzahl der DOFs in diesem Datensatz
    int *dof; // DOF Referenzzahl, muss numdofs lang sein
    int *exdof; // Eigenbau: enthält die exdofs (numexdofs lang)
    int changetime; // compact: letzte Veränderung
    int changedate; // compact: letzte Veränderung
    int changecount; // wie oft wurde der Datensatz geändert?
    int soltime; // compact: Ergebniszeit
    int soldate; // compact: Ergebnisdatum
    int ptr_onodes; // Offset zur Liste der geordneten Knoten (load case)
    int ptr_oelements; // Offset zur Liste der geordneten Elemente (load case)
    int numexdofs; // Anzahl zusätzlicher DOFs für Flotran
    int ptr_extra_a; // Offset zurm EXA-Header
    int ptr_extra_t; // Offset zum EXT_Header
    char title[80];
    char subtitle[80];
    long long offset; // Fileoffset
    // Jetzt noch ein paar Daten aus dem Double Datensatz
    double time; // Zeitpunkt bei transienten Daten
};

struct RSTHEADER
{
    int fun12; // Unit number, hier immer 12
    int maxnodes; // Anzahl der Knoten
    int usednodes; // Anzahl verwendeter Knoten
    int maxres; // max. Anzahl von Datensätzen in diesem File
    int numdofs; // Anzahl von Freiheitsgraden pro Knoten
    int maxelement; // Anzahl von Finiten Elementen im gesamten Modell
    int numelement; // Anzahl von Finiten Elementen
    int analysis; // Art der Berechnung
    int numsets; // Anzahl der Datensätze im File
    int ptr_eof; // Offset zum Fileende
    int ptr_dsi; // Offset zur Data Step Index Tabelle
    int ptr_time; // Offset zur Tabelle der Zeiten
    int ptr_load; // Offset zur Tabelle der Load Steps
    int ptr_elm; // Offset zur element equivalenz Tabelle
    int ptr_node; // Offset zur nodal equivalenz Tabelle
    int ptr_geo; // Offset zur Geometriebeschreibung
    //  int res1[3];        // reserviert
    int units; // Einheitensystem (z.B. 1=SI)
    int numsectors; // Zahl der Sektoren für Symmetrie
    //  int res2;           // reserviert
    long long ptr_end; // 64-Bit file length
    //  int res3[17];       // reserviert, inclusive Lead-out
};

class READRFL
{
public:
    BINHEADER header;
    RSTHEADER rstheader;
    SOLUTIONHEADER solheader;
    DOFROOT *dofroot;
    NODE *node;
    ETY *ety;
    ELEMENT *element;
    FILE *rfp; // Result file pointer

    int anznodes;
    int anzety;
    int anzelem;

    int *nodeindex;
    int *elemindex;

    double *timetable;

    // interne Methoden
    int Reset(void);
    int OpenFile(char *); // Liest header, setzt rfp
    int SwitchEndian(int); // Dreht die Byte-Folge um
    double SwitchEndian(double); // Dreht die Byte-Folge um
    DOFROOT *CreateNewDOFList(void); // Erstellt eine neue DOF-Liste
    int GetNodes(void);
    int ReadSHDR(int); // liest den Solheader
    int GetDataset(int); // Liste die Ergebnisdaten

    READRFL(void);
    ~READRFL();

    int Read(char *, int);
    int WriteData(char *); // Schreibt
    double GetTime(int); // liefert den Zeitwert für die angegebene Lösung
};
#endif
