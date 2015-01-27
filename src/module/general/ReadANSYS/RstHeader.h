/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RST_HEADER_H
#define _RST_HEADER_H

struct RstHeader
{
    int fun12_; // Unit number, hier immer 12
    int maxnodes_; // Anzahl der Knoten
    int usednodes_; // Anzahl verwendeter Knoten
    int maxres_; // max. Anzahl von Datensätzen in diesem File
    int numdofs_; // Anzahl von Freiheitsgraden pro Knoten
    int maxelement_; // Anzahl von Finiten Elementen im gesamten Modell
    int numelement_; // Anzahl von Finiten Elementen
    int analysis_; // Art der Berechnung
    int numsets_; // Anzahl der Datensätze im File
    int ptr_eof_; // Offset zum Fileende
    int ptr_dsi_; // Offset zur Data Step Index Tabelle
    int ptr_time_; // Offset zur Tabelle der Zeiten
    int ptr_load_; // Offset zur Tabelle der Load Steps
    int ptr_elm_; // Offset zur element equivalenz Tabelle
    int ptr_node_; // Offset zur nodal equivalenz Tabelle
    int ptr_geo_; // Offset zur Geometriebeschreibung
    //  int res1[3];        // reserviert
    int units_; // Einheitensystem (z.B. 1=SI)
    int numsectors_; // Zahl der Sektoren für Symmetrie
    //  int res2;           // reserviert
    long long ptr_end_; // 64-Bit file length
    //  int res3[17];       // reserviert, inclusive Lead-out
};
#endif
