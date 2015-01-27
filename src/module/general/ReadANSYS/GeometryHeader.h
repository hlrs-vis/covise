/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GEOMETRY__HEADER_H_
#define _GEOMETRY__HEADER_H_

struct GeometryHeader
{
    int maxety_; // Max. Element -Typ Nummer
    int maxrl_; // Max. RealConstant Nummer
    int nodes_; // Anzahl der definierten Knoten im Modell
    int elements_; // Anzahl definierter Elemente im Modell
    int maxcoord_; // Max. Kooordinatensystem-Referenz
    int ptr_ety_; // Offset zur Tabelle der element typen
    int ptr_rel_; // Offset zur Tabelle der RealConstants
    int ptr_nodes_; // Offset zu den Punktkoordinaten
    int ptr_sys_; // Offset zum Koordinatensystem-Index
    int ptr_eid_; // Offset zu der Tabelle der Element-Indices
    int ptr_mas_; // Offset zur diagonal mass matrix
    int coordsize_; // Anzahl der Daten, die ein KS beschreiben (24)
    int elemsize_; // Max. Anzahl von Punkten, die ein Element haben kann
    int etysize_; // Anzahl der items (?) die  einen Elementtyp beschreiben
    int rcsize_; // Anzahl der items, die einen RC definieren
};
#endif
