/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ETYPE_H_
#define _ETYPE_H_

struct EType
{
    int id_; // Referenz Nummer des Elements
    int routine_; // Element routine Nummer
    int keyops_[12]; // nicht näher definierte Optionen
    int dofpernode_; // Anzahl der DOFs pro node
    int nodes_; // Anzahl der Knoten für dieses Element
    int nodeforce_; // Anzahl der Knoten mit Kräften
    int nodestress_; // Anzahl der Knoten mit Schubspannungen
};
#endif
