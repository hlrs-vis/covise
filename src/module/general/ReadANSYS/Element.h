/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ELEMENT_H_
#define _ELEMENT_H_

struct Element
{
    int material_; // Materialnummer
    int type_; // ETY-Nummer
    int real_; // RC Nummer
    int section_; // Nummer der zugehörigen Section
    int coord_; // Element KS
    int death_; // Death Flag: 1=alive, 0=death
    int solidmodel_; // Solid Model Referenz
    int shape_; // Formidentifier
    int num_; // Elementnummer
    int *nodes_; // Array mit den zugehörigen Knoten
    // Jetzt noch Daten aus Geometrieheader
    int anznodes_; // länge des nodes-Arrays
    ~Element()
    {
        delete[] nodes_;
    }
};
#endif
