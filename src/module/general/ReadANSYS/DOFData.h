/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DOF_DATA_H_
#define _DOF_DATA_H_

struct DOFData
{
public:
    int dataset_; // Nummer des Dataset aus dem die Ergebnisse stammen
    int typ_; // Typ nach File definition
    bool exdof_; // Extension oder normal
    char name_[20]; // Klartextname
    int anz_; // Anzahl Elemente in data
    double *data_; // Pointer auf daten
    double *displacements_; // Pointer auf daten
    int nodesdataanz_;
    int *nodesdata_; // Pointer auf daten
    DOFData()
    {
        data_ = NULL;
        nodesdata_ = NULL;
        displacements_ = NULL;
    }
    ~DOFData()
    {
        clean();
    }

private:
    void clean()
    {
        delete[] data_;
        delete[] displacements_;
        delete[] nodesdata_;
        data_ = NULL;
        displacements_ = NULL;
        nodesdata_ = NULL;
    }
};
#endif
