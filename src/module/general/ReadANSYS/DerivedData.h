/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DERIVED_DATA_H_
#define _DERIVED_DATA_H_

#include <vector>

struct DerivedData
{
public:
    int anz_; // Anzahl Elemente in data
    std::vector<double> *data_; // Pointer auf daten
    DerivedData()
    {
        data_ = NULL;
    }
    ~DerivedData()
    {
        clean();
    }

private:
    void clean()
    {
        delete[] data_;
        data_ = NULL;
    }
};
#endif
