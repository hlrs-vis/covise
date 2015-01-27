/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    Reducer
//
// Description: Class to remove unused coordinates from a DataContainer
//
// Initial version: 11.06.2003
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 / 2003 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  Changes:
//

#ifndef REDUCER_H
#define REDUCER_H

#include "EnFile.h"

class Reducer
{
public:
    // create obj with DataCont to be reduced
    // !!! dc will be manipulated !!!
    Reducer(DataCont &dc, int *im = NULL);

    // removed unused coordinates
    // return the number of unused coordinates
    int removeUnused(float **xn = NULL, float **yn = NULL, float **zn = NULL);

    int reduceData();
    DataCont reduceAndCopyData();

    const int *getIdxMap();

    /// DESTRUCTOR
    ~Reducer();

private:
    // this utility should not be copied or assigned
    Reducer(const Reducer &r);

    const Reducer &operator=(const Reducer &r);

    DataCont &dc_;
    int *idxMap_;
};

#endif
