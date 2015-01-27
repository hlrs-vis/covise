/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__ISSGRID_H)
#define __ISSGRID_H

#include <appl/ApplInterface.h>
using namespace covise;

class ISSGrid;

class ISSGrid
{
protected:
    int validFlag;

    int gridType;
    coDistributedObject *gridPtr;
    int numI, numJ, numK;
    float *dataPtr;

    enum
    {
        UNIGRD
    };

public:
    ISSGrid(coDistributedObject *grid, coDistributedObject *data);
    ~ISSGrid();

    int isValid()
    {
        return (validFlag);
    };

    void getSlice(int i, float *x, float *y, float *z, float *d);
    void getDimensions(int *i, int *j, int *k);
};
#endif // __ISSGRID_H
