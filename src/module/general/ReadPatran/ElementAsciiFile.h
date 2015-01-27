/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ELEMENTASCIIFILE_H_
#define _ELEMENTASCIIFILE_H_

#include <util/coviseCompat.h>
#include "istreamFTN.h"

#ifndef MAXLINE
#define MAXLINE 82
#endif

#define ASCMAXLINES 1000000

class ElementAscFile
{

private:
    istreamFTN in;

    enum
    {
        ELEMENTSTRESS = 3
    };

public:
    // shape code
    enum NSHAPE
    {
        BAR = 2,
        TRI,
        QUAD,
        TET,
        PYR,
        WEDG,
        HEX
    };

    int nwidth;
    int nnodes;

    struct DataRecord
    {
        int id;
        NSHAPE nshape;
        float data; // Result quantities organized by column index
    } dataTab[ASCMAXLINES];

    // Member functions
    ElementAscFile(const char *filename, int column);
    int isValid()
    {
        return (nnodes != 0);
    }

    int getDataField(int fieldFlag, const int *elemMap, int colNo, float *f1,
                     const int diff, const int maxelem);
};
#endif
