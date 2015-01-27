/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ELEMENTFILE_H_
#define _ELEMENTFILE_H_

#include <util/coviseCompat.h>
#include "istreamFTN.h"

#ifndef EMAXCOL
#define EMAXCOL 15 // maximun number of columns of data stored in element results file
#endif
#define MAXLINES 250000

/* PATRAN Element Results File (unformatted version) */
class ElementFile
{

private:
    istreamFTN input; // FORTRAN input stream (binary)

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

    struct Record1
    {
        char title[320];
        int nwidth; // Number of columns of data stored in the file
    } header;

    char subtitle1[320];
    char subtitle2[320];

    int numlines; // Number of lines of data

    struct DataRecord
    {
        int id;
        NSHAPE nshape;
        float data[EMAXCOL]; // Result quantities organized by column index
    } dataTab[MAXLINES];

    // Member functions
    ElementFile(int fd);
    int isValid()
    {
        return (numlines != 0);
    }

    int getDataField(int fieldFlag, const int *elemMap, int colNo, float *f1,
                     const int diff, const int maxelem);
};
#endif
