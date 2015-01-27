/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NODALFILE_H_
#define _NODALFILE_H_

#include <api/coModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include "istreamFTN.h"

#ifndef NMAXCOL
#define NMAXCOL 82
#endif

#ifndef NMAXEL
#define NMAXEL 200
#endif

#ifndef NDISPLACEMENTS
#define NDISPLACEMENTS 2
#endif

#ifndef NNODALSTRESS
#define NNODALSTRESS 1
#endif

#define NASCII 1
#define NBINARY 2

/* PATRAN Nodal Results Data File  */

class NodalFile
{

private:
    istreamFTN input; // FORTRAN input stream (binary)

public:
    struct Record1
    {
        char title[320];
        int nnodes; // Number of nodes
        int maxnod; // Highest node ID number
        float defmax; // Maximum absolute displacement
        int ndmax; // ID of node where maximum displacement occurs
        int nwidth; // Number of columns for nodal information
    } header;

    char subtitle1[320];
    char subtitle2[320];

    int nnodes; // Number of data nodes

    struct DataRecord
    {
        int nodid; // Node ID number
        float data[NMAXEL]; // Result quantities organized by column index
    } *dataTab;

    // Member functions
    NodalFile(const char *filename, int filetype);
    ~NodalFile();
    int isValid()
    {
        return (nnodes != 0);
    }

    int getDataField(int fieldFlag, const int *nodeMap,
                     float *f1, float *f2, float *f3,
                     int diff, const int maxnode);
    int getDataField(int fieldFlag, const int *nodeMap,
                     int colNo, float *f1, int diff, const int maxnode);
};
#endif
