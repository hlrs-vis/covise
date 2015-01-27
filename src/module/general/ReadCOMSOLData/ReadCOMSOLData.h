/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_COMSOLData_H
#define _READ_COMSOLData_H
// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadCOMSOLData
//
// This module Reads COMSOL COMSOLData files
// export as Sectionwise *.txt file
// export cartesian grids only
// select timesteps and data types to export
// then rightclick->export
//
#include <util/coviseCompat.h>
#include <api/coModule.h>
#include "reader/coReader.h"
#include "reader/ReaderControl.h"
using namespace covise;

enum
{
    TXT_BROWSER,
    MESHPORT3D,
    DPORT1_3D,
    DPORT2_3D,
    DPORT3_3D,
    DPORT4_3D,
    DPORT5_3D,
    GEOPORT2D,
    DPORT1_2D,
    DPORT2_2D,
    DPORT3_2D,
    DPORT4_2D,
    DPORT5_2D,
    GEOPORT1D,
    DPORT1_1D,
    DPORT2_1D,
    DPORT3_1D,
    DPORT4_1D,
    DPORT5_1D,
};
class ReadCOMSOLData : public coReader
{
public:
    typedef struct
    {
        float x, y, z;
    } Vect3;

    typedef struct
    {
        int components;
        std::string name;
        int valueIndex;
        coDistributedObject **dataObjs;
        std::string objectName;
    } VarInfo;

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *paraName, bool inMapLoading);

    // ports

    // utility functions
    int readHeader();
    int readASCIIData();

    // already opened file, alway rewound after use
    FILE *d_dataFile;
    std::string fileName;

    void nextLine(FILE *fp);

    char buf[1000];
    int numPoints;
    int dimentions;
    int numElementTypes;
    int numElements;
    int numExpressions;
    int numTimesteps;
    int numScalars;

    std::vector<VarInfo> varInfos;

    int *v_l_tri;
    int *v_l_quad;
    float *xCoords;
    float *yCoords;
    float *zCoords;

public:
    ReadCOMSOLData(int argc, char *argv[]);
    virtual ~ReadCOMSOLData();
};
#endif
