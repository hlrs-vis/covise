/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_GeoDict_H
#define _READ_GeoDict_H
// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadGeoDict
//
// This module Reads GeoDict formated ASCII files
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
};
class ReadGeoDict : public coReader
{
public:
    typedef struct
    {
        float x, y, z;
    } Vect3;

    typedef struct
    {
        bool vector;
        int imageNumber;
        std::string name;
        coDistributedObject **dataObjs;
        std::string objectName;
        float *x_d;
        float *y_d;
        float *z_d;
        bool read;

    } VarInfo;

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *paraName, bool inMapLoading);

    // ports


    // utility functions
    int readHeader();
    int readData();
    int nextLine();

    // already opened file, always rewound after use
    FILE *d_dataFile;
    std::string fileName;


    char buf[5000];
    char line[1024];
    char *currentLine;
    char *nextLineBuf;
    
    int Nx,Ny,Nz;
    float sx,sy,sz;

    std::vector<VarInfo> varInfos;


public:
    ReadGeoDict(int argc, char *argv[]);
    virtual ~ReadGeoDict();
};
#endif
