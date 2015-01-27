/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_MOLDFLOW_H
#define _READ_MOLDFLOW_H m

// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadMoldFlow
//
//
// Initial version: 2003-09-01 Uwe
// +++++++++++++++++++++++++++++++++++++++++
// (C) 2003 by Uwe Woessner
// +++++++++++++++++++++++++++++++++++++++++
// Changes:

#include <util/coviseCompat.h>

#include <api/coModule.h>
using namespace covise;

#define TYPE_PATRAN 0
#define TYPE_XML 1
class coDoFloat;

class ReadMoldFlow : public coModule
{
public:
private:
    //  member functions
    virtual int compute();
    virtual void param(const char *paraName);
    void openFiles();
    int readASCII();

    // ports
    coOutputPort *p_polyOut;
    coOutputPort *p_resultOut;

    // parameter
    coFileBrowserParam *p_nodeFilename;
    coFileBrowserParam *p_resultFilename;
    coFileBrowserParam *p_elementFilename;
    coIntScalarParam *p_numT;

    // utility functions
    // already opened file, alway rewound after use
    FILE *d_resultFile;
    FILE *d_nodeFile;
    FILE *d_elementFile;
    int dataType;
    int numt;
    int n_coord;
    int n_elem;
    coDoFloat *readResults(const char *objName);

public:
    ReadMoldFlow();
    virtual ~ReadMoldFlow();
};
#endif
