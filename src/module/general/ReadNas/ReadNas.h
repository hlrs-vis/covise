/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_NAS_H
#define _READ_NAS_H

// +++++++++++++++++++++++++++++++++++++++++
// MODULE ReadNas
//
// This module interpolates data values from Cell to Vertex
// based data representation
//
// Initial version: 2002-07-17 we
// +++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++
// Changes:
#include <util/coviseCompat.h>
#include <api/coModule.h>
using namespace covise;

class ReadNas : public coModule
{
public:
    typedef struct
    {
        float x, y, z;
    } Vect3;

    

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *paraName, bool inMapLoading);

    // ports
    coOutputPort *p_polyOut;

    // parameter
    coFileBrowserParam *p_filename;

    // utility functions
    int readASCII();

    // covise-specific calls are
    // as far as possible lumped together in this function
    void outputObjects(vector<float> &x, vector<float> &y, vector<float> &z,
                       vector<int> &connList, vector<int> &elemList);

    // already opened file, alway rewound after use
    FILE *d_file;

public:
    
    float readFloat(char *buf, int pos);
    ReadNas(int argc, char *argv[]);
    virtual ~ReadNas();
};
#endif
