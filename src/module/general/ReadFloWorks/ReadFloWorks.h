/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// MODULE   ReadFloWorks
//
// Description: New Technology Ensight read-module
//
// Initial version: 15.04.2002
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef READFLOWWORKS_H
#define READFLOWORKS_H

#include <reader/coReader.h>
#include <reader/ReaderControl.h>
#include "FloWorks.h"

#ifdef __sgi
using namespace std;
#endif

const int Success(0);
const int Failure(1);

class ReadFloWorks : public coReader
{

public:
    /// default CONSTRUCTOR
    ReadFloWorks();

    /// DESTRUCTOR
    ~ReadFloWorks();

    virtual void param(const char *paramName);

    /// compute call-back
    virtual int compute(const char *port);

private:
    int readGeometry(const int &portTok);

    int readScalarData(const int &portTok);

    int readVectorData(const int &portTok);

    FloWorks *flow_;
    DataList dl_;
};
#endif
