/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// CLASS    coReader
//
// Description: general base class for COVISE read-modules
//
// Initial version: April 2002
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef CO_READER_H
#define CO_READER_H

#include <covise/covise.h>

#include <api/coModule.h>
//#include <api/coFileBrowserParam.h>

namespace covise
{

class READEREXPORT coReader : public coModule
{
public:
    /// default CONSTRUCTOR
    coReader(int argc, char *argv[], const string &desc = string(""));

    /// DESTRUCTOR
    virtual ~coReader();

private:
    vector<coFileBrowserParam *> fileBrowsers_;
    vector<coOutputPort *> outPorts_;

    /// ports
    coOutputPort **pOutPorts_;
};
}
#endif
