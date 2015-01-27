/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// MODULE   RWCoviseBlock
//
// Description:
//
// Initial version: 04.2007
// Ported: 05.2009
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2009 by VISENSO GmbH
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//
#ifndef _RWCOVISE_BLOCK_H
#define _RWCOVISE_BLOCK_H

#include "covise/covise.h"
#include <api/coModule.h>
using namespace covise;

class RWCoviseBlock : public coModule
{

public:
    RWCoviseBlock(int argc, char *argv[]);

private:
    // compute callback
    virtual int compute(const char *port);

    coFileBrowserParam *outFileParam_;
    coBooleanParam *readFinishedParam_;
    coInputPort *gridInPort_;
    coInputPort *dataInPort_;
};

#endif
