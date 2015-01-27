/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _WRITEABAQUSINP_H
#define _WRITEABAQUSINP_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2009 HLRS ++
// ++                                                                     ++
// ++ Description: Program for writing iso-surfaces in ABAQUS .inp format ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Ralf Schneider                           ++
// ++               High Performance Computing Center Stuttgart           ++
// ++                           Nobelstrasse 19                           ++
// ++                           70569 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  25.02.2009  V1.0                                             ++
// ++**********************************************************************/

#include <api/coSimpleModule.h>

using namespace covise;

class WriteAbaqusInp : public coSimpleModule
{

private:
    //////////  member functions

    /// this module has only the compute call-back
    virtual int compute(const char *port);

    ////////// parameters
    coFileBrowserParam *p_outFile;

    ////////// ports
    coInputPort *p_inPort;

public:
    WriteAbaqusInp(int argc, char *argv[]);
};
#endif
