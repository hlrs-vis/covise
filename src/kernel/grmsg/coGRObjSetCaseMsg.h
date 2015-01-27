/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRObjSetCaseMsg - stores the case name for an object              ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJSETCASEMSG_H
#define COGROBJSETCASEMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjSetCaseMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjSetCaseMsg(Mtype type, const char *obj_name, const char *caseName);

    // reconstruct from received msg
    coGRObjSetCaseMsg(const char *msg);

    // destructor
    ~coGRObjSetCaseMsg();

    // specific functions
    const char *getCaseName()
    {
        return caseName_;
    };

private:
    char *caseName_;
};
}
#endif
