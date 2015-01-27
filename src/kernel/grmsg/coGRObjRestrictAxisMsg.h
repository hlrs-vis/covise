/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRObjRestrictAxisMsg - stores visibility information for an object++
// ++                 visibility type can be: geometry, interactor etc.   ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJRESTRICTAXISMSG_H
#define COGROBJRESTRICTAXISMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjRestrictAxisMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjRestrictAxisMsg(Mtype type, const char *obj_name, const char *axisName);

    // reconstruct from received msg
    coGRObjRestrictAxisMsg(const char *msg);

    // destructor
    ~coGRObjRestrictAxisMsg();

    // specific functions
    const char *getAxisName()
    {
        return axisName_;
    };

private:
    char *axisName_;
};
}

#endif
