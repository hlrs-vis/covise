/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRObjSetNameMsg - sets the name of an object                      ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJSETNAMEMSG_H
#define COGROBJSETNAMEMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjSetNameMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjSetNameMsg(Mtype type, const char *obj_name, const char *newName);

    // reconstruct from received msg
    coGRObjSetNameMsg(const char *msg);

    // destructor
    ~coGRObjSetNameMsg();

    // specific functions
    const char *getNewName()
    {
        return newName_;
    };

private:
    char *newName_;
};
}
#endif
