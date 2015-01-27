/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRObjKeyWordMsg - transmits a key word                            ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRKEYWORDMSG_H
#define COGRKEYWORDMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRKeyWordMsg : public coGRMsg
{
public:
    // construct msg to send
    coGRKeyWordMsg(const char *keyWord, bool kw);

    // reconstruct from received msg
    coGRKeyWordMsg(const char *msg);

    virtual ~coGRKeyWordMsg();

    // specific functions
    const char *getKeyWord()
    {
        return keyWord_;
    };

private:
    char *keyWord_;
};
}
#endif
