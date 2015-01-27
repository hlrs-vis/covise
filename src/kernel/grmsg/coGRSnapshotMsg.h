/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2009 VISENSO    ++
// ++ coGRSnapshotMsg - sends message to make a snapshot                  ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRSNAPSHOTMSG_H
#define COGRSNAPSHOTMSG_H

#include "coGRMsg.h"
#include <util/coExport.h>

#include <iostream>

namespace grmsg
{

class GRMSGEXPORT coGRSnapshotMsg : public coGRMsg
{
public:
    // construct msg to send
    coGRSnapshotMsg(const char *filename, const char *intention);

    // reconstruct from received msg
    coGRSnapshotMsg(const char *msg);

    virtual ~coGRSnapshotMsg();

    // specific functions
    const char *getFilename()
    {
        return filename_;
    };
    const char *getIntention()
    {
        return intention_;
    };

private:
    char *filename_;
    char *intention_;
};
}

#endif
