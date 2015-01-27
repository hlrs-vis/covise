/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRSetDocPositionMsg - set page of opened document in COVER            ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRSETDOCPOSITIONMSG_H
#define COGRSETDOCPOSITIONMSG_H

#include "coGRDocMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRSetDocPositionMsg : public coGRDocMsg
{
public:
    // construct msg to send
    coGRSetDocPositionMsg(const char *document_name, float x, float y, float z);

    // reconstruct from received msg
    coGRSetDocPositionMsg(const char *msg);

    // specific functions
    void getPosition(float &x, float &y, float &z);

private:
    float pos_[3];
};
}
#endif
