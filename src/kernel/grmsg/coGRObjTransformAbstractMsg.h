/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRObjTransformAbstractMsg - abstract class used for all           ++
// ++                               transform-related messages            ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJTRANSFORMABSTRACTMSG_H
#define COGROBJTRANSFORMABSTRACTMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjTransformAbstractMsg : public coGRObjMsg
{
public:
    // get Matrix
    float getMatrix(int i, int j)
    {
        return mat_[i][j];
    };

protected:
    // construct msg to send
    coGRObjTransformAbstractMsg(coGRMsg::Mtype type, const char *obj_name, float mat00, float mat01, float mat02, float mat03, float mat10, float mat11, float mat12, float mat13, float mat20, float mat21, float mat22, float mat23, float mat30, float mat31, float mat32, float mat33);

    // reconstruct from received msg
    coGRObjTransformAbstractMsg(const char *msg);

private:
    float mat_[4][4];
};
}
#endif
