/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRObjTransformSGItemMsg - stores TransformationMatrix of a        ++
// ++                             SceneGraphItem                          ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJTRANSFORMSGITEMMSG_H
#define COGROBJTRANSFORMSGITEMMSG_H

#include "coGRObjTransformAbstractMsg.h"

namespace grmsg
{

class GRMSGEXPORT coGRObjTransformSGItemMsg : public coGRObjTransformAbstractMsg
{
public:
    // construct msg to send
    coGRObjTransformSGItemMsg(const char *obj_name, float mat00, float mat01, float mat02, float mat03, float mat10, float mat11, float mat12, float mat13, float mat20, float mat21, float mat22, float mat23, float mat30, float mat31, float mat32, float mat33);

    // reconstruct from received msg
    coGRObjTransformSGItemMsg(const char *msg);
};
}
#endif
