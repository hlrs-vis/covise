/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRObjTransformMsg - stores TransformationMatrix of Object         ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJTRANSFORMMSG_H
#define COGROBJTRANSFORMMSG_H

#include "coGRObjTransformAbstractMsg.h"

namespace grmsg
{

class GRMSGEXPORT coGRObjTransformMsg : public coGRObjTransformAbstractMsg

{
public:
    // construct msg to send
    coGRObjTransformMsg(const char *obj_name, float mat00, float mat01, float mat02, float mat03, float mat10, float mat11, float mat12, float mat13, float mat20, float mat21, float mat22, float mat23, float mat30, float mat31, float mat32, float mat33);

    // reconstruct from received msg
    coGRObjTransformMsg(const char *msg);
};
}
#endif
