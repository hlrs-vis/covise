/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRObjTransformMsg.h"

using namespace grmsg;

GRMSGEXPORT coGRObjTransformMsg::coGRObjTransformMsg(const char *obj_name,
                                                     float mat00, float mat01, float mat02, float mat03,
                                                     float mat10, float mat11, float mat12, float mat13,
                                                     float mat20, float mat21, float mat22, float mat23,
                                                     float mat30, float mat31, float mat32, float mat33)
    : coGRObjTransformAbstractMsg(TRANSFORM_OBJECT, obj_name,
                                  mat00, mat01, mat02, mat03, mat10, mat11, mat12, mat13,
                                  mat20, mat21, mat22, mat23, mat30, mat31, mat32, mat33)
{
}

GRMSGEXPORT coGRObjTransformMsg::coGRObjTransformMsg(const char *msg)
    : coGRObjTransformAbstractMsg(msg)
{
}
