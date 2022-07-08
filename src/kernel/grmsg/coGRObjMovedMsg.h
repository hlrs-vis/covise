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

#ifndef COGROBJMOVEDMSG_H
#define COGROBJMOVEDMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjMovedMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjMovedMsg(const char *obj_name, float transX, float transY, float transZ, float rotX, float rotY, float rotZ, float rotAngle);

    // reconstruct from received msg
    coGRObjMovedMsg(const char *msg);

    // getter
    float getTransX() const
    {
        return transX_;
    };
    float getTransY() const
    {
        return transY_;
    };
    float getTransZ() const
    {
        return transZ_;
    };
    float getRotX() const
    {
        return rotX_;
    };
    float getRotY() const
    {
        return rotY_;
    };
    float getRotZ() const
    {
        return rotZ_;
    };
    float getRotAngle() const
    {
        return rotAngle_;
    };

protected:
    // construct msg to send (used by inherited classes to override type)
    coGRObjMovedMsg(coGRMsg::Mtype type, const char *obj_name, float transX, float transY, float transZ, float rotX, float rotY, float rotZ, float rotAngle);

private:
    void initClass(float transX, float transY, float transZ, float rotX, float rotY, float rotZ, float rotAngle);

    float transX_;
    float transY_;
    float transZ_;
    float rotX_;
    float rotY_;
    float rotZ_;
    float rotAngle_;
};
}
#endif
