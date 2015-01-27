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

#ifndef COGROBJGEOMETRYMSG_H
#define COGROBJGEOMETRYMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjGeometryMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjGeometryMsg(const char *obj_name, float width, float height, float length);

    // reconstruct from received msg
    coGRObjGeometryMsg(const char *msg);

    // getter
    float getWidth()
    {
        return width_;
    };
    float getHeight()
    {
        return height_;
    };
    float getLength()
    {
        return length_;
    };

protected:
private:
    float width_;
    float height_;
    float length_;
};
}
#endif
