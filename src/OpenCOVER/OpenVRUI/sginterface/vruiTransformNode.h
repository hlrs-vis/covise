/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_TRANSFORMNODE_H
#define VRUI_TRANSFORMNODE_H

#include <OpenVRUI/sginterface/vruiNode.h>
#include <OpenVRUI/sginterface/vruiMatrix.h>

namespace vrui
{

class OPENVRUIEXPORT vruiTransformNode : public virtual vruiNode
{

public:
    vruiTransformNode()
    {
    }
    virtual ~vruiTransformNode()
    {
    }

    virtual void setScale(float scale) = 0;
    virtual void setScale(float scaleX, float scaleY, float scaleZ) = 0;
    virtual void setTranslation(float x, float y, float z) = 0;
    virtual void setRotation(float x, float y, float z, float angle) = 0;

    virtual vruiMatrix *getMatrix() = 0;
    virtual void setMatrix(vruiMatrix *matrix) = 0;
};
}
#endif
