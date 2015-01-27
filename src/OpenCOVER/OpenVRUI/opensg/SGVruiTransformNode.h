/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SGVRUI_TRANSFORMNODE_H
#define SGVRUI_TRANSFORMNODE_H

#include <OpenVRUI/opensg/SGVruiNode.h>
#include <OpenVRUI/opensg/SGVruiMatrix.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <OpenSG/OSGComponentTransform.h>

class SGVRUIEXPORT SGVruiTransformNode : public virtual SGVruiNode, public virtual vruiTransformNode
{

public:
    SGVruiTransformNode(const osg::NodePtr &node);
    virtual ~SGVruiTransformNode();

    virtual void setScale(float scale);
    virtual void setScale(float scaleX, float scaleY, float scaleZ);
    virtual void setTranslation(float x, float y, float z);
    virtual void setRotation(float x, float y, float z, float angle);

    virtual vruiMatrix *getMatrix();
    virtual void setMatrix(vruiMatrix *matrix);

private:
    osg::ComponentTransformPtr transform;
    SGVruiMatrix matrix;
};
#endif
