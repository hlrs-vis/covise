/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VSGVrui_TRANSFORMNODE_H
#define VSGVrui_TRANSFORMNODE_H

#include <OpenVRUI/vsg/VSGVruiNode.h>
#include <OpenVRUI/vsg/VSGVruiMatrix.h>

#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <vsg/nodes/MatrixTransform.h>

#ifdef _WIN32
#pragma warning(disable : 4250)
#endif

namespace vrui
{

class VSGVRUIEXPORT VSGVruiTransformNode : public virtual VSGVruiNode, public virtual vruiTransformNode
{

public:
    VSGVruiTransformNode(vsg::ref_ptr<vsg::MatrixTransform> node);
    virtual ~VSGVruiTransformNode();

    virtual void setScale(float scale);
    virtual void setScale(float scaleX, float scaleY, float scaleZ);

    virtual void setTranslation(float x, float y, float z);

    virtual void setRotation(float x, float y, float z, float angle);

    virtual vruiMatrix *getMatrix();
    virtual void setMatrix(vruiMatrix *matrix);

protected:
    vsg::ref_ptr<vsg::MatrixTransform> transform;

    VSGVruiMatrix matrix;
};
}
#endif
