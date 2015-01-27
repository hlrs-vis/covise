/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSGVRUI_TRANSFORMNODE_H
#define OSGVRUI_TRANSFORMNODE_H

#include <OpenVRUI/osg/OSGVruiNode.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include <OpenVRUI/sginterface/vruiTransformNode.h>

#include <osg/MatrixTransform>

#ifdef _WIN32
#pragma warning(disable : 4250)
#endif

namespace vrui
{

class OSGVRUIEXPORT OSGVruiTransformNode : public virtual OSGVruiNode, public virtual vruiTransformNode
{

public:
    OSGVruiTransformNode(osg::MatrixTransform *node);
    virtual ~OSGVruiTransformNode();

    virtual void setScale(float scale);
    virtual void setScale(float scaleX, float scaleY, float scaleZ);

    virtual void setTranslation(float x, float y, float z);

    virtual void setRotation(float x, float y, float z, float angle);

    virtual vruiMatrix *getMatrix();
    virtual void setMatrix(vruiMatrix *matrix);

protected:
    osg::ref_ptr<osg::MatrixTransform> transform;

    OSGVruiMatrix matrix;
};
}
#endif
