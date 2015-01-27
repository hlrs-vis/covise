/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiTransformNode.h>

#include <OpenVRUI/util/vruiLog.h>

#include <osg/Vec3>

#include <string>

#include <math.h>

using namespace osg;

namespace vrui
{

OSGVruiTransformNode::OSGVruiTransformNode(MatrixTransform *node)
    : OSGVruiNode(node)
{
    transform = node;
}

OSGVruiTransformNode::~OSGVruiTransformNode()
{
}

void OSGVruiTransformNode::setScale(float scale)
{
    setScale(scale, scale, scale);
}

void OSGVruiTransformNode::setScale(float scaleX, float scaleY, float scaleZ)
{
    Matrix mat = transform->getMatrix();
    Vec3 scale = mat.getScale();

    if (scaleX == 0.0)
        return;
    if (scaleY == 0.0)
        return;
    if (scaleZ == 0.0)
        return;

    //VRUILOG(scaleX << " " << scaleY << " " << scaleZ)
    //VRUILOG(scale)

    Matrix sm;
    Matrix osm;
    sm.makeScale(scaleX / scale[0], scaleY / scale[1], scaleZ / scale[2]);
    mat *= sm;

    transform->setMatrix(mat);
}

void OSGVruiTransformNode::setTranslation(float x, float y, float z)
{
    Matrix mat = transform->getMatrix();
    mat.setTrans(x, y, z);
    transform->setMatrix(mat);
}

void OSGVruiTransformNode::setRotation(float x, float y, float z, float angle)
{

    Matrix mat = transform->getMatrix();

    Matrix sm;
    sm.makeScale(mat.getScale());

    Vec3 trans = mat.getTrans();

    mat.makeRotate(osg::inDegrees(angle), Vec3(x, y, z));
    mat.setTrans(trans);
    mat *= sm;

    transform->setMatrix(mat);
}

vruiMatrix *OSGVruiTransformNode::getMatrix()
{
    matrix.setMatrix(transform->getMatrix());
    return &matrix;
}

void OSGVruiTransformNode::setMatrix(vruiMatrix *matrix)
{

    //   if (this->matrix.getMatrix() != dynamic_cast<OSGVruiMatrix*>(matrix)->getMatrix())
    //     VRUILOG("OSGVruiTransformNode::setMatrix info: setting matrix of "
    //             << transform->getName() << " to " << this->matrix.getMatrix())
    //   else
    //     cerr << ".";

    this->matrix = *dynamic_cast<OSGVruiMatrix *>(matrix);
    transform->setMatrix(this->matrix.getMatrix());
}
}
