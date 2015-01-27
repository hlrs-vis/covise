/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiTransformNode.h>

OSG_USING_NAMESPACE

SGVruiTransformNode::SGVruiTransformNode(const osg::NodePtr &node)
    : SGVruiNode(node)
{
    transform = ComponentTransformPtr::dcast(node->getCore());
}

SGVruiTransformNode::~SGVruiTransformNode()
{
}

void SGVruiTransformNode::setScale(float scale)
{
    setScale(scale, scale, scale);
}

void SGVruiTransformNode::setScale(float scaleX, float scaleY, float scaleZ)
{
    beginEditCP(transform, ComponentTransform::ScaleFieldMask);
    transform->setScale(Vec3f(scaleX, scaleY, scaleZ));
    endEditCP(transform, ComponentTransform::ScaleFieldMask);
}

void SGVruiTransformNode::setTranslation(float X, float Y, float Z)
{
    beginEditCP(transform, ComponentTransform::TranslationFieldMask);
    transform->setTranslation(Vec3f(X, Y, Z));
    endEditCP(transform, ComponentTransform::TranslationFieldMask);
}

void SGVruiTransformNode::setRotation(float x, float y, float z, float angle)
{
    Quaternion q;
    q.setValueAsAxisDeg(x, y, z, angle);
    beginEditCP(transform, ComponentTransform::RotationFieldMask);
    transform->setRotation(q);
    endEditCP(transform, ComponentTransform::RotationFieldMask);
}

vruiMatrix *SGVruiTransformNode::getMatrix()
{
    matrix.setMatrix(transform->getMatrix());
    return &matrix;
}

void SGVruiTransformNode::setMatrix(vruiMatrix *matrix)
{

    //   if (this->matrix.getMatrix() != dynamic_cast<OSGVruiMatrix*>(matrix)->getMatrix())
    //     VRUILOG("OSGVruiTransformNode::setMatrix info: setting matrix of "
    //             << transform->getName() << " to " << this->matrix.getMatrix())
    //   else
    //     cerr << ".";

    this->matrix = *dynamic_cast<SGVruiMatrix *>(matrix);
    beginEditCP(transform, ComponentTransform::MatrixFieldMask);
    transform->setMatrix(this->matrix.getFloatMatrix());
    endEditCP(transform, ComponentTransform::MatrixFieldMask);
}
