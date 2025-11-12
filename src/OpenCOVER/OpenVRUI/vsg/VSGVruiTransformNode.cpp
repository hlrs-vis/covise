/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiTransformNode.h>

#include <OpenVRUI/util/vruiLog.h>

#include <vsg/all.h>

#include <string>

#include <math.h>
#include "mathUtils.h"

using namespace vsg;

namespace vrui
{

VSGVruiTransformNode::VSGVruiTransformNode(vsg::ref_ptr<vsg::MatrixTransform> node)
    : VSGVruiNode(node)
{
    transform = node;
}

VSGVruiTransformNode::~VSGVruiTransformNode()
{
}

void VSGVruiTransformNode::setScale(float scale)
{
    setScale(scale, scale, scale);
}

void VSGVruiTransformNode::setScale(float scaleX, float scaleY, float scaleZ)
{

    if (scaleX == 0.0)
        return;
    if (scaleY == 0.0)
        return;
    if (scaleZ == 0.0)
        return;

    transform->matrix[0][0] = scaleX;
    transform->matrix[1][1] = scaleY;
    transform->matrix[2][2] = scaleZ;
}

void VSGVruiTransformNode::setTranslation(float x, float y, float z)
{
    setTrans(transform->matrix,dvec3(x,y,z));
}

void VSGVruiTransformNode::setRotation(float x, float y, float z, float angle)
{

    
    dmat4 mat = rotate(radians(double(angle)), double(x), double(y), double(z));
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            transform->matrix[i][j] = mat[i][j];

}

vruiMatrix *VSGVruiTransformNode::getMatrix()
{
    matrix.setMatrix(transform->matrix);

    return &matrix;
}

void VSGVruiTransformNode::setMatrix(vruiMatrix *matrix)
{

    this->matrix = *dynamic_cast<VSGVruiMatrix *>(matrix);
    transform->matrix = this->matrix.getMatrix();
}
}
