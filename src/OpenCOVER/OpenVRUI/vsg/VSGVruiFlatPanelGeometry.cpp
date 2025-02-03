/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiFlatPanelGeometry.h>
#include <OpenVRUI/vsg/VSGVruiTransformNode.h>
#include <OpenVRUI/vsg/VSGVruiPresets.h>

#include <OpenVRUI/coFlatPanelGeometry.h>

#include <OpenVRUI/coFlatPanelGeometry.h>

#include <vsg/all.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace vsg;

namespace vrui
{

float VSGVruiFlatPanelGeometry::A = 12.0f;
float VSGVruiFlatPanelGeometry::B = 12.0f;

VSGVruiFlatPanelGeometry::VSGVruiFlatPanelGeometry(coFlatPanelGeometry *geometry)
    : vruiPanelGeometryProvider(geometry)
{

    panelGeometry = geometry;
}

VSGVruiFlatPanelGeometry::~VSGVruiFlatPanelGeometry()
{
}

vsg::ref_ptr<vsg::Node> VSGVruiFlatPanelGeometry::createQuad(const vsg::vec3& origin, const vsg::vec3& horizontal, const vsg::vec3& vertical)
{

    auto builder = vsg::Builder::create();
    //builder->options = options;

    vsg::GeometryInfo geomInfo;
    geomInfo.position = origin;
    geomInfo.dx = horizontal;
    geomInfo.dy = vertical;
    geomInfo.dz.set(0.0f, 0.0f, 1.0f);

    vsg::StateInfo stateInfo;

    return builder->createQuad(geomInfo, stateInfo);
}

void VSGVruiFlatPanelGeometry::attachGeode(vruiTransformNode* node)
{

    VSGVruiTransformNode* transform = dynamic_cast<VSGVruiTransformNode*>(node);
    if (transform)
    {
        dynamic_cast<vsg::MatrixTransform *>(transform->getNodePtr())->addChild(createQuad(vsg::vec3(0.0f, 0.0f, 0.0f), vsg::vec3(A, 0.0f, 0.0f), vsg::vec3(0.0f, B, 0.0f)));
    }
    else
    {
        VRUILOG("OSGVruiFlatPanelGeometry::attachGeode: err: node to attach to is no transform node")
    }
}

float VSGVruiFlatPanelGeometry::getWidth() const
{
    return A;
}

float VSGVruiFlatPanelGeometry::getHeight() const
{
    return B;
}

float VSGVruiFlatPanelGeometry::getDepth() const
{
    return 0;
}
}
