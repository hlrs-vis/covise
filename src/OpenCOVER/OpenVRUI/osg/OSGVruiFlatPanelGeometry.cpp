/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiFlatPanelGeometry.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>

#include <OpenVRUI/coFlatPanelGeometry.h>

#include <osg/PrimitiveSet>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/StateSet>

#include <OpenVRUI/util/vruiLog.h>

using namespace osg;

namespace vrui
{

float OSGVruiFlatPanelGeometry::A = 12.0f;
float OSGVruiFlatPanelGeometry::B = 12.0f;

ref_ptr<Vec3Array> OSGVruiFlatPanelGeometry::coord = 0;
//ref_ptr<Vec4Array> OSGVruiFlatPanelGeometry::color  = 0;
ref_ptr<Vec3Array> OSGVruiFlatPanelGeometry::normal = 0;

OSGVruiFlatPanelGeometry::OSGVruiFlatPanelGeometry(coFlatPanelGeometry *geometry)
    : vruiPanelGeometryProvider(geometry)
{

    panelGeometry = geometry;
    backgroundMaterial = panelGeometry->getBackgroundMaterial();
}

OSGVruiFlatPanelGeometry::~OSGVruiFlatPanelGeometry()
{
}

void OSGVruiFlatPanelGeometry::createSharedLists()
{
    if (coord == 0)
    {
        //color    = new Vec4Array(1);
        coord = new Vec3Array(4);
        normal = new Vec3Array(1);

        (*coord)[0].set(0.0f, B, 0.0f);
        (*coord)[1].set(0.0f, 0.0f, 0.0f);
        (*coord)[2].set(A, 0.0f, 0.0f);
        (*coord)[3].set(A, B, 0.0f);

        //(*color)[0]->set(1.0f, 1.0f, 1.0f, 1.0f);

        (*normal)[0].set(0.0f, 0.0f, 1.0f);
    }
}

void OSGVruiFlatPanelGeometry::attachGeode(vruiTransformNode *node)
{
    createSharedLists();

    ref_ptr<Geometry> geometry = new Geometry();
    //geoset1->setAttr(PFGS_COLOR4, PFGS_OVERALL,    color,  NULL);
    geometry->setVertexArray(coord.get());
    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geometry->setNormalArray(normal.get());
    geometry->setNormalBinding(Geometry::BIND_OVERALL);

    geometry->setStateSet(OSGVruiPresets::getStateSet(backgroundMaterial));

    ref_ptr<Geode> geode = new Geode();
    geode->addDrawable(geometry.get());

    OSGVruiTransformNode *transform = dynamic_cast<OSGVruiTransformNode *>(node);
    if (transform)
    {
        transform->getNodePtr()->asGroup()->addChild(geode.get());
    }
    else
    {
        VRUILOG("OSGVruiFlatPanelGeometry::attachGeode: err: node to attach to is no transform node")
    }
}

float OSGVruiFlatPanelGeometry::getWidth() const
{
    return A;
}

float OSGVruiFlatPanelGeometry::getHeight() const
{
    return B;
}

float OSGVruiFlatPanelGeometry::getDepth() const
{
    return 0;
}
}
