/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "BillardBall.h"
#include <osg/ShapeDrawable>
#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>

BillardBall::BillardBall(osg::Matrix initialMat, float size, string geoFileName)
    : coVR3DTransRotInteractor(initialMat, size, coInteraction::ButtonA, "Menu", geoFileName.c_str(), coInteraction::Medium)
{

    geoFileName_ = geoFileName;
    createGeometry();
}

BillardBall::~BillardBall()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "BillardBall::~BillardBall\n");
}

void BillardBall::createGeometry()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "BillardBall::createGeometry\n");

    geometryNode = coVRFileManager::instance()->loadIcon(geoFileName_.c_str());
    osg::BoundingBox bb;
    bb = cover->getBBox(geometryNode);
    fprintf(stderr, "%s = %f\n", geoFileName_.c_str(), bb._max.x() - bb._min.x());

    // remove old geometry
    scaleTransform->removeChild(0, scaleTransform->getNumChildren());

    // add geometry to node
    scaleTransform->addChild(geometryNode.get());
}

void BillardBall::keepSize()
{
    osg::Matrix m;
    m.makeScale(6.0, 6.0, 6.0);
    scaleTransform->setMatrix(m);
}
